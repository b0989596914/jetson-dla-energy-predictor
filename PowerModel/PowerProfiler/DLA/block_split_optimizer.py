"""
針對 ONNX 模型進行 切分與最佳化 
-> 只是從一個沒有被切分過的onnx檔案，用identity切割成幾個block的onnx檔案
指令 -> run_all_blockopt.sh
"""
import onnx
from onnx import helper, shape_inference, NodeProto, GraphProto
from typing import List, Tuple
import argparse
import copy
import json

# ============================
# 讀取 pattern json
# ============================
def load_patterns(json_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)
    return cfg["MIN_BLOCK_PATTERNS"]

# ============================
# DLA 可支援的 ops
# ============================
DLA_SUPPORTED_OPS = {
    "Conv", "Relu", "BatchNormalization", "Add", "Mul", "Clip",
    "Concat", "Resize", "Pad", "AveragePool", "MaxPool",
    "Sum", "Sub", "Div", "Pow", "Max", "Min","Gemm","Constant",
    "LRN", "Scale", "Slice", "Softmax", "Abs", "Sin", "Cos", "Atan"
}

# ============================
# DLA 支援判斷
# ============================
def is_dla_supported_node(node: NodeProto) -> bool:
    return node.op_type in DLA_SUPPORTED_OPS

# ============================
# Pattern 比對
# ============================
def match_pattern(nodes: List[NodeProto], start_idx: int, pattern: List[str]) -> bool:
    if start_idx + len(pattern) > len(nodes):
        return False
    for i in range(len(pattern)):
        if nodes[start_idx + i].op_type != pattern[i]:
            return False
    return True

# ============================
# Block 切割
# ============================
def partition_into_blocks(nodes: List[NodeProto], patterns: List[List[str]]) -> List[List[NodeProto]]:
    i, blocks = 0, []
    while i < len(nodes):
        matched = False
        # pattern 按照長度長到短排序
        for pattern in sorted(patterns, key=lambda x: -len(x)):
            if match_pattern(nodes, i, pattern):
                blocks.append(nodes[i:i+len(pattern)])
                i += len(pattern)
                matched = True
                break
        if not matched:
            blocks.append([nodes[i]])
            i += 1
    return blocks

# ============================
# Block 合併 (避免超過 DLA block 限制)
# ============================
def merge_blocks_strictly_to_target(blocks: List[List[NodeProto]], target: int) -> List[List[NodeProto]]:
    total_layers = sum(len(b) for b in blocks)
    avg_layers = total_layers // target
    extra = total_layers % target

    merged = []
    temp = []
    count = 0

    for block in blocks:
        temp.extend(block)
        count += len(block)
        threshold = avg_layers + (1 if extra > 0 else 0)
        if count >= threshold:
            merged.append(temp)
            temp = []
            count = 0
            if extra > 0:
                extra -= 1

    if temp:
        merged.append(temp)

    return merged

# ============================
# 依 DLA 支援切成 GPU / DLA 區塊
# ============================
def split_nodes_by_dla_support(nodes: List[NodeProto]):
    result = []
    group = []
    current_type = None

    for node in nodes:
        is_dla = is_dla_supported_node(node)

        if current_type is None:
            current_type = is_dla

        if is_dla == current_type:
            group.append(node)
        else:
            result.append((current_type, group))
            group = [node]
            current_type = is_dla

    if group:
        result.append((current_type, group))

    return result

# ============================
# 插入 Identity、重建模型
# ============================
def apply_identity_and_rebuild(graph: GraphProto, blocks):
    new_nodes = []
    output_map = {}
    tensor_consumers = {}

    for node in graph.node:
        for idx, input_name in enumerate(node.input):
            tensor_consumers.setdefault(input_name, []).append((node, idx))

    for block_idx, block in enumerate(blocks):
        if not block:
            continue

        for node in block:
            new_node = copy.deepcopy(node)
            for i in range(len(new_node.input)):
                if new_node.input[i] in output_map:
                    new_node.input[i] = output_map[new_node.input[i]]
            new_nodes.append(new_node)

        # 不是最後一個 block → 插 Identity
        if block_idx < len(blocks) - 1:
            last_node = block[-1]
            for output_name in last_node.output:
                consumers = tensor_consumers.get(output_name, [])
                if any(idx == 0 for (n, idx) in consumers):
                    identity_output = f"{output_name}_block{block_idx}_id"
                    identity_node = helper.make_node(
                        "Identity",
                        inputs=[output_name],
                        outputs=[identity_output],
                        name=f"Identity_block_{block_idx}"
                    )
                    output_map[output_name] = identity_output
                    new_nodes.append(identity_node)

    del graph.node[:]
    graph.node.extend(new_nodes)


# ============================
# 主函式：整體流程
# ============================
def optimize_onnx_blocks(input_path, output_path, max_dla_blocks, patterns):

    onnx_model = onnx.load(input_path)
    graph = onnx_model.graph
    nodes = list(graph.node)

    # 先拆成 GPU / DLA 區段
    segments = split_nodes_by_dla_support(nodes)

    all_dla_blocks = []
    gpu_blocks = []

    for is_dla, group in segments:
        if is_dla:
            sub_blocks = partition_into_blocks(group, patterns)
            all_dla_blocks.extend(sub_blocks)
        else:
            gpu_blocks.extend([[node] for node in group])

    print(f"[INFO] 初始 DLA block 數：{len(all_dla_blocks)}")

    if len(all_dla_blocks) > max_dla_blocks:
        print(f"[WARN] DLA block 超過 {max_dla_blocks}，開始合併")
        all_dla_blocks = merge_blocks_strictly_to_target(all_dla_blocks, max_dla_blocks)

    print(f"[INFO] 合併後 DLA block：{len(all_dla_blocks)}")

    all_blocks = all_dla_blocks + gpu_blocks

    print("[INFO] Block 統計：")
    for idx, blk in enumerate(all_blocks):
        ops = [n.op_type for n in blk]
        print(f"  Block {idx:3d} → {len(blk)} layers | {ops}")

    apply_identity_and_rebuild(graph, all_blocks)

    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)
    print(f"[INFO] ✓ 儲存：{output_path}")


# ============================
# Main
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', required=True)
    parser.add_argument('--output_path', default="optimized.onnx")
    parser.add_argument('--max_dla_blocks', type=int, default=16)
    parser.add_argument('--pattern_json', required=True)
    args = parser.parse_args()

    patterns = load_patterns(args.pattern_json)

    optimize_onnx_blocks(
        input_path=args.onnx_path,
        output_path=args.output_path,
        max_dla_blocks=args.max_dla_blocks,
        patterns=patterns
    )

import onnx

model = onnx.load("./onnx/MobileNetV2/bs1/MobileNetV2_imagenet100_bs1.onnx")
for i, node in enumerate(model.graph.node[:100]):
    print(f"{i}: {node.op_type} | {node.name} | inputs: {node.input} | outputs: {node.output}")
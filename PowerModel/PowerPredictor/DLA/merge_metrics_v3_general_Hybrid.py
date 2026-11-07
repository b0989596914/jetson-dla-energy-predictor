'''
這個檔案是新的：通用模型的結果

predictor後先把所有block的資料合併 -> 存去 new_general_Hybrid_universal 資料夾中
存過去之後可以做兩件事：
1. 畫比較圖 (plot_combine...)
2. 找出表現最好的預測器 (find_best_predictor.py)

python merge_metrics_v3_general_Hybrid.py --model VGG16 --preset A
python merge_metrics_v3_general_Hybrid.py --model ResNet50 --preset A
python merge_metrics_v3_general_Hybrid.py --model VGG19 --preset A
python merge_metrics_v3_general_Hybrid.py --model MobileNetV2 --preset A

'''
import os
import argparse
import re
import csv
from typing import Dict, List, Tuple

# 可用的模型和方法
SUPPORTED_MODELS = ["VGG16", "VGG19", "ResNet50", "MobileNetV2"]
PREDICTOR_METHODS = ["lasso", "xgboost", "mlp", "lightgbm", "extratree","vote_ensemble","stack_ensemble"]

# 檔名設定
TRAINING_ROOT_DIR_NAME = "Universal_Results"
NEW_OUTPUT_DIR = "new_general_Hybrid_universal"

# TRAINING_ROOT_DIR_NAME = "Universal_Results_group_0412presetB_1"
# NEW_OUTPUT_DIR = "new_general_Hybrid_universal_0412presetB_1"

# 型別
FileEntry = Tuple[int, str, str]   # (block_index, metrics.txt 路徑, 資料夾名稱)

def _to_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x.lower() in ["nan", "none", ""]:
        return None
    return float(x)

def parse_metrics_file(filepath: str) -> dict:
    data = {
        "model_name": None,
        "block_index": None,
        "train_mae": None, "train_rmse": None, "train_r2": None, "train_smape": None, "train_mape": None,
        "test_mae": None,  "test_rmse": None,  "test_r2": None,  "test_smape": None,  "test_mape": None,
        "unseen_mae": None,"unseen_rmse": None,"unseen_r2": None,"unseen_smape": None,"unseen_mape": None,
        "unseen_stddev": None,
    }

    if not os.path.exists(filepath):
        return data

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    model_re = re.compile(r"Model Name:\s*(.+)")
    block_re = re.compile(r"Unseen Block:\s*(\d+)")

    # 支援有無 MAPE 的版本
    metrics_re = re.compile(
        r"(Train|Test|Unseen)\s+Metrics:\s*"
        r"MAE=([0-9.eE+-]+|nan|NaN),\s*"
        r"RMSE=([0-9.eE+-]+|nan|NaN),\s*"
        r"R2=([0-9.eE+-]+|nan|NaN),\s*"
        r"SMAPE=([0-9.eE+-]+|nan|NaN)%"
        r"(?:,\s*MAPE=([0-9.eE+-]+|nan|NaN)%)?" # MAPE 為選填
    )

    std_re = re.compile(r"Unseen StdDev:\s*([0-9.eE+-]+|nan|NaN)")

    for line in lines:
        line = line.strip()
        m = model_re.match(line)
        if m:
            data["model_name"] = m.group(1).strip()
            continue
        m = block_re.match(line)
        if m:
            data["block_index"] = int(m.group(1))
            continue
        m = metrics_re.match(line)
        if m:
            split_name = m.group(1).lower()
            data[f"{split_name}_mae"]   = _to_float(m.group(2))
            data[f"{split_name}_rmse"]  = _to_float(m.group(3))
            data[f"{split_name}_r2"]    = _to_float(m.group(4))
            data[f"{split_name}_smape"] = _to_float(m.group(5))
            if m.group(6): # 如果有抓到 MAPE
                data[f"{split_name}_mape"] = _to_float(m.group(6))
            continue
        m = std_re.match(line)
        if m:
            data["unseen_stddev"] = _to_float(m.group(1))
            continue
    return data

def merge_metrics_files_by_predictor(target_model: str, target_preset: str, root_dir: str):
    # 修正路徑以符合 Universal_Results/{MODEL}/preset{A}
    search_start_dir = os.path.join(root_dir, TRAINING_ROOT_DIR_NAME, target_model, "preset" + target_preset)
    
    if not os.path.isdir(search_start_dir):
        # 嘗試處理沒有 preset 層級的模型 (VGG19, MobileNetV2 可能直接在模型目錄下)
        search_start_dir = os.path.join(root_dir, TRAINING_ROOT_DIR_NAME, target_model)
        if not os.path.isdir(search_start_dir):
            print(f"❌ 找不到資料夾：{os.path.abspath(search_start_dir)}")
            return

    output_dir_path = os.path.join(root_dir, NEW_OUTPUT_DIR, target_model, f"preset{target_preset}")
    os.makedirs(output_dir_path, exist_ok=True)

    # 修改正則表達式以匹配 Universal_block{idx}_{method}
    # 範例資料夾：Universal_block0_vote_ensemble
    pattern_str = f"^Universal_block(\\d+)_({'|'.join(PREDICTOR_METHODS)})$"
    target_pattern = re.compile(pattern_str)

    files_to_merge: Dict[str, List[FileEntry]] = {m: [] for m in PREDICTOR_METHODS}

    print(f"--- 開始掃描：{search_start_dir} ---")
    for dirpath, _, filenames in os.walk(search_start_dir):
        dirname = os.path.basename(dirpath)
        m = target_pattern.match(dirname)
        if m and "metrics.txt" in filenames:
            block_idx = int(m.group(1))
            predictor = m.group(2)
            metrics_path = os.path.join(dirpath, "metrics.txt")
            files_to_merge[predictor].append((block_idx, metrics_path, dirname))

    csv_output_path = os.path.join(output_dir_path, f"merged_{target_model}_metrics_all.csv")
    csv_headers = [
        "model", "predictor", "block_index",
        "train_mae", "train_rmse", "train_r2", "train_smape", "train_mape",
        "test_mae", "test_rmse", "test_r2", "test_smape", "test_mape",
        "unseen_mae", "unseen_rmse", "unseen_r2", "unseen_smape", "unseen_mape",
        "unseen_stddev",
    ]

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

        for method in PREDICTOR_METHODS:
            entries = files_to_merge[method]
            if not entries: continue

            entries.sort(key=lambda x: x[0])
            txt_output_path = os.path.join(output_dir_path, f"merged_{target_model}_{method}_training_data.txt")

            with open(txt_output_path, "w", encoding="utf-8") as txtout:
                print(f"📄 合併預測器 [{method}] -> {os.path.basename(txt_output_path)}")
                for block_index, metrics_path, dirname in entries:
                    txtout.write("=" * 80 + "\n")
                    txtout.write(f"Metrics for Block {block_index} ({dirname})\n")
                    txtout.write("=" * 80 + "\n")
                    
                    with open(metrics_path, "r", encoding="utf-8") as infile:
                        txtout.write(infile.read() + "\n\n")

                    parsed = parse_metrics_file(metrics_path)
                    writer.writerow([
                        target_model,
                        method,
                        block_index,
                        parsed.get("train_mae"), parsed.get("train_rmse"), parsed.get("train_r2"), parsed.get("train_smape"), parsed.get("train_mape"),
                        parsed.get("test_mae"), parsed.get("test_rmse"), parsed.get("test_r2"), parsed.get("test_smape"), parsed.get("test_mape"),
                        parsed.get("unseen_mae"), parsed.get("unseen_rmse"), parsed.get("unseen_r2"), parsed.get("unseen_smape"), parsed.get("unseen_mape"),
                        parsed.get("unseen_stddev"),
                    ])

    print(f"✅ 完成！CSV 檔案：{csv_output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--preset", type=str, default="A")
    args = parser.parse_args()
    merge_metrics_files_by_predictor(args.model, args.preset, ".")

if __name__ == "__main__":
    main()
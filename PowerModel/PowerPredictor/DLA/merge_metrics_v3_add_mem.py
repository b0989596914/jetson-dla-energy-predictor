'''
# predictor後先把所有block的資料合併 -> 存去new_v3_add_mem資料夾中

python merge_metrics_v3_add_mem.py --model VGG16 --preset A
python merge_metrics_v3_add_mem.py --model ResNet50 --preset A

python merge_metrics_v3_add_mem.py --model VGG19 --preset A
python merge_metrics_v3_add_mem.py --model MobileNetV2 --preset A

'''
import os
import argparse
import re
import csv
from typing import Dict, List, Tuple

# 可用的模型和方法
SUPPORTED_MODELS = ["VGG16", "VGG19", "ResNet50", "MobileNetV2"]
PREDICTOR_METHODS = ["lasso", "xgboost", "mlp", "lightgbm", "extratree"]

# 檔名設定
TRAINING_ROOT_DIR_NAME = "training_data_v3_add_mem"
NEW_OUTPUT_DIR = "new_v3_add_mem"

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

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    model_re = re.compile(r"Model Name:\s*(.+)")
    block_re = re.compile(r"Unseen Block:\s*(\d+)")

    # ① 有 MAPE 的版本
    metrics_with_mape = re.compile(
        r"(Train|Test|Unseen)\s+Metrics:\s*"
        r"MAE=([0-9.eE+-]+|nan|NaN),\s*"
        r"RMSE=([0-9.eE+-]+|nan|NaN),\s*"
        r"R2=([0-9.eE+-]+|nan|NaN),\s*"
        r"SMAPE=([0-9.eE+-]+|nan|NaN)%,\s*"
        r"MAPE=([0-9.eE+-]+|nan|NaN)%"
    )

    # ② 沒有 MAPE 的版本
    metrics_no_mape = re.compile(
        r"(Train|Test|Unseen)\s+Metrics:\s*"
        r"MAE=([0-9.eE+-]+|nan|NaN),\s*"
        r"RMSE=([0-9.eE+-]+|nan|NaN),\s*"
        r"R2=([0-9.eE+-]+|nan|NaN),\s*"
        r"SMAPE=([0-9.eE+-]+|nan|NaN)%"
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

        m = metrics_with_mape.match(line)
        if m:
            split_name = m.group(1).lower()   # train / test / unseen
            data[f"{split_name}_mae"]   = _to_float(m.group(2))
            data[f"{split_name}_rmse"]  = _to_float(m.group(3))
            data[f"{split_name}_r2"]    = _to_float(m.group(4))
            data[f"{split_name}_smape"] = _to_float(m.group(5))
            data[f"{split_name}_mape"]  = _to_float(m.group(6))
            continue

        m = metrics_no_mape.match(line)
        if m:
            split_name = m.group(1).lower()
            data[f"{split_name}_mae"]   = _to_float(m.group(2))
            data[f"{split_name}_rmse"]  = _to_float(m.group(3))
            data[f"{split_name}_r2"]    = _to_float(m.group(4))
            data[f"{split_name}_smape"] = _to_float(m.group(5))
            continue

        m = std_re.match(line)
        if m:
            data["unseen_stddev"] = _to_float(m.group(1))
            continue

    return data

def merge_metrics_files_by_predictor(target_model: str, target_preset: str, root_dir: str):
    if target_model not in SUPPORTED_MODELS:
        print(f"錯誤：不支援的模型名稱 '{target_model}'")
        return

    # 掃的根目錄：training_data/{MODEL}/presetA
    search_start_dir = os.path.join(root_dir, TRAINING_ROOT_DIR_NAME, target_model, "preset" + target_preset)
    if not os.path.isdir(search_start_dir):
        print(f"❌ 找不到資料夾：{os.path.abspath(search_start_dir)}")
        return

    # 輸出的根目錄：new/{MODEL}/presetA
    output_dir_path = os.path.join(root_dir, NEW_OUTPUT_DIR, target_model, f"preset{target_preset}")
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"✅ 輸出資料夾：{os.path.abspath(output_dir_path)}")

    # 先收集所有方法的檔案
    files_to_merge: Dict[str, List[FileEntry]] = {m: [] for m in PREDICTOR_METHODS}
    pattern_str = f"^{re.escape(target_model)}_block(\\d+)_({'|'.join(PREDICTOR_METHODS)})$"
    target_pattern = re.compile(pattern_str)

    print(f"--- 開始收集 {target_model} 的 metrics.txt ---")
    for dirpath, _, filenames in os.walk(search_start_dir):
        dirname = os.path.basename(dirpath)
        m = target_pattern.match(dirname)
        if m and "metrics.txt" in filenames:
            block_idx = int(m.group(1))
            predictor = m.group(2)
            metrics_path = os.path.join(dirpath, "metrics.txt")
            files_to_merge[predictor].append((block_idx, metrics_path, dirname))
            print(f"  收到：{metrics_path}")

    # 單一 CSV：全方法、全部 block
    csv_headers = [
        "model", "predictor", "block_index",
        "train_mae", "train_rmse", "train_r2", "train_smape", "train_mape",
        "test_mae", "test_rmse", "test_r2", "test_smape", "test_mape",
        "unseen_mae", "unseen_rmse", "unseen_r2", "unseen_smape", "unseen_mape",
        "unseen_stddev",
    ]
    csv_output_path = os.path.join(output_dir_path, f"merged_{target_model}_metrics_all.csv")

    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

        # 各方法各寫一份 txt，並把內容寫進同一張 csv
        for method in PREDICTOR_METHODS:
            entries = files_to_merge[method]
            if not entries:
                print(f"⚠️ [{method}] 沒有找到檔案，略過 txt，但 CSV 不會寫入該方法的列。")
                continue

            entries.sort(key=lambda x: x[0])

            txt_output_path = os.path.join(
                output_dir_path,
                f"merged_{target_model}_{method}_training_data.txt"
            )

            with open(txt_output_path, "w", encoding="utf-8") as txtout:
                print(f"--- 寫入 {method} 的合併檔：{txt_output_path} ---")
                for block_index, metrics_path, dirname in entries:
                    txtout.write("=" * 80 + "\n")
                    txtout.write(f"Metrics for Block {block_index} ({dirname})\n")
                    txtout.write("=" * 80 + "\n")
                    try:
                        with open(metrics_path, "r", encoding="utf-8") as infile:
                            content = infile.read()
                            txtout.write(content)
                            txtout.write("\n\n")
                    except Exception as e:
                        txtout.write(f"[錯誤] 讀取失敗：{metrics_path} - {e}\n\n")
                        print(f"  ❌ 讀取失敗：{metrics_path} - {e}")

                    parsed = parse_metrics_file(metrics_path)
                    writer.writerow([
                        parsed.get("model_name") or target_model,
                        method,
                        parsed.get("block_index") or block_index,
                        parsed.get("train_mae"),
                        parsed.get("train_rmse"),
                        parsed.get("train_r2"),
                        parsed.get("train_smape"),
                        parsed.get("train_mape"),
                        parsed.get("test_mae"),
                        parsed.get("test_rmse"),
                        parsed.get("test_r2"),
                        parsed.get("test_smape"),
                        parsed.get("test_mape"),
                        parsed.get("unseen_mae"),
                        parsed.get("unseen_rmse"),
                        parsed.get("unseen_r2"),
                        parsed.get("unseen_smape"),
                        parsed.get("unseen_mape"),
                        parsed.get("unseen_stddev"),
                    ])

    print(f"✅ 全部方法已寫入同一張 CSV：{csv_output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="合併 training_data/{MODEL}/presetA/.../metrics.txt，並輸出單一 CSV（含 lasso/xgboost/mlp/lightgbm/extratree）。"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="要合併的模型名稱，例如 VGG16"
    )
    parser.add_argument("--preset", type=str, default="A")
    args = parser.parse_args()
    merge_metrics_files_by_predictor(args.model, args.preset, ".")

if __name__ == "__main__":
    main()

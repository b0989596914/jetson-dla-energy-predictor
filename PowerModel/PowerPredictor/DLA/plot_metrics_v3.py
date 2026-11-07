#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 將合併後的資料 -> 單獨畫一張視覺化的長條圖 -> 存在charts_v3資料夾中

12/5 v3測試資料：
python plot_metrics_v3.py --in ./new_v3_2/VGG16/presetA/merged_VGG16_lasso_training_data.txt --outdir ./charts_v3_2
python plot_metrics_v3.py --in ./new_v3_2/VGG16/presetA/merged_VGG16_mlp_training_data.txt --outdir ./charts_v3_2
python plot_metrics_v3.py --in ./new_v3_2/VGG16/presetA/merged_VGG16_xgboost_training_data.txt --outdir ./charts_v3_2


"""
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


def strip_all_suffixes(p: Path) -> Path:
    """移除所有副檔名（例：.txt、.log.gz 都會被去掉）"""
    q = p
    for _ in p.suffixes:
        q = q.with_suffix("")
    return q

def sanitize_filename(s: str) -> str:
    """把不適合檔名的字元換成底線"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

TRAIN_RE = re.compile(
    r"Train Metrics:\s*MAE=([-\d\.]+),\s*RMSE=([-\d\.]+),\s*R2=([-\d\.]+),\s*SMAPE=([-\d\.]+)%"
)
TEST_RE = re.compile(
    r"Test\s*Metrics:\s*MAE=([-\d\.]+),\s*RMSE=([-\d\.]+),\s*R2=([-\d\.]+),\s*SMAPE=([-\d\.]+)%"
)
UNSEEN_RE = re.compile(
    r"Unseen Metrics:\s*MAE=([-\d\.]+),\s*RMSE=([-\d\.]+),\s*R2=([-\d\.]+),\s*SMAPE=([-\d\.]+)%"
)
STD_RE = re.compile(r"Unseen StdDev:\s*([-\d\.]+)")
BLOCK_RE = re.compile(r"Model Name:\s*(.+?)\s*?\nUnseen Block:\s*(\d+)", flags=re.S)


def _to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_blocks(text: str) -> List[Dict[str, Any]]:
    """
    解析每個 block：抓 Train/Test/Unseen 指標與 Unseen StdDev。
    假設每段以：
        Model Name: ...
        Unseen Block: N
        Train Metrics: ...
        Test  Metrics: ...
        Unseen Metrics: ...
        Unseen StdDev: ...
    的順序出現（中間允許其他行，例如係數區塊）。
    """
    # 以 "Model Name: ...\nUnseen Block: N" 為段落錨點
    items = []
    # 找到所有段落起點
    starts = list(BLOCK_RE.finditer(text))
    for i, m in enumerate(starts):
        model = m.group(1).strip()
        block_id = int(m.group(2))
        start_idx = m.end()
        end_idx = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        chunk = text[start_idx:end_idx]

        # 指標
        train = TRAIN_RE.search(chunk)
        test = TEST_RE.search(chunk)
        unseen = UNSEEN_RE.search(chunk)
        stdm = STD_RE.search(chunk)

        row = {"model": model, "block": block_id}

        if train:
            row.update(
                {
                    "MAE_train": _to_float(train.group(1)),
                    "RMSE_train": _to_float(train.group(2)),
                    "R2_train": _to_float(train.group(3)),
                    "SMAPE_train": _to_float(train.group(4)),
                }
            )
        if test:
            row.update(
                {
                    "MAE_test": _to_float(test.group(1)),
                    "RMSE_test": _to_float(test.group(2)),
                    "R2_test": _to_float(test.group(3)),
                    "SMAPE_test": _to_float(test.group(4)),
                }
            )
        if unseen:
            row.update(
                {
                    "MAE_unseen": _to_float(unseen.group(1)),
                    "RMSE_unseen": _to_float(unseen.group(2)),
                    "R2_unseen": _to_float(unseen.group(3)),
                    "SMAPE_unseen": _to_float(unseen.group(4)),
                }
            )
        if stdm:
            row["Unseen_StdDev"] = _to_float(stdm.group(1))

        items.append(row)
    return sorted(items, key=lambda x: x["block"])


# def plot_grouped_bars(df: pd.DataFrame, metric: str, out_path: Path, title_prefix: str):
#     """
#     畫群組長條圖：同一個 block 上放 Train/Test/Unseen 三根柱。
#     metric: 'MAE' | 'RMSE' | 'R2' | 'SMAPE'
#     """
#     import numpy as np

#     x = np.arange(len(df["block"]))
#     width = 0.25

#     y_train = df[f"{metric}_train"].values
#     y_test = df[f"{metric}_test"].values
#     y_unseen = df[f"{metric}_unseen"].values

#     plt.figure()
#     plt.bar(x - width, y_train, width, label="Train")
#     plt.bar(x, y_test, width, label="Test")
#     plt.bar(x + width, y_unseen, width, label="Unseen")

#     plt.xticks(x, [str(b) for b in df["block"].tolist()])
#     plt.xlabel("Unseen Block ID")
#     plt.ylabel(metric if metric != "R2" else "R²")
#     plt.title(f"{title_prefix} — {metric} (Train/Test/Unseen) by Block")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()
def plot_grouped_bars(df: pd.DataFrame, metric: str, out_path: Path, title_prefix: str):
    """
    畫群組長條圖：同一個 block 上放 Train/Test/Unseen 三根柱。
    metric: 'MAE' | 'RMSE' | 'R2' | 'SMAPE'
    """
    import numpy as np

    x = np.arange(len(df["block"]))
    width = 0.25

    y_train = df[f"{metric}_train"].values
    y_test = df[f"{metric}_test"].values
    y_unseen = df[f"{metric}_unseen"].values

    # --- 關鍵修改部分開始 ---
    # 找出所有 y 值的最大值，作為 y 軸的上限
    all_y_values = np.concatenate([y_train, y_test, y_unseen])
    # 忽略 NaN 值，計算最大值
    max_y = np.nanmax(all_y_values)
    
    # 為了讓圖表更好看，可以稍微增加一點空間 (例如增加 5%)
    # 這裡假設所有指標都是正值或 0
    y_limit = max_y * 1.05 if max_y > 0 else 1.05 # 如果 max_y <= 0，設一個預設值
    
    plt.figure()
    plt.bar(x - width, y_train, width, label="Train")
    plt.bar(x, y_test, width, label="Test")
    plt.bar(x + width, y_unseen, width, label="Unseen")
    
    # 設置 y 軸上限
    plt.ylim(0, y_limit) 
    # --- 關鍵修改部分結束 ---

    plt.xticks(x, [str(b) for b in df["block"].tolist()])
    plt.xlabel("Unseen Block ID")
    plt.ylabel(metric if metric != "R2" else "R²")
    plt.title(f"{title_prefix} — {metric} (Train/Test/Unseen) by Block")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_unseen_stddev(df: pd.DataFrame, out_path: Path, title_prefix: str):
    plt.figure()
    plt.plot(df["block"].values, df["Unseen_StdDev"].values, marker="o")
    plt.xlabel("Unseen Block ID")
    plt.ylabel("Unseen StdDev (mJ)")
    plt.title(f"{title_prefix} — Unseen StdDev by Block")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_unseen_only(df: pd.DataFrame, metric: str, out_path: Path, title_prefix: str):
    plt.figure()
    plt.bar(df["block"].astype(str).values, df[f"{metric}_unseen"].values)
    plt.xlabel("Unseen Block ID")
    plt.ylabel(metric if metric != "R2" else "R²")
    plt.title(f"{title_prefix} — {metric} (Unseen) by Block")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="原始文字結果檔 (txt)")
    ap.add_argument("--outdir", default="", help="輸出資料夾")
    ap.add_argument("--title", default="", help="圖表標題前綴（留空會自動從檔名判斷）")
    args = ap.parse_args()
    
    in_path = Path(args.infile)
    base_tag = sanitize_filename(strip_all_suffixes(in_path).name)
    print(f"[INFO] output filename prefix = {base_tag}")

    # 自動解析 model / preset
    parts = in_path.parts
    model_name = None
    preset_name = None
    for p in parts:
        if p in ["VGG16", "VGG19", "ResNet50", "MobileNetV2"]:
            model_name = p
        if p.startswith("preset"):
            preset_name = p

    if model_name is None:
        raise SystemExit("❌ 無法從輸入路徑解析出 model（例如 VGG16）")

    if preset_name is None:
        preset_name = "presetUnknown"

    # 決定輸出資料夾：outdir/model/presetX/
    outdir = Path(args.outdir) / model_name / preset_name
    outdir.mkdir(parents=True, exist_ok=True)

    # 標題前綴
    if args.title.strip():
        title_prefix = args.title.strip()
    else:
        m = re.match(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)", base_tag)
        if m:
            model_tag = m.group(1).upper()
            method_raw = m.group(2).lower()
            METHOD_MAP = {"lasso": "LASSO", "mlp": "MLP", "feature": "XGBOOST"}
            method_name = METHOD_MAP.get(method_raw, method_raw.upper())
            title_prefix = f"{model_tag} ({method_name})"
        else:
            title_prefix = base_tag

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    rows = parse_blocks(text)
    if not rows:
        raise SystemExit("❌ 沒找到任何 block 資料。")

    df = pd.DataFrame(rows).sort_values("block").reset_index(drop=True)

    out_path = outdir / f"{base_tag}_smape_by_block.png"
    plot_grouped_bars(df, "SMAPE", out_path, title_prefix)
    print(f"✅ 圖表：{out_path}")
    print("🎉 完成。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把多個 predictor 的 merged_*_training_data.txt 做 unseen 指標比較折線圖（依 block）
只用兩個參數：
  --input  可丟多個 txt（至少 2 個）
  --outdir 輸出資料夾

Example:
python plot_combine_metrics_v4_unseen_lineChart.py \
  --input \
    ./new_v3_add_mem/ResNet50/presetA/merged_ResNet50_lasso_training_data.txt \
    ./new_v3_add_mem/ResNet50/presetA/merged_ResNet50_xgboost_training_data.txt \
    ./new_v3_add_mem/ResNet50/presetA/merged_ResNet50_mlp_training_data.txt \
    ./new_v3_add_mem/ResNet50/presetA/merged_ResNet50_lightgbm_training_data.txt \
    ./new_v3_add_mem/ResNet50/presetA/merged_ResNet50_extratree_training_data.txt \
  --outdir ./charts_unseen_line_compare

VGG16：
python plot_combine_metrics_v4_unseen_lineChart.py \
  --input ./new_v3_add_mem/VGG16/presetA/merged_VGG16_{lasso,xgboost,mlp,lightgbm,extratree}_training_data.txt \
  --outdir ./charts_unseen_line_compare

ResNet50：
python plot_combine_metrics_v4_unseen_lineChart.py \
  --input ./new_v3_add_mem/ResNet50/presetA/merged_ResNet50_{lasso,xgboost,mlp,lightgbm,extratree}_training_data.txt \
  --outdir ./charts_unseen_line_compare

"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers: infer label from filename
# -----------------------------

import re
from pathlib import Path

def infer_predictor_from_filename(p: Path) -> str:
    """
    從 merged_*_training_data.txt 檔名推 predictor
    支援：lasso / xgboost / mlp / lightgbm / extratree
    """
    s = p.name.lower()

    # 常見關鍵字（以你目前檔名格式為主）
    if "lightgbm" in s:
        return "lightgbm"
    if "xgboost" in s:
        return "xgboost"
    if "lasso" in s:
        return "lasso"
    if "mlp" in s:
        return "mlp"

    # ExtraTrees 可能有多種寫法
    for k in ["extratree", "extra_tree", "extra-trees", "extratrees", "et"]:
        if k in s:
            return "extratree"

    # fallback：嘗試抓 merged_MODEL_xxx_training_data.txt 的 xxx
    m = re.match(r"merged_[A-Za-z0-9]+_([A-Za-z0-9]+)_training_data\.txt", p.name)
    if m:
        return m.group(1).lower()

    return "unknown"


def infer_model_from_filename(p: Path) -> str:
    """
    從 merged_*_training_data.txt 檔名推 model（用於輸出資料夾）
    """
    s = p.name.lower()
    if "vgg16" in s:
        return "VGG16"
    if "vgg19" in s:
        return "VGG19"
    if "resnet50" in s:
        return "ResNet50"
    if "mobilenetv2" in s:
        return "MobileNetV2"
    return "MODEL"





# -----------------------------
# Parse merged txt
# -----------------------------
NUM = r"([0-9.eE+-]+|nan|NaN|inf|-inf|INF|-INF)"
UNSEEN_RE = re.compile(
    rf"Unseen Metrics:\s*MAE={NUM},\s*RMSE={NUM},\s*R2={NUM},\s*SMAPE={NUM}%"
)
BLOCK_RE = re.compile(
    r"Unseen Block:\s*(\d+)"
)

def _to_float(s: str) -> float:
    s = str(s).strip()
    if s.lower() in ["nan", "none", ""]:
        return float("nan")
    if s.lower() == "inf":
        return float("inf")
    if s.lower() == "-inf":
        return float("-inf")
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_unseen_metrics_from_merged_txt(text: str) -> pd.DataFrame:
    """
    從 merged_*_training_data.txt 抽出每個 block 的 Unseen MAE/RMSE/R2/SMAPE
    依序掃描，遇到 Unseen Block 就記住 block，再遇到 Unseen Metrics 就寫入。
    """
    rows: List[Dict[str, Any]] = []
    cur_block = None

    for line in text.splitlines():
        line = line.strip()

        m_b = BLOCK_RE.search(line)
        if m_b:
            cur_block = int(m_b.group(1))
            continue

        m_u = UNSEEN_RE.search(line)
        if m_u and cur_block is not None:
            rows.append({
                "block": cur_block,
                "MAE": _to_float(m_u.group(1)),
                "RMSE": _to_float(m_u.group(2)),
                "R2": _to_float(m_u.group(3)),
                "SMAPE": _to_float(m_u.group(4)),
            })
            # 一個 block 只取一筆 unseen（通常就是一筆）
            cur_block = None

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("block").drop_duplicates("block").reset_index(drop=True)
    return df


def align_common_blocks(dfs: Dict[str, pd.DataFrame]) -> Tuple[List[int], Dict[str, pd.DataFrame]]:
    """
    對齊所有 predictor 共同的 blocks，避免線條長度不一樣。
    """
    block_sets = []
    for name, df in dfs.items():
        block_sets.append(set(df["block"].tolist()))
    common = sorted(list(set.intersection(*block_sets))) if block_sets else []

    if not common:
        raise SystemExit("❌ 沒有任何共同的 block，無法畫比較折線圖（請確認每個 txt 都有 Unseen Block/Unseen Metrics）。")

    out = {}
    for name, df in dfs.items():
        out[name] = df[df["block"].isin(common)].sort_values("block").reset_index(drop=True)

    return common, out


# -----------------------------
# Plot
# -----------------------------
def plot_metric_lines(blocks: List[int], dfs: Dict[str, pd.DataFrame], metric: str, out_path: Path, title: str):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(blocks))

    for pred, df in dfs.items():
        if metric not in df.columns:
            continue
        y = df[metric].values.astype(float)
        # 忽略全 NaN
        if np.all(np.isnan(y)):
            continue
        plt.plot(x, y, marker="o", label=pred)

    plt.xticks(x, [str(b) for b in blocks])
    plt.xlabel("Unseen Block ID")
    plt.ylabel(metric if metric != "R2" else "R²")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="要比較的 merged_*_training_data.txt（可多個，至少 2 個）")
    ap.add_argument("--outdir", required=True, help="輸出資料夾")
    args = ap.parse_args()

    files = [Path(p) for p in args.input]
    if len(files) < 2:
        raise SystemExit("❌ --input 至少要提供 2 個 txt 檔案才能比較。")

    for fp in files:
        if not fp.is_file():
            raise SystemExit(f"❌ 找不到檔案：{fp}")

    # 先推斷 model，並檢查所有 input 是否同一個 model
    models = [infer_model_from_filename(fp) for fp in files]
    model_name = models[0]
    if any(m != model_name for m in models):
        raise SystemExit(f"❌ --input 檔案混到不同 model：{models}，請一次只比較同一個 model。")

    # 輸出資料夾：outdir/model_name/
    outdir = Path(args.outdir) / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    # 讀取 + 解析
    dfs: Dict[str, pd.DataFrame] = {}


    for fp in files:
        pred = infer_predictor_from_filename(fp)
        text = fp.read_text(encoding="utf-8", errors="ignore")
        df = parse_unseen_metrics_from_merged_txt(text)
        if df.empty:
            print(f"⚠️ 解析不到 unseen 指標：{fp}")
            continue
        # 防止 label 重複：同 predictor 多次輸入就加尾碼
        label = pred
        if label in dfs:
            k = 2
            while f"{label}_{k}" in dfs:
                k += 1
            label = f"{label}_{k}"
        dfs[label] = df
        print(f"✅ Loaded {fp.name} as '{label}', blocks={len(df)}")

    if len(dfs) < 2:
        raise SystemExit("❌ 至少要成功解析 2 個 predictor 才能比較。")

    # 對齊共同 blocks
    blocks, dfs_aligned = align_common_blocks(dfs)
    print(f"🔗 Common blocks: {blocks}")

    # 自動偵測有哪些 metric 可畫（只要至少一個 predictor 有非 NaN）
    candidate_metrics = ["SMAPE", "R2", "MAE", "RMSE"]
    available_metrics = []
    for m in candidate_metrics:
        ok = False
        for _, df in dfs_aligned.items():
            if m in df.columns and not np.all(np.isnan(df[m].values.astype(float))):
                ok = True
                break
        if ok:
            available_metrics.append(m)

    if not available_metrics:
        raise SystemExit("❌ 沒有任何可用的 unseen 指標欄位（SMAPE/R2/MAE/RMSE）。")

    # 畫圖
    for metric in available_metrics:
        out_path = outdir / f"{model_name}_unseen_{metric}_line_compare.png"
        title = f"{model_name} — Unseen {metric} by Block (Predictor Compare)"
        plot_metric_lines(blocks, dfs_aligned, metric, out_path, title)
        print(f"📈 Saved: {out_path}")

    print("🎉 Done.")


if __name__ == "__main__":
    main()

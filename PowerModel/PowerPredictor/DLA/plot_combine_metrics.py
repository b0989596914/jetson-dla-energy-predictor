#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse MobileNetV2 per-block metrics from plain text and plot charts.
python plot_combine_metrics.py --old_in ./draw_plot_old/mobilenetv2_lasso.txt --new_in ./draw_plot/mobilenetv2_lasso.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/mobilenetv2_mlp.txt --new_in ./draw_plot/mobilenetv2_mlp.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/mobilenetv2_feature.txt --new_in ./draw_plot/mobilenetv2_feature.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/resnet50_lasso.txt --new_in ./draw_plot/resnet50_lasso.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/resnet50_mlp.txt --new_in ./draw_plot/resnet50_mlp.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/resnet50_feature.txt --new_in ./draw_plot/resnet50_feature.txt

python plot_combine_metrics.py --old_in ./draw_plot_old/vgg16_lasso.txt --new_in ./draw_plot/vgg16_lasso.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/vgg16_mlp.txt --new_in ./draw_plot/vgg16_mlp.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/vgg16_feature.txt --new_in ./draw_plot/vgg16_feature.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/vgg19_lasso.txt --new_in ./draw_plot/vgg19_lasso.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/vgg19_mlp.txt --new_in ./draw_plot/vgg19_mlp.txt
python plot_combine_metrics.py --old_in ./draw_plot_old/vgg19_feature.txt --new_in ./draw_plot/vgg19_feature.txt

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
# 舊的 plot_grouped_bars 函數可以刪除或保留備用。
# 我們將實現新的 plot_combined_grouped_bars。

def plot_combined_grouped_bars(
    df_new: pd.DataFrame, 
    df_old: pd.DataFrame, 
    metric: str, 
    out_path: Path, 
    title_prefix: str
):
    """
    畫群組長條圖：在同一圖上繪製新舊資料的 Train/Test/Unseen 柱。
    排列順序：Old Train, New Train, Old Test, New Test, Old Unseen, New Unseen
    """
    import numpy as np

    # 1. 提取所有需要繪製的 Y 值
    y_new_train = df_new[f"{metric}_train"].values
    y_new_test = df_new[f"{metric}_test"].values
    y_new_unseen = df_new[f"{metric}_unseen"].values

    y_old_train = df_old[f"{metric}_train"].values
    y_old_test = df_old[f"{metric}_test"].values
    y_old_unseen = df_old[f"{metric}_unseen"].values

    # 2. 確定 X 軸
    x_blocks = df_new["block"].values
    x = np.arange(len(x_blocks))
    
    # 3. 找出所有 Y 值的最大值，作為共同 Y 軸的上限
    all_y_values = np.concatenate([
        y_new_train, y_new_test, y_new_unseen,
        y_old_train, y_old_test, y_old_unseen,
    ])
    max_y = np.nanmax(all_y_values)
    y_limit = max_y * 1.05 if max_y > 0 else 1.05

    # 4. 設定長條圖寬度和新位置
    total_bars = 6  # 舊/新 Train, 舊/新 Test, 舊/新 Unseen
    group_width = 0.8  
    bar_width = group_width / total_bars 
    
    # 重新計算長條圖的 X 座標位置，從最左邊 (-2.5 * bar_width) 開始排列
    offset = - (total_bars / 2) * bar_width + (bar_width / 2)
    
    # 順序：Old Train, New Train, Old Test, New Test, Old Unseen, New Unseen
    x_pos = {}
    x_pos["old_train"] = x + offset + 0 * bar_width
    x_pos["new_train"] = x + offset + 1 * bar_width
    x_pos["old_test"]  = x + offset + 2 * bar_width
    x_pos["new_test"]  = x + offset + 3 * bar_width
    x_pos["old_unseen"]= x + offset + 4 * bar_width
    x_pos["new_unseen"]= x + offset + 5 * bar_width
    
    
    # 5. 繪圖
    plt.figure(figsize=(12, 6))

    # --- 調整顏色，讓比較組更接近 ---
    # Train 顏色 (灰色/藍色)
    plt.bar(x_pos["old_train"], y_old_train, bar_width, label="Old Train", color="dimgray")
    plt.bar(x_pos["new_train"], y_new_train, bar_width, label="New Train", color="tab:blue")

    # Test 顏色 (深灰/橘色)
    plt.bar(x_pos["old_test"], y_old_test, bar_width, label="Old Test", color="gray")
    plt.bar(x_pos["new_test"], y_new_test, bar_width, label="New Test", color="tab:orange")
    
    # Unseen 顏色 (淺灰/綠色)
    plt.bar(x_pos["old_unseen"], y_old_unseen, bar_width, label="Old Unseen", color="silver")
    plt.bar(x_pos["new_unseen"], y_new_unseen, bar_width, label="New Unseen", color="tab:green")
    
    
    # 6. 設定標籤和範圍
    plt.ylim(0, y_limit) 
    plt.xticks(x, [str(b) for b in x_blocks])
    plt.xlabel("Unseen Block ID")
    plt.ylabel(metric if metric != "R2" else "R²")
    plt.title(f"{title_prefix} — {metric} (Combined Old & New) by Block")
    
    # 調整圖例位置為兩行或三行
    plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(rect=[0, 0.15, 1, 1]) 
    plt.savefig(out_path, dpi=150)
    plt.close()

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
    ap.add_argument("--old_in", required=True, help="舊版原始文字結果檔 (txt)")
    ap.add_argument("--new_in", required=True, help="新版原始文字結果檔 (txt)")
    ap.add_argument("--outdir", default="./charts_combine", help="輸出資料夾")
    ap.add_argument("--title", default="", help="圖表標題前綴，若留空則根據檔名自動生成") # 改變預設值
    args = ap.parse_args()
    
    # 輸出資料夾準備
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # ... (讀取新舊資料的程式碼不變)
    # --- 讀取舊資料 ---
    print(f"[INFO] Parsing old data: {args.old_in}")
    old_text = Path(args.old_in).read_text(encoding="utf-8", errors="ignore")
    old_rows = parse_blocks(old_text)
    if not old_rows:
        raise SystemExit(f"❌ 舊資料 {args.old_in} 沒有成功解析到任何 block。")
    df_old = pd.DataFrame(old_rows).sort_values("block").reset_index(drop=True)
    
    # --- 讀取新資料 ---
    print(f"[INFO] Parsing new data: {args.new_in}")
    new_text = Path(args.new_in).read_text(encoding="utf-8", errors="ignore")
    new_rows = parse_blocks(new_text)
    if not new_rows:
        raise SystemExit(f"❌ 新資料 {args.new_in} 沒有成功解析到任何 block。")
    df_new = pd.DataFrame(new_rows).sort_values("block").reset_index(drop=True)
    
    # 取得新輸入檔名（不含路徑與所有副檔名），當作輸出檔名前綴
    new_in_path = Path(args.new_in)
    base_tag = sanitize_filename(strip_all_suffixes(new_in_path).name)
    
    # --- 關鍵修改：自動生成標題前綴 ---
    if args.title:
        # 如果使用者透過 --title 傳入了標題，則使用它
        title_prefix = args.title
    else:
        # 從檔案標籤 (base_tag, 例如 'mobilenetv2_lasso') 中解析模型和方法
        m = re.match(r"(.+?)_(.+)", base_tag)
        if m:
            model_name = m.group(1).upper() # 例如 MobileNetV2
            method_name = m.group(2).upper() # 例如 LASSO
            title_prefix = f"{model_name} ({method_name})"
        else:
            # 如果無法解析，退回使用檔案名
            title_prefix = base_tag
    
    print(f"[INFO] Chart Title Prefix = {title_prefix}")
    # 輸出檔名仍然使用 base_tag
    print(f"[INFO] output filename prefix = {base_tag}_combined")
    
    # --- 繪製合併圖表 ---
    metric = "SMAPE"
    out_path = outdir / f"{base_tag}_combined_{metric}_by_block.png"
    
    # 確保兩組資料的 block ID 數量一致，否則繪圖會出錯
    if len(df_old) != len(df_new):
          print("⚠️ 警告：新舊資料的 Block 數量不一致，請檢查！")
    
    # 將新的 title_prefix 傳入繪圖函數
    plot_combined_grouped_bars(df_new, df_old, metric, out_path, title_prefix)
    print(f"✅ 合併圖表：{out_path}")


    print("🎉 完成。")

if __name__ == "__main__":
    main()
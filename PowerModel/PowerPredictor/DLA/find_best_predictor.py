#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 多指標平衡 (Multi-metric Balance)
# 綜合排名系統 (Composite Ranking System)」，這在工業界評估預測模型（如 Power Model 或效能模型）時非常常見

'''
VGG16：
python find_best_predictor.py --csv ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_metrics_all.csv --out predictor_rank/ranking_result_VGG16.csv

VGG19：
python find_best_predictor.py --csv ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_metrics_all.csv --out predictor_rank/ranking_result_VGG19.csv

ResNet50：
python find_best_predictor.py --csv ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_metrics_all.csv --out predictor_rank/ranking_result_ResNet50.csv

MobileNetV2：
python find_best_predictor.py --csv ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_metrics_all.csv --out predictor_rank/ranking_result_MobilenetV2.csv


'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 多指標平衡 (Multi-metric Balance)
# 綜合排名系統 (Composite Ranking System)

import pandas as pd
import numpy as np
import argparse
import os
import re

def parse_txt_metrics(filepath):
    """
    解析跨模型通用預測器產生的 merged_{model}_{predictor}_training_data.txt
    """
    results = []
    filename = os.path.basename(filepath)
    
    # 從檔名推測 predictor 名稱
    predictor_name = "unknown"
    for method in ["lasso", "xgboost", "mlp", "lightgbm", "extratree", "vote_ensemble", "stack_ensemble"]:
        if method in filename:
            predictor_name = method
            break

    if not os.path.exists(filepath):
        return results

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 正則表達式匹配 Unseen 數據
    block_pattern = re.compile(r"Unseen Block:\s*(\d+)")
    metrics_pattern = re.compile(r"Unseen\s+Metrics:\s*MAE=([0-9.eE+-]+),\s*RMSE=([0-9.eE+-]+),\s*R2=([0-9.eE+-]+),\s*SMAPE=([0-9.eE+-]+)%")

    segments = re.split(r"={50,}", content)
    for seg in segments:
        b_match = block_pattern.search(seg)
        m_match = metrics_pattern.search(seg)
        if b_match and m_match:
            results.append({
                "predictor": predictor_name,
                "block_index": int(b_match.group(1)),
                "unseen_mae": float(m_match.group(1)),
                "unseen_rmse": float(m_match.group(2)),
                "unseen_r2": float(m_match.group(3)),
                "unseen_smape": float(m_match.group(4))
            })
    return results

def compute_best_predictor(file_paths, output_path=None):
    # =========================
    # 1. 讀取與合併資料 (支援 CSV 與 TXT)
    # =========================
    all_data = []
    # 如果傳入的是單一字串，轉成 list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for path in file_paths:
        if not os.path.exists(path):
            print(f"⚠️ 找不到檔案: {path}")
            continue
        
        if path.endswith(".csv"):
            all_data.append(pd.read_csv(path))
        else:
            txt_results = parse_txt_metrics(path)
            if txt_results:
                all_data.append(pd.DataFrame(txt_results))

    if not all_data:
        raise ValueError("❌ 沒有讀取到任何有效資料。")

    df = pd.concat(all_data, ignore_index=True)

    required_cols = ["predictor", "block_index", "unseen_mae", "unseen_rmse", "unseen_r2", "unseen_smape"]
    df = df.dropna(subset=required_cols).copy()

    # =========================
    # 2. 權重設定
    # =========================
    WEIGHTS = {
        "mean_unseen_mae": 0.18,
        "mean_unseen_rmse": 0.18,
        "mean_unseen_r2": 0.18,
        "mean_unseen_smape": 0.18,
        "worst_unseen_mae": 0.08,
        "worst_unseen_rmse": 0.08,
        "worst_unseen_r2": 0.05,
        "worst_unseen_smape": 0.05,
        "block_win_bonus": 0.05
    }

    # =========================
    # 3. 各 predictor 統計
    # =========================
    summary = df.groupby("predictor").agg(
        mean_unseen_mae=("unseen_mae", "mean"),
        mean_unseen_rmse=("unseen_rmse", "mean"),
        mean_unseen_r2=("unseen_r2", "mean"),
        mean_unseen_smape=("unseen_smape", "mean"),
        worst_unseen_mae=("unseen_mae", "max"),
        worst_unseen_rmse=("unseen_rmse", "max"),
        worst_unseen_r2=("unseen_r2", "min"),
        worst_unseen_smape=("unseen_smape", "max"),
    ).reset_index()

    # =========================
    # 4. block 勝場數
    # =========================
    win_count = {p: 0 for p in summary["predictor"]}
    for block_id, g in df.groupby("block_index"):
        g = g.dropna(subset=["unseen_mae", "unseen_rmse", "unseen_r2", "unseen_smape"])
        if g.empty: continue
        win_count[g.loc[g["unseen_mae"].idxmin(), "predictor"]] += 1
        win_count[g.loc[g["unseen_rmse"].idxmin(), "predictor"]] += 1
        win_count[g.loc[g["unseen_r2"].idxmax(), "predictor"]] += 1
        win_count[g.loc[g["unseen_smape"].idxmin(), "predictor"]] += 1
    summary["block_wins"] = summary["predictor"].map(win_count)

    # =========================
    # 5. rank 化
    # =========================
    smaller_better = ["mean_unseen_mae", "mean_unseen_rmse", "mean_unseen_smape", "worst_unseen_mae", "worst_unseen_rmse", "worst_unseen_smape"]
    larger_better = ["mean_unseen_r2", "worst_unseen_r2", "block_wins"]

    for col in smaller_better:
        summary[f"{col}_rank"] = summary[col].rank(method="min", ascending=True)
    for col in larger_better:
        summary[f"{col}_rank"] = summary[col].rank(method="min", ascending=False)

    # =========================
    # 6. 綜合分數計算
    # =========================
    summary["composite_score"] = (
        WEIGHTS["mean_unseen_mae"]   * summary["mean_unseen_mae_rank"] +
        WEIGHTS["mean_unseen_rmse"]  * summary["mean_unseen_rmse_rank"] +
        WEIGHTS["mean_unseen_r2"]    * summary["mean_unseen_r2_rank"] +
        WEIGHTS["mean_unseen_smape"] * summary["mean_unseen_smape_rank"] +
        WEIGHTS["worst_unseen_mae"]  * summary["worst_unseen_mae_rank"] +
        WEIGHTS["worst_unseen_rmse"] * summary["worst_unseen_rmse_rank"] +
        WEIGHTS["worst_unseen_r2"]   * summary["worst_unseen_r2_rank"] +
        WEIGHTS["worst_unseen_smape"] * summary.get("worst_unseen_smape_rank", summary["worst_unseen_smape"].rank()) +
        WEIGHTS["block_win_bonus"]   * summary["block_wins_rank"]
    )

    summary = summary.sort_values("composite_score", ascending=True).reset_index(drop=True)
    summary["final_rank"] = np.arange(1, len(summary) + 1)

    # =========================
    # 7. 最終排序與輸出顯示
    # =========================
    cols_to_show = [
        "final_rank", "predictor", "composite_score", "block_wins",
        "mean_unseen_smape", "mean_unseen_mae", "worst_unseen_smape"
    ]
    result_df = summary[cols_to_show].copy()

    print("\n===== 綜合評估排名（只看 unseen，分數越小越好）=====")
    print(result_df.to_string(index=False))

    # =========================
    # 8. 儲存 CSV 與 TXT
    # =========================
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 存 CSV
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        # 存 TXT (你要求的邏輯)
        txt_out_path = output_path.replace(".csv", ".txt")
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("🏆 通用預測器排名結果 (Universal Predictor Ranking)\n")
            f.write("="*80 + "\n")
            f.write(result_df.to_string(index=False))
            f.write("\n" + "="*80 + "\n")
            f.write(f"\n最佳預測器: {result_df.iloc[0]['predictor']}\n")
        
        print(f"\n✅ CSV 排名結果已輸出到: {output_path}")
        print(f"✅ TXT 排名結果已輸出到: {txt_out_path}")

    return result_df

def main():
    parser = argparse.ArgumentParser(description="找出綜合評估下最好的 predictor")
    # 修正：nargs='+' 支援多檔案輸入
    parser.add_argument("--csv", required=True, nargs='+', help="輸入 CSV 或多個 TXT 合併檔")
    parser.add_argument("--out", default=None, help="輸出的排名檔案路徑 (CSV 格式)")
    args = parser.parse_args()

    compute_best_predictor(args.csv, args.out)

if __name__ == "__main__":
    main()
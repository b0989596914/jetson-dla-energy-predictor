#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 多指標平衡 (Multi-metric Balance)
# 綜合排名系統 (Composite Ranking System)」，這在工業界評估預測模型（如 Power Model 或效能模型）時非常常見

'''

只適合用通用模型的預測器結果好壞比較：

python find_best_predictor_general.py --csv ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_lasso_training_data.txt ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_xgboost_training_data.txt ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_mlp_training_data.txt ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_lightgbm_training_data.txt ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_extratree_training_data.txt ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_vote_ensemble_training_data.txt ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_stack_ensemble_training_data.txt --out predictor_rank/ranking_result_universal_in_VGG16.csv
python find_best_predictor_general.py --csv ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_lasso_training_data.txt ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_xgboost_training_data.txt ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_mlp_training_data.txt ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_lightgbm_training_data.txt ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_extratree_training_data.txt ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_vote_ensemble_training_data.txt ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_stack_ensemble_training_data.txt --out predictor_rank/ranking_result_universal_in_VGG19.csv
python find_best_predictor_general.py --csv ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_lasso_training_data.txt ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_xgboost_training_data.txt ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_mlp_training_data.txt ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_lightgbm_training_data.txt ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_extratree_training_data.txt ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_vote_ensemble_training_data.txt ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_stack_ensemble_training_data.txt --out predictor_rank/ranking_result_universal_in_ResNet50.csv
python find_best_predictor_general.py --csv ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_lasso_training_data.txt ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_xgboost_training_data.txt ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_mlp_training_data.txt ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_lightgbm_training_data.txt ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_extratree_training_data.txt ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_vote_ensemble_training_data.txt ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_stack_ensemble_training_data.txt --out predictor_rank/ranking_result_universal_in_MobileNetV2.csv

'''
import pandas as pd
import numpy as np
import argparse
import os
import re

def parse_txt_metrics(filepath):
    """
    針對通用預測器產生的 merged_{model}_{predictor}_training_data.txt 進行解析
    將其轉化為 DataFrame，以便後續進行與原始 CSV 相同的運算。
    """
    results = []
    filename = os.path.basename(filepath)
    
    # 從檔名推測 predictor 名稱
    predictor_name = "unknown"
    for method in ["lasso", "xgboost", "mlp", "lightgbm", "extratree", "vote_ensemble", "stack_ensemble"]:
        if method in filename:
            predictor_name = method
            break

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 正則表達式：與你 metrics.txt 的輸出格式完全對齊
    block_pattern = re.compile(r"Unseen Block:\s*(\d+)")
    metrics_pattern = re.compile(r"Unseen\s+Metrics:\s*MAE=([0-9.eE+-]+),\s*RMSE=([0-9.eE+-]+),\s*R2=([0-9.eE+-]+),\s*SMAPE=([0-9.eE+-]+)%")
    std_pattern = re.compile(r"Unseen StdDev:\s*([0-9.eE+-]+)")

    segments = re.split(r"={50,}", content)
    for seg in segments:
        b_match = block_pattern.search(seg)
        m_match = metrics_pattern.search(seg)
        s_match = std_pattern.search(seg)
        if b_match and m_match:
            results.append({
                "predictor": predictor_name,
                "block_index": int(b_match.group(1)),
                "unseen_mae": float(m_match.group(1)),
                "unseen_rmse": float(m_match.group(2)),
                "unseen_r2": float(m_match.group(3)),
                "unseen_smape": float(m_match.group(4)),
                "unseen_stddev": float(s_match.group(1)) if s_match else 0.0
            })
    return results

def compute_best_predictor(file_paths, output_path=None):
    # =========================
    # 1. 讀取並合併資料 (多檔案整合)
    # =========================
    all_data = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"⚠️ 找不到檔案: {path}")
            continue
        
        if path.endswith(".csv"):
            all_data.append(pd.read_csv(path))
        elif path.endswith(".txt"):
            txt_results = parse_txt_metrics(path)
            if txt_results:
                all_data.append(pd.DataFrame(txt_results))

    if not all_data:
        raise ValueError("❌ 沒有讀取到任何有效資料。")

    # 將所有不同 predictor 的資料合併成一張大表，格式就跟原本的單一 CSV 一模一樣了
    df = pd.concat(all_data, ignore_index=True)

    required_cols = ["predictor", "block_index", "unseen_mae", "unseen_rmse", "unseen_r2", "unseen_smape"]
    df = df.dropna(subset=required_cols).copy()

    # ========================================================
    # 2. 權重設定 (完全延續你提供的原本邏輯)
    # ========================================================
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
    # 4. block 勝場數 (核心競爭力分析)
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
    # 5. rank 化與綜合評分
    # =========================
    smaller_better_cols = ["mean_unseen_mae", "mean_unseen_rmse", "mean_unseen_smape", "worst_unseen_mae", "worst_unseen_rmse", "worst_unseen_smape"]
    larger_better_cols = ["mean_unseen_r2", "worst_unseen_r2", "block_wins"]

    for col in smaller_better_cols:
        summary[f"{col}_rank"] = summary[col].rank(method="min", ascending=True)
    for col in larger_better_cols:
        summary[f"{col}_rank"] = summary[col].rank(method="min", ascending=False)

    summary["composite_score"] = (
        WEIGHTS["mean_unseen_mae"]   * summary["mean_unseen_mae_rank"] +
        WEIGHTS["mean_unseen_rmse"]  * summary["mean_unseen_rmse_rank"] +
        WEIGHTS["mean_unseen_r2"]    * summary["mean_unseen_r2_rank"] +
        WEIGHTS["mean_unseen_smape"] * summary["mean_unseen_smape_rank"] +
        WEIGHTS["worst_unseen_mae"]  * summary["worst_unseen_mae_rank"] +
        WEIGHTS["worst_unseen_rmse"] * summary["worst_unseen_rmse_rank"] +
        WEIGHTS["worst_unseen_r2"]   * summary["worst_unseen_r2_rank"] +
        WEIGHTS.get("worst_unseen_smape", 0.05) * summary["worst_unseen_smape_rank"] + # 補上這項
        WEIGHTS["block_win_bonus"]   * summary["block_wins_rank"]
    )

    # =========================
    # 6. 最終排序與輸出
    # =========================
    summary = summary.sort_values("composite_score", ascending=True).reset_index(drop=True)
    summary["final_rank"] = np.arange(1, len(summary) + 1)

    cols_to_show = ["final_rank", "predictor", "composite_score", "block_wins", "mean_unseen_smape", "mean_unseen_mae", "worst_unseen_smape"]
    
    print("\n" + "="*80)
    print("🏆 通用預測器排名結果 (與單模型評估標準一致)")
    print("="*80)
    print(summary[cols_to_show].to_string(index=False))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 排名結果已存至: {output_path}")
        
    # 寫成txt檔案
    txt_out_path = output_path.replace(".csv", ".txt")
    with open(txt_out_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("🏆 通用預測器排名結果 (Universal Predictor Ranking)\n")
        f.write("="*80 + "\n")
        f.write(summary[["final_rank", "predictor", "composite_score", "block_wins", 
                         "mean_unseen_smape", "mean_unseen_mae", "worst_unseen_smape"]].to_string(index=False))
        f.write("\n" + "="*80 + "\n")
    print(f"✅ 文本格式排名已輸出到: {txt_out_path}")

def main():
    parser = argparse.ArgumentParser(description="跨模型通用預測器評估系統")
    # 支援多個檔案傳入
    parser.add_argument("--csv", required=True, nargs='+', help="輸入一個或多個 TXT/CSV 檔案")
    parser.add_argument("--out", default=None, help="輸出的排名 CSV 路徑")
    args = parser.parse_args()

    compute_best_predictor(args.csv, args.out)

if __name__ == "__main__":
    main()
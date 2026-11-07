#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把多個 predictor 的 merged_*_training_data.txt 做 unseen 指標比較折線圖（依 block）
只用兩個參數：
  --input  可丟多個 txt（至少 2 個）
  --outdir 輸出資料夾

"--mode general"：通用型互相比較 (看誰最強)
"--mode compare"：一般型 vs. 通用型一對一 (證明進步)


-----------general不同預測器的比較-------------------

# 在通用資料集的訓練下，各預測器在 VGG16 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_lasso_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_mlp_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_extratree_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare \
  --annotate

# 在通用資料集的訓練下，各預測器在 VGG19 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_lasso_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_mlp_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_extratree_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare \
  --annotate

# 在通用資料集的訓練下，各預測器在 ResNet50 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_lasso_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_mlp_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_extratree_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare \
  --annotate
  
# 在通用資料集的訓練下，各預測器在 MobileNetV2 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_lasso_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_mlp_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_extratree_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare \
  --annotate
  
-----------------單一型 vs 通用型的比較-------------------
# 單一型：僅使用該model當作訓練集 vs 通用型：在通用資料集的訓練下，各預測器在 MobileNetV2 中的表現：

------lasso-----------
# lasso + VGG16
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_lasso_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_lasso_training_data.txt \
    --outdir ./charts_results --annotate
    
# lasso + VGG19
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_lasso_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_lasso_training_data.txt \
    --outdir ./charts_results --annotate
    
# lasso + ResNet50
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_lasso_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_lasso_training_data.txt \
    --outdir ./charts_results --annotate
    
# lasso + MobileNetV2
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_lasso_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_lasso_training_data.txt \
    --outdir ./charts_results --annotate

------xgboost------
# xgboost + VGG16
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_xgboost_training_data.txt \
    --outdir ./charts_results --annotate

# xgboost + VGG19
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_xgboost_training_data.txt \
    --outdir ./charts_results --annotate

# xgboost + ResNet50
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_xgboost_training_data.txt \
    --outdir ./charts_results --annotate

# xgboost + MobileNetV2
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_xgboost_training_data.txt \
    --outdir ./charts_results --annotate


------mlp------
# mlp + VGG16
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_mlp_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_mlp_training_data.txt \
    --outdir ./charts_results --annotate

# mlp + VGG19
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_mlp_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_mlp_training_data.txt \
    --outdir ./charts_results --annotate

# mlp + ResNet50
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_mlp_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_mlp_training_data.txt \
    --outdir ./charts_results --annotate

# mlp + MobileNetV2
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_mlp_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_mlp_training_data.txt \
    --outdir ./charts_results --annotate

------lightgbm------
# lightgbm + VGG16
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_lightgbm_training_data.txt \
    --outdir ./charts_results --annotate

# lightgbm + VGG19
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_lightgbm_training_data.txt \
    --outdir ./charts_results --annotate

# lightgbm + ResNet50
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_lightgbm_training_data.txt \
    --outdir ./charts_results --annotate

# lightgbm + MobileNetV2
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_lightgbm_training_data.txt \
    --outdir ./charts_results --annotate

------extratree------
# extratree + VGG16
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_extratree_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_extratree_training_data.txt \
    --outdir ./charts_results --annotate

# extratree + VGG19
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_extratree_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_extratree_training_data.txt \
    --outdir ./charts_results --annotate

# extratree + ResNet50
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_extratree_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_extratree_training_data.txt \
    --outdir ./charts_results --annotate

# extratree + MobileNetV2
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_extratree_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_extratree_training_data.txt \
    --outdir ./charts_results --annotate


------vote_ensemble------
# vote_ensemble + VGG16
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_vote_ensemble_training_data.txt \
    --outdir ./charts_results --annotate

# vote_ensemble + VGG19
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_vote_ensemble_training_data.txt \
    --outdir ./charts_results --annotate

# vote_ensemble + ResNet50
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_vote_ensemble_training_data.txt \
    --outdir ./charts_results --annotate

# vote_ensemble + MobileNetV2
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_vote_ensemble_training_data.txt \
    --outdir ./charts_results --annotate
    

------stack_ensemble------
# stack_ensemble + VGG16
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG16/presetA/merged_VGG16_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_stack_ensemble_training_data.txt \
    --outdir ./charts_results --annotate

# stack_ensemble + VGG19
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/VGG19/presetA/merged_VGG19_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_stack_ensemble_training_data.txt \
    --outdir ./charts_results --annotate

# stack_ensemble + ResNet50
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/ResNet50/presetA/merged_ResNet50_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_stack_ensemble_training_data.txt \
    --outdir ./charts_results --annotate

# stack_ensemble + MobileNetV2
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode compare \
  --input \
    ./new_v3_add_mem_add_io_size/MobileNetV2/presetA/merged_MobileNetV2_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_stack_ensemble_training_data.txt \
    --outdir ./charts_results --annotate



------通用型在unseen的比較 (有加上後來修改的stack一起比較)-------

# 在通用資料集的訓練下，各預測器在 VGG16 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_extratree_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG16/presetA/merged_VGG16_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal_with_new_stack/VGG16/presetA/merged_VGG16_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare_with_new_stack \
  --annotate

# 在通用資料集的訓練下，各預測器在 VGG19 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_extratree_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/VGG19/presetA/merged_VGG19_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal_with_new_stack/VGG19/presetA/merged_VGG19_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare_with_new_stack \
  --annotate

# 在通用資料集的訓練下，各預測器在 ResNet50 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_extratree_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/ResNet50/presetA/merged_ResNet50_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal_with_new_stack/ResNet50/presetA/merged_ResNet50_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare_with_new_stack \
  --annotate
  
# 在通用資料集的訓練下，各預測器在 MobileNetV2 中的表現：
python plot_combine_metrics_general_unseen_lineChart.py \
  --mode general \
  --input \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_xgboost_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_lightgbm_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_extratree_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_vote_ensemble_training_data.txt \
    ./new_general_Hybrid_universal/MobileNetV2/presetA/merged_MobileNetV2_stack_ensemble_training_data.txt \
    ./new_general_Hybrid_universal_with_new_stack/MobileNetV2/presetA/merged_MobileNetV2_stack_ensemble_training_data.txt \
  --outdir ./charts_universal_compare_with_new_stack \
  --annotate
  



"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Helpers: 推斷標籤與模型
# -----------------------------
def infer_predictor_from_filename(p: Path) -> str:
    s = p.name.lower()
    
    # 🎯 關鍵修正：將 'ensemble' 映射為 'vote_ensemble'
    # 先處理複合名稱
    if "stack_ensemble" in s:
        return "stack_ensemble"
    
    # 只要包含 vote_ensemble 或 單純的 ensemble，都統一叫 vote_ensemble
    if "vote_ensemble" in s or "ensemble" in s:
        return "vote_ensemble"
    
    # 其餘模型
    for method in ["lightgbm", "xgboost", "lasso", "mlp", "extratree"]:
        if method in s:
            return method
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
# Parse: 解析邏輯 (支援多種 Block 標記格式)
# -----------------------------
NUM = r"([0-9.eE+-]+|nan|NaN|inf|-inf|INF|-INF)"
UNSEEN_RE = re.compile(rf"Unseen\s+Metrics:\s*MAE={NUM},\s*RMSE={NUM},\s*R2={NUM},\s*SMAPE={NUM}%", re.IGNORECASE)
BLOCK_RE = re.compile(r"(?:Unseen Block|Metrics for Block):\s*(\d+)", re.IGNORECASE)

def _to_float(s: str) -> float:
    try: return float(str(s).strip().replace('%', ''))
    except: return float("nan")

def parse_unseen_metrics_from_merged_txt(text: str) -> pd.DataFrame:
    rows = []
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
            cur_block = None
    
    if not rows: return pd.DataFrame()
    # 確保排序並去除重複
    df = pd.DataFrame(rows).sort_values("block").drop_duplicates("block").reset_index(drop=True)
    return df


# -----------------------------
# Plot: 強化顏色對比與重複名稱處理
# -----------------------------
def plot_chart(blocks, dfs, metric, out_path, title, annotate, mode="general"):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(blocks))
    
    if mode == "general":
        # 一般模式：使用系統預設調色盤區分不同算法
        for label, df in dfs.items():
            y = df[metric].values.astype(float)
            plt.plot(x, y, marker="o", label=label, linewidth=2)
    else:
        # 🎯 Compare 模式：固定兩種顏色對比
        # Single 使用灰色虛線，Universal 使用橘色實線
        color_map = {"Single": "#95a5a6", "Universal": "#e67e22"}
        style_map = {"Single": "--", "Universal": "-"}
        marker_map = {"Single": "o", "Universal": "s"} # 圓點 vs 方塊
        
        # 取得所有基礎標籤（處理掉 Single/Universal 與後續數字）
        # 使用正規表達式移除字尾的類型與序號，例如 stack_ensemble_Universal_1 -> stack_ensemble
        def clean_label(k):
            return re.sub(r'_(Single|Universal)(_\d+)?', '', k)

        unique_base_names = sorted(list(set([clean_label(k) for k in dfs.keys()])))
        
        for base in unique_base_names:
            # 找出所有屬於該 base 的 keys
            matching_keys = [k for k in dfs.keys() if clean_label(k) == base]
            for k in matching_keys:
                suffix = "Universal" if "Universal" in k else "Single"
                # 如果有數字後綴，保留在圖例中以便區分
                version_match = re.search(r'_(\d+)$', k)
                display_label = f"{base.upper()} ({suffix})" + (f" v{version_match.group(1)}" if version_match else "")
                
                y = dfs[k][metric].values.astype(float)
                plt.plot(x, y, marker=marker_map[suffix], 
                        linestyle=style_map[suffix], 
                        color=color_map[suffix], 
                        label=display_label, 
                        linewidth=2.5, markersize=7)

    # 圖表裝飾
    plt.title(title, loc='center', fontsize=14, pad=15, fontweight='bold')
    plt.xticks(x, [str(b) for b in blocks])
    plt.xlabel("Unseen Block ID", fontsize=12)
    plt.ylabel(f"Unseen {metric}", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    if annotate:
        for line in plt.gca().get_lines():
            y_data = line.get_ydata()
            color = line.get_color()
            for xi, yi in enumerate(y_data):
                if not np.isnan(yi):
                    plt.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", 
                                 xytext=(0, 7), ha="center", fontsize=8, 
                                 color=color, fontweight='bold')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--mode", choices=["general", "compare"], default="general")
    args = ap.parse_args()

    files = [Path(p) for p in args.input]
    model_name = infer_model_from_filename(files[0])
    
    dfs = {}
    label_counters = {} # 用於追蹤重複的標籤
    
    all_preds = sorted(list(set([infer_predictor_from_filename(fp) for fp in files])))
    valid_preds = [p for p in all_preds if p != "unknown"]
    
    if args.mode == "compare":
        sub_folder_name = valid_preds[0] if valid_preds else "Comparison"
    else:
        sub_folder_name = "General_Compare"

    outdir = Path(args.outdir) / model_name / sub_folder_name
    outdir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for fp in files:
        path_str = str(fp.resolve()).lower()
        pred = infer_predictor_from_filename(fp)
        type_suffix = "Universal" if ("general" in path_str or "universal" in path_str) else "Single"
        
        # 🎯 核心修改：處理重複標籤
        base_label = f"{pred}_{type_suffix}"
        if base_label not in label_counters:
            label_counters[base_label] = 1
            label = base_label
        else:
            label_counters[base_label] += 1
            label = f"{base_label}_{label_counters[base_label]}"
        
        df = parse_unseen_metrics_from_merged_txt(fp.read_text(encoding="utf-8", errors="ignore"))
        if not df.empty:
            dfs[label] = df
            print(f"✅ 載入: {label} (檔案: {fp.name})")

    if not dfs:
        print("❌ 找不到有效數據。")
        return

    block_sets = [set(df["block"].tolist()) for df in dfs.values()]
    common_blocks = sorted(list(set.intersection(*block_sets)))
    aligned_dfs = {n: d[d["block"].isin(common_blocks)].sort_values("block") for n, d in dfs.items()}

    for metric in ["SMAPE", "MAE", "RMSE", "R2"]:
        csv_name = f"{model_name}_unseen_{metric}_comparison_table.csv"
        img_name = f"{model_name}_unseen_{metric}_line_compare.png"
        
        table_data = {"Block": common_blocks}
        for label, df in aligned_dfs.items(): 
            table_data[label] = df[metric].values
        pd.DataFrame(table_data).to_csv(outdir / csv_name, index=False)
        
        plot_chart(common_blocks, aligned_dfs, metric, outdir / img_name, f"{model_name} {metric}", args.annotate, args.mode)
        
        print(f"📊 已產出: {img_name}")

if __name__ == "__main__":
    main()
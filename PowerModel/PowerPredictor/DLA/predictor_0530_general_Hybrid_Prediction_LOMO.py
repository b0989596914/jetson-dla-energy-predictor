#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
為了證明跨模型有比單一型的好：

predictor_0530_general_Hybrid_Prediction_LOMO.py -> 是做跨模型通用預測器(LOMO的)
會存在 LOMO_Universal_Results/


--------非跨模型預測器----預測VGG16----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/VGG16/Res_To_VGG16"

用ResNet50當訓練集
預測VGG16

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models ResNet50 \
    --unseen_models VGG16 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/VGG19_To_VGG16"

用VGG19當訓練集
預測VGG16

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG19 \
    --unseen_models VGG16 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/VGG16/Mob_To_VGG16"

用MobileNetV2當訓練集
預測VGG16

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models MobileNetV2 \
    --unseen_models VGG16 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv




------全域 LOMO跨模型預測器-----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/VGG16/Another_To_VGG16"

用VGG19、ResNet50、MobileNetV2當訓練集
預測VGG16

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG19 ResNet50 MobileNetV2 \
    --unseen_models VGG16 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv



--------非跨模型預測器----預測VGG19----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/VGG19/Res_To_VGG19"

用ResNet50當訓練集
預測 VGG19

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models ResNet50 \
    --unseen_models VGG19 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/VGG19/VGG16_To_VGG19"

用VGG16當訓練集
預測 VGG19

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG16 \
    --unseen_models VGG19 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/VGG19/Mob_To_VGG19"

用MobileNetV2當訓練集
預測 VGG19

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models MobileNetV2 \
    --unseen_models VGG19 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv




------全域 LOMO跨模型預測器-----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/VGG19/Another_To_VGG19"

用VGG16、ResNet50、MobileNetV2當訓練集
預測 VGG19

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG16 ResNet50 MobileNetV2 \
    --unseen_models VGG19 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv




--------非跨模型預測器----預測ResNet50----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/ResNet50/VGG19_To_Res"

用VGG19當訓練集
預測 ResNet50

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG19 \
    --unseen_models ResNet50 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/ResNet50/VGG16_To_Res"

用VGG16當訓練集
預測 ResNet50

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG16 \
    --unseen_models ResNet50 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/ResNet50/Mob_To_Res"

用MobileNetV2當訓練集
預測 ResNet50

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models MobileNetV2 \
    --unseen_models ResNet50 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv




------全域 LOMO跨模型預測器-----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/ResNet50/Another_To_Res

用VGG16、VGG19、MobileNetV2當訓練集
預測 ResNet50

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG16 VGG19 MobileNetV2 \
    --unseen_models ResNet50 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv


--------非跨模型預測器----預測MobileNetV2----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/MobileNetV2/VGG19_To_MobileNetV2"

用VGG19當訓練集
預測 MobileNetV2

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG19 \
    --unseen_models MobileNetV2 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/MobileNetV2/VGG16_To_MobileNetV2"

用VGG16當訓練集
預測 MobileNetV2

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG16 \
    --unseen_models MobileNetV2 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

--
universal_base_dir = "LOMO_Universal_Results/MobileNetV2/Res_To_MobileNetV2"

用ResNet50當訓練集
預測 MobileNetV2

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models ResNet50 \
    --unseen_models MobileNetV2 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv




------全域 LOMO跨模型預測器-----------

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "LOMO_Universal_Results/MobileNetV2/Another_To_MobileNetV2"

用VGG16、VGG19、ResNet50當訓練集
預測 MobileNetV2

python predictor_0530_general_Hybrid_Prediction_LOMO.py \
    --model vote_ensemble \
    --train_models VGG16 VGG19 ResNet50 \
    --unseen_models MobileNetV2 \
    --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv


"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor

from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor

# 堆疊式學習法
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV  # 推薦作為 Meta-Learner

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


# ----------------------------
# Metrics
# ----------------------------
def compute_smape(y_true, y_pred, eps=1e-8):
    """
    Symmetric MAPE (%): 100 * mean( |y - yhat| / ((|y| + |yhat|)/2) )
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > eps
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0


# ----------------------------
# Train model
# ----------------------------
def train_model(X_train, y_train, model_type: str):
    
    if LGBMRegressor is None:
        raise RuntimeError("找不到 lightgbm，請先 pip install lightgbm")
    # 先定義好兩個核心強模型
    lgb_model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_child_samples=5,
        random_state=42,
        n_jobs=-1
    )

    et_model = ExtraTreesRegressor(
        n_estimators=2000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",   # <= 推薦（也可改 1.0 但要是 float）
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )
    if model_type == "lasso":
        model = Lasso(alpha=0.0001, max_iter=500000, positive=True, random_state=42)

    elif model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            learning_rate=0.05,
            random_state=42,
        )

    elif model_type == "lightgbm":
        model = lgb_model

    elif model_type == "extratree":
        # 注意：max_features 一定要是 float/int/None/"sqrt"/"log2"
        # 用 sqrt 通常穩定，避免你之前遇到 '1.0' 字串問題
        model = et_model

    elif model_type == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            validation_fraction=0.1,
            learning_rate="adaptive",
            max_iter=50000,
            random_state=42,
        )
    
    # 最基本的兩個模型的融合(以權重0.5:0.5的方式)
    # 使用 VotingRegressor 進行簡單有效的融合
    # weights=[0.5, 0.5] 可以根據實驗結果調整，例如 [0.7, 0.3]
    elif model_type == "vote_ensemble":
        model = VotingRegressor(
            estimators=[('lgb', lgb_model), ('et', et_model)],
            weights=[0.5, 0.5] 
        )
    
    # 定義 Meta-Learner (元學習器)
    # RidgeCV 會自動尋找最佳的正規化參數，適合處理子模型輸出的相關性
    elif model_type == "stack_ensemble":
        meta_model = RidgeCV()

        # 使用 StackingRegressor 替換 VotingRegressor
        model = StackingRegressor(
            estimators=[('lgb', lgb_model), ('et', et_model)],
            final_estimator=meta_model,
            cv=5,            # 使用 5 折交叉驗證來訓練 Meta-Learner，避免過擬合
            n_jobs=-1,
            passthrough=False # 若設為 True，Meta-Learner 會同時參考原始特徵與子模型預測值
    )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model


# ----------------------------
# Evaluation / Plot
# ----------------------------
def evaluate_and_print(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    smape = compute_smape(y_true, y_pred)

    print(f"\n=== {name} Evaluation ===")
    print(f"MAE   : {mae:.6f} mJ")
    print(f"RMSE  : {rmse:.6f} mJ")
    print(f"R²    : {r2:.6f}")
    if np.isnan(smape):
        print("SMAPE : n/a")
    else:
        print(f"SMAPE : {smape:.2f}%")

    return mae, rmse, r2, smape

# LOMO新增
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    smape = compute_smape(y_true, y_pred)
    return mae, rmse, r2, smape


def plot_prediction(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color="blue", label="Predicted")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             color="red", linestyle="--", label="Ideal (y=x)")
    plt.xlabel("True Energy (mJ)")
    plt.ylabel("Predicted Energy (mJ)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ----------------------------
# Save metrics + importance
# ----------------------------
def save_metrics(
    output_dir, model_name, block_index, model_type,
    mae_train, rmse_train, r2_train,
    mae_test, rmse_test, r2_test,
    mae_unseen, rmse_unseen, r2_unseen,
    smape_train, smape_test, smape_unseen,
    std_unseen, model, feature_names,
    n_train, n_test, n_unseen
):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Unseen Block: {block_index}\n")
        f.write(f"Train Metrics: MAE={mae_train:.6f}, RMSE={rmse_train:.6f}, R2={r2_train:.6f}, SMAPE={(smape_train if not np.isnan(smape_train) else float('nan')):.2f}%\n")
        f.write(f"Test  Metrics: MAE={mae_test:.6f}, RMSE={rmse_test:.6f}, R2={r2_test:.6f}, SMAPE={(smape_test  if not np.isnan(smape_test)  else float('nan')):.2f}%\n")
        f.write(f"Unseen Metrics: MAE={mae_unseen:.6f}, RMSE={rmse_unseen:.6f}, R2={r2_unseen:.6f}, SMAPE={(smape_unseen if not np.isnan(smape_unseen) else float('nan')):.2f}%\n")
        f.write(f"Unseen StdDev: {std_unseen:.6f}\n")

        f.write(f"Train Samples: {n_train}\n")
        f.write(f"Test Samples: {n_test}\n")
        f.write(f"Unseen Samples: {n_unseen}\n")

        # --- importance / coef ---
        if model_type == "lasso":
            f.write("\n=== Non-zero Lasso Coefficients ===\n")
            for name, val in zip(feature_names, model.coef_):
                if abs(val) > 1e-6:
                    f.write(f"{name:35}: {val:.6f}\n")
            f.write(f"{'__intercept__':35}: {float(model.intercept_):.6f}\n")

        elif model_type in ("xgboost", "lightgbm", "extratree"):
            f.write(f"\n=== Top 10 Feature Importances ({model_type.upper()}) ===\n")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                topk = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
                for name, val in topk:
                    f.write(f"{name:35}: {val:.6f}\n")
            else:
                f.write("(no feature_importances_ available)\n")
        # 自動取得子模型、正規化、對齊特徵
        elif model_type == "vote_ensemble":
            f.write("\n=== Top 10 Feature Importances (Vote Ensemble Normalized Average) ===\n")
            # model 是 VotingRegressor，model.estimators_ 存放訓練好的子模型
            # 根據 train_model 定義：index 0 為 lgb, index 1 為 et
            try:
                lgb_imp = model.estimators_[0].feature_importances_
                et_imp = model.estimators_[1].feature_importances_

                # 正規化處理：讓不同演算法的重要性總和皆為 1，避免量級干擾 
                lgb_norm = lgb_imp / (np.sum(lgb_imp) + 1e-9)
                et_norm = et_imp / (np.sum(et_imp) + 1e-9)

                # 計算混合後的平均重要性
                avg_imp = (lgb_norm + et_norm) / 2.0

                topk = sorted(zip(feature_names, avg_imp), key=lambda x: -x[1])[:10]
                for name, val in topk:
                    f.write(f"{name:35}: {val:.6f}\n")
            except Exception as e:
                f.write(f"(Could not compute vote ensemble importance: {e})\n")
        elif model_type == "stack_ensemble":
            f.write("\n=== Stacking Meta-Learner Analysis ===\n")
            # 取得 Meta-Learner 的係數 (如果是線性模型如 Ridge)
            if hasattr(model.final_estimator_, "coef_"):
                coefs = model.final_estimator_.coef_
                f.write(f"LightGBM Weight (learned): {coefs[0]:.4f}\n")
                f.write(f"ExtraTree Weight (learned): {coefs[1]:.4f}\n")
            
            f.write("\n=== Top 10 Feature Importances (Base Models Average) ===\n")
            # 取得第一層基礎模型的重要性
            lgb_imp = model.named_estimators_['lgb'].feature_importances_
            et_imp = model.named_estimators_['et'].feature_importances_
            
            lgb_norm = lgb_imp / (np.sum(lgb_imp) + 1e-9)
            et_norm = et_imp / (np.sum(et_imp) + 1e-9)
            avg_imp = (lgb_norm + et_norm) / 2.0
            
            topk = sorted(zip(feature_names, avg_imp), key=lambda x: -x[1])[:10]
            for name, val in topk:
                f.write(f"{name:35}: {val:.6f}\n")

# ----------------------------
# Main (Universal LOBO)
# ----------------------------
def main(args):
    # 1. 讀取並合併多個 CSV 檔案
    all_dfs = []
    for csv_path in args.csv:
        temp_df = pd.read_csv(csv_path)
        # 取得模型名稱 (從檔案名稱抓取，如 VGG16)
        m_name = os.path.basename(csv_path).split("_")[0]
        temp_df["source_model"] = m_name
        # 建立全域唯一的 Block ID (例如: VGG16_0, ResNet50_0)
        temp_df["global_block_id"] = temp_df["source_model"] + "_" + temp_df["block_index"].astype(str)
        all_dfs.append(temp_df)
        
    # 合併所有模型資料
    df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 由 block_index 推回 preset 與 local block
    df["block_index"] = df["block_index"].astype(int)
    df["preset_num"] = df["block_index"] // 1000
    df["preset_name"] = df["preset_num"].map({
        1: "A",
        2: "B",
        3: "C",
        4: "D"
    }).fillna("Unknown")
    df["local_block"] = df["block_index"] % 1000

    model_name = "Universal"

    print(f"📊 已合併資料集，總樣本數: {len(df)}")
    print(f"🔗 包含模型: {df['source_model'].unique()}")

    # if "block_index" not in df.columns:
    #     raise ValueError("需要有 block_index 欄位來進行 LOBO")
    
    # 2. 定義排除欄位 (加入我們新創的輔助欄位)
    EXCLUDE_FINAL = {
        "energy_mj", "block_index", "round", "source_name", "source_model", "global_block_id",
        "block_row", "is_single_layer", "is_eval_eligible",
        "preset_num", "preset_name", "local_block",
        "preset_A", "preset_B", "preset_C", "preset_D"
    }
    
    # LOMO修改的部分
    # 3. 找出所有要跑 Leave-One-Model-Out 的目標模型
    available_models = sorted(df["source_model"].unique())
    print(f"🔗 可用模型: {available_models}")

    # 如果沒有指定 unseen_models，就維持原本標準 LOMO：每個 model 都輪流當 unseen
    if args.unseen_models is None or len(args.unseen_models) == 0:
        target_lomo_models = available_models
    else:
        target_lomo_models = args.unseen_models

    # 檢查 unseen_models 是否存在
    missing_unseen = set(target_lomo_models) - set(available_models)
    if missing_unseen:
        raise ValueError(f"❌ unseen_models 不存在於 CSV 中: {missing_unseen}")

    # 如果有指定 train_models，也檢查是否存在
    if args.train_models is not None and len(args.train_models) > 0:
        missing_train = set(args.train_models) - set(available_models)
        if missing_train:
            raise ValueError(f"❌ train_models 不存在於 CSV 中: {missing_train}")

    print(f"🎯 Unseen models: {target_lomo_models}")

    universal_base_dir = "LOMO_Universal_Results/MobileNetV2/Another_To_MobileNetV2"

    # --- LOMO loop ---
    for unseen_model in target_lomo_models:
        print(f"\n==============================")
        print(f"🚫 Leave model out: {unseen_model}")
        print(f"==============================")

        # 1. 一整個 model 移出訓練集
        # train_models = [m for m in target_lomo_models if m != unseen_model]
        # train_tag = "train_" + "_".join(train_models)
        
        # 1. 決定這一輪要用哪些 model 當訓練集
        if args.train_models is None or len(args.train_models) == 0:
            # 沒指定 train_models：維持原本 LOMO，除了 unseen_model 以外都拿來訓練
            train_models = [m for m in available_models if m != unseen_model]
        else:
            # 有指定 train_models：完全照使用者指定
            train_models = args.train_models
        train_tag = "train_" + "_".join(train_models)

        # 避免同一個 model 同時出現在訓練集和 unseen
        if unseen_model in train_models:
            raise ValueError(
                f"❌ 設定錯誤：{unseen_model} 同時出現在 train_models 和 unseen_models，"
                f"請把它從 train_models 移除。"
            )

        print(f"📚 Train models : {train_models}")
        print(f"🧪 Unseen model : {unseen_model}")

        df_seen = df[df["source_model"].isin(train_models)].copy()
        df_unseen_model = df[df["source_model"] == unseen_model].copy()

        if df_seen.empty or df_unseen_model.empty:
            print(f"[Skip] {unseen_model} 資料不足")
            continue

        # 2. 取得數值特徵
        all_numeric_cols = df_seen.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in all_numeric_cols if c not in EXCLUDE_FINAL]

        print(f"[{args.model}] Unseen Model {unseen_model}: 特徵數 {len(feature_cols)} → {feature_cols}")

        X_seen = df_seen[feature_cols]
        y_seen = df_seen["energy_mj"]

        # 3. seen 資料切成 train / internal test
        X_train, X_test, y_train, y_test = train_test_split(
            X_seen, y_seen, test_size=0.2, random_state=42
        )

        # 4. 處理 inf / nan
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
        if all_nan_cols:
            print(f"⚠️ {unseen_model}: train 全 NaN 欄位 {len(all_nan_cols)} 個，移除：{all_nan_cols}")
            X_train = X_train.drop(columns=all_nan_cols)
            X_test = X_test.drop(columns=all_nan_cols, errors="ignore")
            feature_cols = [c for c in feature_cols if c not in all_nan_cols]

        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_train.columns,
            index=X_test.index
        )

        # 5. 只訓練一次模型
        model = train_model(X_train, y_train, args.model)

        # 6. Train / internal test 評估
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mae_train, rmse_train, r2_train, smape_train = compute_metrics(y_train, y_train_pred)
        mae_test, rmse_test, r2_test, smape_test = compute_metrics(y_test, y_test_pred)

        print("\n=== Seen Data Evaluation ===")
        print(f"Train R2={r2_train:.6f}, SMAPE={smape_train:.2f}%")
        print(f"Test  R2={r2_test:.6f}, SMAPE={smape_test:.2f}%")

        # 7. 對 unseen model 裡面的每個 block 分別評估
        block_results = []

        # 這裡建議只評估多層 block，且 is_eval_eligible == 1
        df_eval_blocks = df_unseen_model[
            (df_unseen_model["preset_num"] == 1) &
            (df_unseen_model["is_single_layer"] == 0) &
            (df_unseen_model["is_eval_eligible"] == 1)
        ].copy()

        target_blocks = sorted(df_eval_blocks["block_index"].unique())

        print(f"📌 {unseen_model} preset A unseen blocks 數量: {len(target_blocks)}")

        for block_index in target_blocks:
            df_unseen_block = df_eval_blocks[df_eval_blocks["block_index"] == block_index].copy()

            if len(df_unseen_block) < 2:
                print(f"⏭️ Skip {unseen_model} block {block_index}: 樣本數不足")
                continue

            X_unseen = df_unseen_block[feature_cols]
            y_unseen = df_unseen_block["energy_mj"]

            X_unseen = X_unseen.replace([np.inf, -np.inf], np.nan)
            X_unseen = pd.DataFrame(
                imputer.transform(X_unseen),
                columns=X_train.columns,
                index=X_unseen.index
            )

            y_unseen_pred = model.predict(X_unseen)

            mae_unseen, rmse_unseen, r2_unseen, smape_unseen = compute_metrics(
                y_unseen, y_unseen_pred
            )

            preset_num = int(df_unseen_block["preset_num"].iloc[0])
            preset_name = df_unseen_block["preset_name"].iloc[0]
            local_block = int(df_unseen_block["local_block"].iloc[0])

            block_results.append({
                "unseen_model": unseen_model,
                "preset_num": preset_num,
                "preset_name": preset_name,
                "block_index": int(block_index),
                "local_block": local_block,
                "n_unseen": len(y_unseen),
                "mae": mae_unseen,
                "rmse": rmse_unseen,
                "r2": r2_unseen,
                "smape": smape_unseen,
                "true_std": float(np.std(y_unseen))
            })

            print(
                f"Unseen {unseen_model} block {local_block}: "
                f"R2={r2_unseen:.6f}, SMAPE={smape_unseen:.2f}%"
            )

            # 每個 block 各自存圖
            block_output_dir = os.path.join(
                universal_base_dir,
                f"{train_tag}__leave_{unseen_model}",
                args.model,
                f"preset{preset_name}",
                f"block{local_block}"
            )
            os.makedirs(block_output_dir, exist_ok=True)

            plot_prediction(
                y_unseen,
                y_unseen_pred,
                f"Unseen {unseen_model} Block {local_block} Prediction",
                os.path.join(
                    block_output_dir,
                    f"unseen_{unseen_model}_block{local_block}_{args.model}.png"
                )
            )

        # 8. 彙整該 unseen model 的所有 block 結果
        if len(block_results) == 0:
            print(f"⚠️ {unseen_model} 沒有可評估的 block")
            continue

        block_results_df = pd.DataFrame(block_results)

        avg_smape = block_results_df["smape"].mean()
        avg_r2 = block_results_df["r2"].mean()
        min_r2 = block_results_df["r2"].min()
        neg_r2_count = int((block_results_df["r2"] < 0).sum())
        valid_blocks = len(block_results_df)

        print(f"\n=== LOMO Summary: Leave {unseen_model} Out ===")
        print(f"Valid Blocks      : {valid_blocks}")
        print(f"Avg Unseen SMAPE  : {avg_smape:.2f}%")
        print(f"Avg Unseen R2     : {avg_r2:.6f}")
        print(f"Min Unseen R2     : {min_r2:.6f}")
        print(f"Negative R2 Blocks: {neg_r2_count}")

        
        model_output_dir = os.path.join(
            universal_base_dir,
            f"{train_tag}__leave_{unseen_model}",
            args.model
        )
        os.makedirs(model_output_dir, exist_ok=True)

        block_results_df.to_csv(
            os.path.join(model_output_dir, "block_level_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # 10. 儲存該 unseen model 的 summary
        summary_df = pd.DataFrame([{
            "unseen_model": unseen_model,
            "model_type": args.model,
            "valid_blocks": valid_blocks,
            "avg_unseen_smape": avg_smape,
            "avg_unseen_r2": avg_r2,
            "min_unseen_r2": min_r2,
            "negative_r2_blocks": neg_r2_count,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "unseen_samples": len(df_eval_blocks),
            "train_r2": r2_train,
            "test_r2": r2_test,
            "train_smape": smape_train,
            "test_smape": smape_test
        }])

        summary_df.to_csv(
            os.path.join(model_output_dir, "model_level_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修正點：加入 nargs='+' 才能接收多個檔案路徑
    parser.add_argument("--csv", required=True, nargs='+', help="Path to training csv file(s)")
    parser.add_argument("--model", default="lasso",
                        choices=["lasso", "xgboost", "mlp", "lightgbm", "extratree", "vote_ensemble", "stack_ensemble"])
    parser.add_argument(
        "--train_models",
        nargs="*",
        default=None,
        help="指定訓練用的 source_model，例如 ResNet50 VGG19"
    )

    parser.add_argument(
        "--unseen_models",
        nargs="*",
        default=None,
        help="指定要作為 unseen 的 source_model，例如 VGG16 MobileNetV2"
    )
    args = parser.parse_args()
    main(args)

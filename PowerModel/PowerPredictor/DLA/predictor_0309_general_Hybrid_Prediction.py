#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全域型通用預測器


predictor_0309_general_Hybrid_Prediction.py -> 是做跨模型通用預測器 (之前是單一的預測器)
會存在 Universal_Results/

訓練和評估機器學習模型 (Lasso, XGBoost, MLP, LightGBM, ExtraTrees)
來預測 DLA 上的能耗: 根據神經網路層/區塊的結構特徵，預測實際能耗（mJ）

- 目的: 預測不同 block 的能耗 (mJ)
- 資料集: 各模型的 CSV
- 評估方法: Leave-One-Block-Out (LOBO)
- 輸出: metrics.txt + prediction scatter plots + feature importance / coef

# 直接定義通用結果的根目錄 -> 要改成
universal_base_dir = "Universal_Results"


------通用型預測器-----------

python predictor_0309_general_Hybrid_Prediction.py \
  --model lasso \
  --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

python predictor_0309_general_Hybrid_Prediction.py \
  --model xgboost \
  --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

python predictor_0309_general_Hybrid_Prediction.py \
  --model mlp \
  --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

python predictor_0309_general_Hybrid_Prediction.py \
  --model lightgbm \
  --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

python predictor_0309_general_Hybrid_Prediction.py \
  --model extratree \
  --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

python predictor_0309_general_Hybrid_Prediction.py \
  --model vote_ensemble \
  --csv ./training_data_v3_add_mem_add_io_size/VGG16/VGG16_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/VGG19/VGG19_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/ResNet50/ResNet50_with_blockopt_preset_all_v3_mem.csv ./training_data_v3_add_mem_add_io_size/MobileNetV2/MobileNetV2_with_blockopt_preset_all_v3_mem.csv

python predictor_0309_general_Hybrid_Prediction.py \
  --model stack_ensemble \
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
    model_name = "Universal"  # 統一命名為 Universal

    print(f"📊 已合併資料集，總樣本數: {len(df)}")
    print(f"🔗 包含模型: {df['source_model'].unique()}")

    # if "block_index" not in df.columns:
    #     raise ValueError("需要有 block_index 欄位來進行 LOBO")
    
    # 2. 定義排除欄位 (加入我們新創的輔助欄位)
    EXCLUDE_FINAL = {
        "energy_mj", "block_index", "round", "source_name", "source_model", "global_block_id",
        "block_row", "is_single_layer", "is_eval_eligible",
        "preset_id", "preset_A", "preset_B", "preset_C", "preset_D"
    }
    # 3. 找出所有要跑 LOBO 的目標 Block (來自所有模型的 Preset A 多層 Block)
    # 我們這裡使用 global_block_id 來進行迭代
    df_A = df[df["preset_id"] == "A"]
    if df_A.empty:
        raise ValueError("❌ 找不到任何模型的 preset A 資料")

    # 篩選出多層的 Block 作為測試對象
    target_lobo_blocks = sorted(df_A[df_A["is_single_layer"] == 0]["global_block_id"].unique())
    print(f"🎯 跨模型 LOBO 目標數量: {len(target_lobo_blocks)}")
    
    

    # --- LOBO loop ---
    for block_index in target_lobo_blocks:
        # ===== Skip 條件設定 =====
        min_unseen_samples = 25

        # 將目前的 global_block_id 當作 Unseen，其餘當作 Seen (訓練+測試)
        df_unseen = df[df["global_block_id"] == block_index]
        df_seen = df[df["global_block_id"] != block_index]
        # 1) unseen 樣本數少於 25 就跳過
        unseen_count = len(df_unseen)
        if unseen_count < min_unseen_samples:
            print(
                f"⏭️ Skip {block_index}: "
                f"filtered unseen count={unseen_count} < {min_unseen_samples}"
            )
            continue
        
        
        # 取得純數值特徵
        all_numeric_cols = df_seen.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in all_numeric_cols if c not in EXCLUDE_FINAL]

        print(f"[{args.model}] Block {block_index}: 特徵數 {len(feature_cols)} → {feature_cols}")
        # 分割資料
        X_seen = df_seen[feature_cols]
        y_seen = df_seen["energy_mj"]
        X_unseen = df_unseen[feature_cols]
        y_unseen = df_unseen["energy_mj"]
        
        if X_seen.empty or X_unseen.empty:
            print(f"[Skip] Block {block_index} 數據不足。")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_seen, y_seen, test_size=0.2, random_state=42
        )

        # inf -> nan
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_unseen = X_unseen.replace([np.inf, -np.inf], np.nan)

        # drop all-NaN cols in train
        all_nan_cols = X_train.columns[X_train.isna().all()].tolist()
        if all_nan_cols:
            print(f"⚠️ Block {block_index}: train 全 NaN 欄位 {len(all_nan_cols)} 個，移除：{all_nan_cols}")
            X_train = X_train.drop(columns=all_nan_cols)
            X_test = X_test.drop(columns=all_nan_cols, errors="ignore")
            X_unseen = X_unseen.drop(columns=all_nan_cols, errors="ignore")
            feature_cols = [c for c in feature_cols if c not in all_nan_cols]

        # impute
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns, index=X_test.index)
        X_unseen = pd.DataFrame(imputer.transform(X_unseen), columns=X_train.columns, index=X_unseen.index)

        # train
        model = train_model(X_train, y_train, args.model)

        # predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_unseen_pred = model.predict(X_unseen)

        # eval
        print(f"\n=== Evaluation on Block {block_index} ===")
        mae_train, rmse_train, r2_train, smape_train = evaluate_and_print("Train", y_train, y_train_pred)
        mae_test, rmse_test, r2_test, smape_test = evaluate_and_print("Test", y_test, y_test_pred)
        mae_unseen, rmse_unseen, r2_unseen, smape_unseen = evaluate_and_print("Unseen Block", y_unseen, y_unseen_pred)

        std_unseen = float(np.std(y_unseen))
        print("🔍 Unseen Block True Energy StdDev:", std_unseen)

        # output dir mapping
        # base_dir = os.path.dirname(args.csv)

        # preset_id_num = block_index // 1000
        # preset_map = {1: "presetA", 2: "presetB", 3: "presetC", 4: "presetD"}
        # preset_name = preset_map.get(preset_id_num, "presetUnknown")
        # global_local_block = block_index % 1000

        # output_dir = os.path.join(
        #     base_dir,
        #     preset_name,
        #     f"{model_name}_block{global_local_block}_{args.model}"
        # )
        # os.makedirs(output_dir, exist_ok=True)
        
        # 4. 修改輸出路徑：依照來源模型分類存放
        s_model = df_unseen["source_model"].iloc[0]
        l_block = int(df_unseen["block_index"].iloc[0]) # 原始完整 ID (1000, 1001...)
        
        # 修正點 1：不要使用 os.path.dirname(args.csv)，因為 args.csv 是列表
        # 直接定義通用結果的根目錄
        universal_base_dir = "Universal_Results"
        # 修正點 2：計算 preset_id_num 必須使用「數值型」的 ID (l_block)
        # 否則 block_index // 1000 會因為 block_index 是字串而報錯
        preset_id_num = l_block // 1000
        preset_map = {1: "presetA", 2: "presetB", 3: "presetC", 4: "presetD"}
        preset_name = preset_map.get(preset_id_num, "presetUnknown")
        
        # 2. 關鍵修正：計算相對編號 (1000 -> 0, 1001 -> 1, 2005 -> 5)
        l_block = l_block % 1000
        
        # 修正點 3：重新構建 output_dir
        output_dir = os.path.join(
            universal_base_dir,
            s_model,
            preset_name,
            f"Universal_block{l_block}_{args.model}"
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 修正點 4：呼叫 save_metrics (將 local_block 改為 l_block)
        save_metrics(
            output_dir, 
            model_name,    # 這裡是 "Universal"
            l_block, 
            args.model,
            mae_train, rmse_train, r2_train,
            mae_test, rmse_test, r2_test,
            mae_unseen, rmse_unseen, r2_unseen,
            smape_train, smape_test, smape_unseen,
            std_unseen, model, X_train.columns,
            len(y_train), len(y_test), len(y_unseen)
        )

        # plots
        # 修正點 5：修改繪圖參數 (將 global_local_block 改為 l_block)
        plot_prediction(
            y_train, y_train_pred,
            f"Train Set (Block {l_block} of {s_model} Unseen)",
            os.path.join(output_dir, f"train_prediction_{s_model}_block{l_block}_{args.model}.png")
        )
        plot_prediction(
            y_test, y_test_pred,
            f"Test Set (Block {l_block} of {s_model} Unseen)",
            os.path.join(output_dir, f"test_prediction_{s_model}_block{l_block}_{args.model}.png")
        )
        plot_prediction(
            y_unseen, y_unseen_pred,
            f"Unseen Block {l_block} ({s_model}) Prediction",
            os.path.join(output_dir, f"unseen_block_prediction_{s_model}_block{l_block}_{args.model}.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修正點：加入 nargs='+' 才能接收多個檔案路徑
    parser.add_argument("--csv", required=True, nargs='+', help="Path to training csv file(s)")
    parser.add_argument("--model", default="lasso",
                        choices=["lasso", "xgboost", "mlp", "lightgbm", "extratree", "vote_ensemble", "stack_ensemble"])
    args = parser.parse_args()
    main(args)

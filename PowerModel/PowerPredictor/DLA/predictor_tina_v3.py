"""
訓練和評估機器學習模型(Lasso, XGBoost, MLP)來預測 DLA 上的能耗: 根據神經網路層的結構特徵，去預測該層在 Jetson DLA 上運行的實際能耗（mJ）
目的: 預測 DLA 上的不同計算區塊 (block) 的能耗 (mJ)
資料集: 來自不同神經網路架構 (VGG16、VGG19、ResNet50、MobileNetv2) 和資料集的 CSV 檔案
評估方法: Leave-One-Block-Out (LOBO) 交叉驗證
輸出: 評估指標 (MAE, RMSE, R², SMAPE, MAPE), 預測圖, 模型係數/特徵重要性


VGG16 + lasso model：
python predictor_tina_v3.py --csv ./training_data_v3/VGG16/VGG16_with_blockopt_preset_all.csv --model lasso

VGG16 + xgboost model：
python predictor_tina_v3.py --csv ./training_data_v3/VGG16/VGG16_with_blockopt_preset_all.csv --model xgboost

VGG16 + mlp model：
python predictor_tina_v3.py --csv ./training_data_v3/VGG16/VGG16_with_blockopt_preset_all.csv --model mlp

現在只測試VGG16 + ResNet50

ResNet50 + lasso model：
python predictor_tina_v3.py --csv ./training_data_v3/ResNet50/ResNet50_with_blockopt_preset_all.csv --model lasso

ResNet50 + xgboost：
python predictor_tina_v3.py --csv ./training_data_v3/ResNet50/ResNet50_with_blockopt_preset_all.csv --model xgboost

ResNet50 + mlp model：
python predictor_tina_v3.py --csv ./training_data_v3/ResNet50/ResNet50_with_blockopt_preset_all.csv --model mlp



---------
VGG19 + lasso model：
python predictor_tina_v3.py --csv ./training_data_v3/VGG19/VGG19_with_blockopt.csv --model lasso

VGG19 + xgboost model：
python predictor_tina_v3.py --csv ./training_data_v3/VGG19/VGG19_with_blockopt.csv --model xgboost

VGG19 + mlp model：
python predictor_tina_v3.py --csv ./training_data_v3/VGG19/VGG19_with_blockopt.csv --model mlp

MobileNetV2 + lasso model：
python predictor_tina_v3.py --csv ./training_data_v3/MobileNetV2/MobileNetV2_with_blockopt.csv --model lasso

MobileNetV2 + xgboost：
python predictor_tina_v3.py --csv ./training_data_v3/MobileNetV2/MobileNetV2_with_blockopt.csv --model xgboost

MobileNetV2 + mlp model：
python predictor_tina_v3.py --csv ./training_data_v3/MobileNetV2/MobileNetV2_with_blockopt.csv --model mlp

"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt

# 引入必要的函式庫
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import make_scorer

# --- 效能指標計算函式 (保持不變) ---

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

# --- 模型訓練函式 (保持不變) ---

def train_model(X_train, y_train, model_type):
    # Lasso 參數調整：為了適應未標準化/MinMaxScaled 的特徵，alpha 需要謹慎選擇。
    # 這裡採用您原始的固定 alpha，但建議使用 LassoCV 或 GridSearchCV。
    if model_type == 'lasso':
        # 建議：保留您的原始固定 alpha 邏輯
        model = Lasso(alpha=0.0001, max_iter=500000, positive = True, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=300, max_depth=6, subsample=0.8, colsample_bytree=0.8, gamma=0.1, learning_rate=0.05, random_state=42)
    elif model_type == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', validation_fraction=0.1, learning_rate='adaptive', max_iter=50000, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.fit(X_train, y_train)
    return model

# --- 評估與繪圖函式 (保持不變) ---

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
    plt.scatter(y_true, y_pred, color='blue', label='Predicted')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
            color='red', linestyle='--', label='Ideal (y=x)')
    plt.xlabel("True Energy (mJ)")
    plt.ylabel("Predicted Energy (mJ)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- 儲存指標函式 (保持不變) ---

def save_metrics(output_dir, model_name, block_index, model_type,
                mae_train, rmse_train, r2_train,
                mae_test, rmse_test, r2_test,
                mae_unseen, rmse_unseen, r2_unseen,
                smape_train, smape_test, smape_unseen,
                std_unseen, model, feature_names, full_df,
                n_train, n_test, n_unseen):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Unseen Block: {block_index}\n")
        f.write(f"Train Metrics: MAE={mae_train:.6f}, RMSE={rmse_train:.6f}, R2={r2_train:.6f}, SMAPE={(smape_train if not np.isnan(smape_train) else float('nan')):.2f}%\n")
        f.write(f"Test  Metrics: MAE={mae_test:.6f}, RMSE={rmse_test:.6f}, R2={r2_test:.6f}, SMAPE={(smape_test  if not np.isnan(smape_test)  else float('nan')):.2f}%\n")
        f.write(f"Unseen Metrics: MAE={mae_unseen:.6f}, RMSE={rmse_unseen:.6f}, R2={r2_unseen:.6f}, SMAPE={(smape_unseen if not np.isnan(smape_unseen) else float('nan')):.2f}%\n")
        f.write(f"Unseen StdDev: {std_unseen:.6f}\n")
        
        # 加入資料筆數統計
        f.write(f"Train Samples: {n_train}\n")
        f.write(f"Test Samples: {n_test}\n")
        f.write(f"Unseen Samples: {n_unseen}\n")

        
        
        if model_type == 'lasso':
            f.write("\n=== Non-zero Lasso Coefficients ===\n")
            for name, val in zip(feature_names, model.coef_):
                if abs(val) > 1e-6:
                    f.write(f"{name:35}: {val:.6f}\n")
            # 寫入截距
            f.write(f"{'__intercept__':35}: {float(model.intercept_):.6f}\n")
            
        elif model_type == 'xgboost':
            f.write("\n=== Top 10 Feature Importances (XGBoost) ===\n")
            importances = model.feature_importances_
            feature_importance = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
            for name, val in feature_importance:
                f.write(f"{name:35}: {val:.6f}\n")


# --- 主要邏輯函數 (新增特徵處理) ---

def main(args):
    df = pd.read_csv(args.csv) 
    model_name = os.path.basename(args.csv).split("_")[0] 

    if 'block_index' not in df.columns:
        raise ValueError("需要有 block_index 欄位來進行挑戰組分析")

    # 1. 特徵處理與準備
    # 移除非數值欄位，並保留所有數值特徵
    # 假設 preprocess.py 已經處理好所有的 Log, Mem, Density 欄位
    EXCLUDE_FINAL = {
        'energy_mj', 'block_index', 'round', 'source_name',
        'block_row', 'is_single_layer', 'is_eval_eligible',
        'preset_id', 'preset_A', 'preset_B', 'preset_C', 'preset_D'
        # 注意：preset_id（字串）排除
    }
    

    # 2. 確定 LOBO 的目標區塊 (原始邏輯)
    batch0_idxs = df.index[df['batch_size'] == 0].tolist()
    segments = []
    current_segment = [batch0_idxs[0]] 
    for i in range(1, len(batch0_idxs)):
        if batch0_idxs[i] == batch0_idxs[i - 1] + 1:
            current_segment.append(batch0_idxs[i])
        else:
            segments.append(current_segment)
            current_segment = [batch0_idxs[i]]
    segments.append(current_segment)

    block_sets = []
    for seg in segments[:3]:
        # 使用 loc 確保正確的索引
        block_set = set(df.loc[seg, 'block_index'].unique())
        block_sets.append(block_set)
    print(f"🔍 三個資料集的 Block Index 集合: {[sorted(list(bs)) for bs in block_sets]}")
    # 三個資料集的 block_index 交集
    # 三個資料集的 block_index 交集 (已儲存為 list)
    # target_lobo_blocks = sorted(list(set.intersection(*block_sets)))
    
    # 改成只測試preset A的block -> 當作LOBO的測試結果評估
    
    # --- LOBO：只測試 preset A 的多層 block（is_single_layer == 0）---
    if 'preset_id' not in df.columns:
        raise ValueError("❌ CSV 缺少 preset_id 欄位")

    df_A = df[df['preset_id'] == 'A']   # 只取 preset A

    if df_A.empty:
        raise ValueError("❌ 找不到 preset A 的資料")

    # 只取多層 block（is_single_layer = 0）
    if 'is_single_layer' not in df.columns:
        raise ValueError("❌ CSV 缺少 is_single_layer 欄位")

    lobo_blocks = df_A[df_A['is_single_layer'] == 0]['block_index'].unique()

    # 排序
    target_lobo_blocks = sorted(list(lobo_blocks))

    print(f"🎯 LOBO 目標 block（preset A + 多層）: {target_lobo_blocks}")


    
    
    
    # print(f"🎯 執行 LOBO 的 Block Index 交集: {target_lobo_blocks}")
    # 注意：這邊只有警告且把交集後的單層都移除出LOBO，現在的LOBO僅保留全部多層
    
    # --- 修正後的單層 Block 篩選與印出 ---
    if 'num_layers' in df.columns:
        # 找出所有 num_layers <= 1 的 Block 索引 (這是 set)
        single_layer_block_indices = set(df[df['is_single_layer'] == 1]['block_index'].unique())
        
        # 關鍵修正：將 target_lobo_blocks (list) 轉換回 set，再進行交集
        target_lobo_set = set(target_lobo_blocks) 

        common_single_layer_blocks = sorted(list(target_lobo_set.intersection(single_layer_block_indices)))

        print(f"🔬 根據 'is_single_layer == 1'，以下 Block 被視為單層 Block: {sorted(list(single_layer_block_indices))}")
        print(f"❌ 這些 Block 在 LOBO 交集中且被標記為單層: {common_single_layer_blocks}")
        
        # 移除單層 block
        target_lobo_blocks = [b for b in target_lobo_blocks if b not in common_single_layer_blocks]
        print(f"🔥 修正後 LOBO Block 清單（已排除單層）: {target_lobo_blocks}")
    else:
        print("⚠️ 警告：CSV 缺少 'num_layers' 欄位，無法識別單層 Block。")

    # target_lobo_blocks 目前是 list，為了執行差集，先轉為 set
    target_lobo_set = set(target_lobo_blocks) 

    # 3. 執行 LOBO 迴圈
    drop_cols_base = ['energy_mj', 'block_index', 'round', 'is_single_layer', 'is_eval_eligible', 'source_name', 'block_row'] # 擴展排除列表

    for block_index in target_lobo_blocks:
        
        # LOBO 方法
        df_unseen = df[df['block_index'] == block_index]
        df_seen = df[df['block_index'] != block_index]
        
        # 移除訓練與測試集中不需要的欄位 (非特徵)
        # 自動偵測所有數值欄位（包括 preset_one-hot）
        # 會自動包含 preset_A, preset_B, preset_C, preset_D
        all_numeric_cols = df_seen.select_dtypes(include=[np.number]).columns

        feature_cols = [c for c in all_numeric_cols if c not in EXCLUDE_FINAL]
        print(f"[{args.model}] Block {block_index}: 特徵數 {len(feature_cols)} → {feature_cols}")
        
        # 特徵矩陣
        X_seen = df_seen[feature_cols]
        y_seen = df_seen['energy_mj']

        X_unseen = df_unseen[feature_cols]
        y_unseen = df_unseen['energy_mj']

        print(f"[{args.model}] Block {block_index}: 特徵數 {len(feature_cols)}")
        
        if X_seen.empty or X_unseen.empty:
            print(f"[Skip] Block {block_index} 數據不足 (訓練集/測試集為空)。")
            continue
        
        # Seen 資料切分成 train / test
        X_train, X_test, y_train, y_test = train_test_split(X_seen, y_seen, test_size=0.2, random_state=42)

        # 訓練模型
        model = train_model(X_train, y_train, args.model)
        
        # 產生預測
        y_train_pred = model.predict(X_train) 
        y_test_pred = model.predict(X_test) 
        y_unseen_pred = model.predict(X_unseen) 

        # 評估三組預測結果
        print(f"\n=== Evaluation on Block {block_index} ===")
        mae_train, rmse_train, r2_train, smape_train = evaluate_and_print("Train", y_train, y_train_pred)
        mae_test, rmse_test, r2_test, smape_test     = evaluate_and_print("Test", y_test, y_test_pred)
        mae_unseen, rmse_unseen, r2_unseen, smape_unseen = evaluate_and_print("Unseen Block", y_unseen, y_unseen_pred)

        std_unseen = np.std(y_unseen)
        print("🔍 Unseen Block True Energy StdDev:", std_unseen)

        # Output directory (與您前一個版本的一致)
        base_dir = os.path.dirname(args.csv)
        # output_dir = os.path.join(base_dir, f"{model_name}_block{block_index}_{args.model}")

        
        # --- 將 block_index 反推回 preset 與 local block ---
        preset_id_num = block_index // 1000   # 1 = A, 2 = B, 3 = C, 4 = D
        preset_map = {1: "presetA", 2: "presetB", 3: "presetC", 4: "presetD"}
        preset_name = preset_map.get(preset_id_num, "presetUnknown")

        local_block = block_index % 1000  # 真正的 block 編號

        print(f"📁 Block {block_index} → {preset_name} → block{local_block}")

        # --- 分資料夾存放 ---
        output_dir = os.path.join(
            base_dir,
            preset_name,                      # 先分到 preset 的資料夾下
            f"{model_name}_block{local_block}_{args.model}"  # 再建立 block 與模型的資料夾
        )

        os.makedirs(output_dir, exist_ok=True)

        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True) 

        save_metrics(output_dir, model_name, local_block, args.model,
                    mae_train, rmse_train, r2_train,
                    mae_test, rmse_test, r2_test,
                    mae_unseen, rmse_unseen, r2_unseen,
                    smape_train, smape_test, smape_unseen, 
                    std_unseen, model, X_train.columns, df,
                    len(y_train), len(y_test), len(y_unseen))

        # Plot
        plot_prediction(
            y_train, y_train_pred,
            f"Train Set (Block {local_block} Unseen)",
            os.path.join(output_dir, f"train_prediction_{model_name}_block{local_block}_{args.model}.png")
        )

        plot_prediction(
            y_test, y_test_pred,
            f"Test Set (Block {local_block} Unseen)",
            os.path.join(output_dir, f"test_prediction_{model_name}_block{local_block}_{args.model}.png")
        )

        plot_prediction(
            y_unseen, y_unseen_pred,
            f"Unseen Block {local_block} Prediction",
            os.path.join(output_dir, f"unseen_block_prediction_{model_name}_block{local_block}_{args.model}.png")
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to training csv file')
    parser.add_argument('--model', default='lasso', choices=['lasso', 'xgboost', 'mlp'])
    args = parser.parse_args()
    main(args)
'''
cpu step2: 訓練CPU功耗模型(?) -> 用 sklearn 的 Ridge 或 GBDT
input:  cpu_power_data_freq.npz
output: cpu_power_predictor_use_prev_pred.joblib 

python train_cpu_power_model_sklearn_use_prev_pred.py

'''
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# --- 讀取數據 ---
data = np.load("cpu_power_data_freq.npz", allow_pickle=True)
power = data["powers"].astype(float).ravel()            # 功耗 (W)
total_usage = data["util_all"].astype(float).ravel()    # 整體 CPU 使用率 (%)
usage_per_policy = data["util_by_policy"].astype(float) # [K,T] 或 [T,K]
freq_per_policy = data["freq_by_policy"].astype(float)  # [K,T] 或 [T,K]
policy_names = data["policy_names"].tolist()

# --- 對齊形狀到 [num_clusters, num_timesteps] ---
if usage_per_policy.shape[0] > usage_per_policy.shape[1]:
    # 可能是 [T,K]，轉置
    usage_per_policy = usage_per_policy.T
    freq_per_policy = freq_per_policy.T
num_clusters, num_timesteps = usage_per_policy.shape
assert power.shape[0] == num_timesteps, f"功耗長度 {power.shape[0]} 與時間長度 {num_timesteps} 不一致"

# --- 估計採樣間隔（若有時間戳記） ---
if "ts" in data.files:
    timestamps = data["ts"].astype(float).ravel()
    sample_interval = np.median(np.diff(timestamps)) if timestamps.size > 1 else 0.2
else:
    sample_interval = 0.2

# --- 丟棄啟停 2 秒數據，避免過渡干擾 ---
cutoff_timesteps = int(max(1, round(2.0 / max(sample_interval, 1e-6))))
slice_data = slice(cutoff_timesteps, -cutoff_timesteps if cutoff_timesteps > 0 else None)
power = power[slice_data]
total_usage = total_usage[slice_data]
usage_per_policy = usage_per_policy[:, slice_data]
freq_per_policy = freq_per_policy[:, slice_data]
num_timesteps = power.shape[0]

# --- 衍生特徵 ---
# 1) 叢集層面：cluster_usage, cluster_freq, usage_freq_product, is_cluster_active (>3%), active_core_count
cores_per_cluster = np.ones(num_clusters) * 4  # 假設每個叢集有 4 核
is_cluster_active = (usage_per_policy > 3.0).astype(float)
active_core_count = np.round(usage_per_policy / 100.0 * cores_per_cluster[:, None])

# 2) 系統層面：total_usage_sum, total_active_cores, active_cluster_count, max_usage, usage_variance, usage_std_1s
def moving_std(data, window_size):
    if window_size <= 1:
        return np.zeros_like(data)
    padded_data = np.pad(data, (window_size-1, 0), mode='edge')
    std_values = np.lib.stride_tricks.sliding_window_view(padded_data, window_size).std(axis=1)
    return std_values[-len(data):]
window_size = max(1, int(round(1.0 / max(sample_interval, 1e-6))))
usage_std_1s = moving_std(total_usage, window_size)

def gini_coefficient(arr):
    sorted_arr = np.sort(arr)
    if sorted_arr.sum() == 0:
        return 0.0
    n = len(sorted_arr)
    return float(np.sum((2 * np.arange(1, n+1) - n - 1) * sorted_arr) / (n * np.sum(sorted_arr)))

features = []
prev_pred = 0.0  # 初始前一筆預測值為 0
for t in range(num_timesteps):
    cluster_usage = usage_per_policy[:, t]
    cluster_freq = freq_per_policy[:, t]
    # 叢集特徵
    feat = []
    feat.extend(cluster_usage.tolist())
    feat.extend(cluster_freq.tolist())
    feat.extend((cluster_usage * cluster_freq).tolist())
    feat.extend(is_cluster_active[:, t].tolist())
    feat.extend(active_core_count[:, t].tolist())
    # 系統特徵
    total_usage_sum = float(cluster_usage.sum())
    total_active_cores = float(active_core_count[:, t].sum())
    active_cluster_count = float(is_cluster_active[:, t].sum())
    feat += [total_usage_sum, total_active_cores, active_cluster_count,
            float(cluster_usage.max()), float(cluster_usage.var()), float(usage_std_1s[t])]
    # 時間相依特徵：前一筆預測值
    feat += [float(prev_pred)]
    features.append(feat)
    prev_pred = power[t]  # 使用當前實際功耗模擬下一次的預測值

feature_matrix = np.asarray(features, dtype=float)
target = power

# --- 時序切分 ---
train_size = int(num_timesteps * 0.8)
train_features = feature_matrix[:train_size]
test_features = feature_matrix[train_size:]
train_target = target[:train_size]
test_target = target[train_size:]

# --- 模型A：梯度提升樹（非線性） ---
gbdt_model = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
)
gbdt_model.fit(train_features, train_target)
gbdt_predictions = gbdt_model.predict(test_features)

# --- 模型B：Ridge回歸（線性） ---
ridge_model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
ridge_model.fit(train_features, train_target)
ridge_predictions = ridge_model.predict(test_features)

print(f"[GBDT] MAE={mean_absolute_error(test_target, gbdt_predictions):.3f}  R²={r2_score(test_target, gbdt_predictions):.3f}")
print(f"[Ridge] MAE={mean_absolute_error(test_target, ridge_predictions):.3f}  R²={r2_score(test_target, ridge_predictions):.3f}")

# --- 儲存模型與特徵資訊 ---
joblib.dump(
    {
        "gbdt_model": gbdt_model,
        "ridge_model": ridge_model,
        "feature_info": {
            "num_clusters": int(num_clusters),
            "feature_order": [
                "cluster_usage[k]*K", "cluster_freq[k]*K", "usage_freq_product[k]*K",
                "is_cluster_active[k]*K", "active_core_count[k]*K",
                "total_usage_sum", "total_active_cores", "active_cluster_count",
                "max_usage", "usage_variance", "usage_std_1s", "prev_pred"
            ],
            "policy_names": policy_names,
            "sample_interval": sample_interval,
        }
    },
    "cpu_power_predictor_use_prev_pred.joblib"
)
print("Saved: cpu_power_predictor_use_prev_pred.joblib")
'''
step2-1 在train完之後，可以用這個plot...畫圖來評估模型

評估並繪圖 CPU 功耗模型（sklearn 版本）
input:  cpu_power_data_freq.npz
output: sklearn_eval_plots_use_prev_pred/ 目錄下的圖表與指標
==================

指令：
python plot_cpu_power_eval_use_prev_pred.py

==================
[GBDT ][Core] R²=0.9773 | MAE=0.637 W | RMSE=0.847 W | Bias=-0.573 W | Hit@±0.2W=37.5% | Hit@±0.5W=52.6% | MAE_1s=0.606 W
[Ridge][Core] R²=0.9938 | MAE=0.277 W | RMSE=0.443 W | Bias=-0.087 W | Hit@±0.2W=54.3% | Hit@±0.5W=81.0% | MAE_1s=0.171 W

'''
import numpy as np, matplotlib.pyplot as plt, pandas as pd, joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

fn = "cpu_power_data_freq.npz"
d = np.load(fn, allow_pickle=True)
P    = d["powers"].astype(float).ravel()
allU = d["util_all"].astype(float).ravel()
U_p  = d["util_by_policy"].astype(float)
F_p  = d["freq_by_policy"].astype(float)
names= d["policy_names"].tolist()

# 對齊 [K,T]
if U_p.shape[0] > U_p.shape[1]:
    U_p = U_p.T; F_p = F_p.T
K, T = U_p.shape
assert len(P)==T

# 估 interval
if "ts" in d.files and len(d["ts"])>1:
    ts = d["ts"].astype(float).ravel()
    interval = float(np.median(np.diff(ts)))
else:
    interval = 0.2

# 去頭尾 2s
cut = max(1, round(2.0/max(interval,1e-6)))
cut = int(cut)
sl = slice(cut, -cut if cut>0 else None)
P, allU = P[sl], allU[sl]
U_p, F_p = U_p[:,sl], F_p[:,sl]
T = len(P)

# 特徵
def movstd(x, w):
    if w<=1: return np.zeros_like(x)
    pad = np.pad(x,(w-1,0),mode='edge')
    v = np.lib.stride_tricks.sliding_window_view(pad,w).std(axis=1)
    return v[-len(x):]
win = max(1, int(round(1.0/max(interval,1e-6))))
stdU_1s = movstd(allU, win)

Ck = np.ones(K)*4  # 先假設每叢集4核
on = (U_p>3.0).astype(float)
ac = np.round(U_p/100.0*Ck[:,None])

rows=[]
prev_pred = 0.0  # 初始前一筆預測值為 0
for t in range(T):
    u = U_p[:,t]; f = F_p[:,t]
    feats=[]
    feats += u.tolist()
    feats += f.tolist()
    feats += (u*f).tolist()
    feats += on[:,t].tolist()
    feats += ac[:,t].tolist()
    U = float(u.sum()); A=float(ac[:,t].sum()); C=float(on[:,t].sum())
    feats += [U,A,C,float(u.max()),float(u.var()),float(stdU_1s[t])]
    # 替換 prev_power 為 prev_pred
    feats += [float(prev_pred)]
    rows.append(feats)
    prev_pred = P[t]  # 使用當前實際功耗模擬下一次的預測值
X = np.asarray(rows, float); y=P

# 時序切分
n_tr = int(T*0.8)
Xtr,Xte = X[:n_tr],X[n_tr:]; ytr,yte = y[:n_tr],y[n_tr:]

# 模型
gbdt = GradientBoostingRegressor(loss="squared_error", n_estimators=500, max_depth=3, learning_rate=0.05)
gbdt.fit(Xtr,ytr); pred_g = gbdt.predict(Xte)
ridge = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
ridge.fit(Xtr,ytr); pred_r = ridge.predict(Xte)

# ===== 核心指標（你要的那組） =====
def eval_core(y_true, y_pred, dt):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_pred - y_true
    ae = np.abs(err)
    mae  = float(ae.mean())
    rmse = float(np.sqrt((err**2).mean()))
    bias = float(err.mean())
    # R²（與 sklearn 一致）
    r2 = 1.0 - ((err**2).sum() / ((y_true - y_true.mean())**2).sum())
    hit02 = float((ae <= 0.2).mean() * 100.0)
    hit05 = float((ae <= 0.5).mean() * 100.0)
    # 1 秒移動平均 MAE
    w = max(1, int(round(1.0/max(dt,1e-9))))
    def movavg(x, w):
        if w<=1: return x
        k = np.ones(w)/w
        return np.convolve(x, k, mode="valid")
    mae_1s = float(np.mean(np.abs(movavg(y_pred, w) - movavg(y_true, w))))
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Bias": bias, "Hit02": hit02, "Hit05": hit05, "MAE_1s": mae_1s}

core_g = eval_core(yte, pred_g, interval)
core_r = eval_core(yte, pred_r, interval)

print(f"[GBDT ][Core] R²={core_g['R2']:.4f} | MAE={core_g['MAE']:.3f} W | RMSE={core_g['RMSE']:.3f} W | "
      f"Bias={core_g['Bias']:.3f} W | Hit@±0.2W={core_g['Hit02']:.1f}% | Hit@±0.5W={core_g['Hit05']:.1f}% | "
      f"MAE_1s={core_g['MAE_1s']:.3f} W")

print(f"[Ridge][Core] R²={core_r['R2']:.4f} | MAE={core_r['MAE']:.3f} W | RMSE={core_r['RMSE']:.3f} W | "
      f"Bias={core_r['Bias']:.3f} W | Hit@±0.2W={core_r['Hit02']:.1f}% | Hit@±0.5W={core_r['Hit05']:.1f}% | "
      f"MAE_1s={core_r['MAE_1s']:.3f} W")

# 也保留原本的簡易指標輸出
mae_g, r2_g = mean_absolute_error(yte,pred_g), r2_score(yte,pred_g)
mae_r, r2_r = mean_absolute_error(yte,pred_r), r2_score(yte,pred_r)
# print(f"[GBDT ] MAE={mae_g:.3f} R2={r2_g:.3f}")
# print(f"[Ridge] MAE={mae_r:.3f} R2={r2_r:.3f}")

# ===== 圖表 =====
os.makedirs("sklearn_eval_plots_use_prev_pred", exist_ok=True)
t_axis = np.arange(len(yte))*interval

# 圖1：時間序（GBDT）
plt.figure()
plt.plot(t_axis, yte, label="True Power (W)")
plt.plot(t_axis, pred_g, label="Predicted (GBDT)")
plt.xlabel("Time in test split (s)"); plt.ylabel("Power (W)")
plt.title("CPU Power: True vs Predicted (GBDT) — Test")
plt.legend(); plt.tight_layout()
plt.savefig("sklearn_eval_plots_use_prev_pred/01_timeseries_gbdt.png", dpi=150)

# 圖2：散點（GBDT）
plt.figure()
plt.scatter(yte, pred_g, s=8)
mn, mx = float(min(yte.min(), pred_g.min())), float(max(yte.max(), pred_g.max()))
plt.plot([mn,mx],[mn,mx])
plt.xlabel("True Power (W)"); plt.ylabel("Predicted Power (W)")
plt.title("Scatter: True vs Predicted (GBDT) — Test")
plt.tight_layout()
plt.savefig("sklearn_eval_plots_use_prev_pred/02_scatter_gbdt.png", dpi=150)

# 圖3：殘差分佈（GBDT）
plt.figure()
plt.hist(pred_g - yte, bins=50)
plt.xlabel("Residual (W)"); plt.ylabel("Count")
plt.title("Residuals (GBDT) — Test")
plt.tight_layout()
plt.savefig("sklearn_eval_plots_use_prev_pred/03_residuals_gbdt.png", dpi=150)

# 圖4：時間序（Ridge）
plt.figure()
plt.plot(t_axis, yte, label="True Power (W)")
plt.plot(t_axis, pred_r, label="Predicted (Ridge)")
plt.xlabel("Time in test split (s)"); plt.ylabel("Power (W)")
plt.title("CPU Power: True vs Predicted (Ridge) — Test")
plt.legend(); plt.tight_layout()
plt.savefig("sklearn_eval_plots_use_prev_pred/04_timeseries_ridge.png", dpi=150)

# 圖5：散點（Ridge）
plt.figure()
plt.scatter(yte, pred_r, s=8)
mn, mx = float(min(yte.min(), pred_r.min())), float(max(yte.max(), pred_r.max()))
plt.plot([mn,mx],[mn,mx])
plt.xlabel("True Power (W)"); plt.ylabel("Predicted Power (W)")
plt.title("Scatter: True vs Predicted (Ridge) — Test")
plt.tight_layout()
plt.savefig("sklearn_eval_plots_use_prev_pred/05_scatter_ridge.png", dpi=150)

# 圖6：殘差分佈（Ridge）
plt.figure()
plt.hist(pred_r - yte, bins=50)
plt.xlabel("Residual (W)"); plt.ylabel("Count")
plt.title("Residuals (Ridge) — Test")
plt.tight_layout()
plt.savefig("sklearn_eval_plots_use_prev_pred/06_residuals_ridge.png", dpi=150)

# ===== 存指標 =====
pd.DataFrame({
    "Model":["GBDT","Ridge"],
    "MAE (W)":[mae_g,mae_r],
    "R^2":[r2_g,r2_r],
    "Samples":[len(yte),len(yte)]
}).to_csv("sklearn_eval_plots_use_prev_pred/metrics.csv", index=False)

# 你的 Core 指標版本
pd.DataFrame([
    {"Model":"GBDT","R^2":core_g["R2"],"MAE (W)":core_g["MAE"],"RMSE (W)":core_g["RMSE"],
     "Bias (W)":core_g["Bias"],"Hit@±0.2W (%)":core_g["Hit02"],"Hit@±0.5W (%)":core_g["Hit05"],
     "MAE_1s (W)":core_g["MAE_1s"], "Samples":len(yte)},
    {"Model":"Ridge","R^2":core_r["R2"],"MAE (W)":core_r["MAE"],"RMSE (W)":core_r["RMSE"],
     "Bias (W)":core_r["Bias"],"Hit@±0.2W (%)":core_r["Hit02"],"Hit@±0.5W (%)":core_r["Hit05"],
     "MAE_1s (W)":core_r["MAE_1s"], "Samples":len(yte)},
]).to_csv("sklearn_eval_plots_use_prev_pred/metrics_core.csv", index=False)

print("Saved figures to ./sklearn_eval_plots_use_prev_pred/")
print("Saved core metrics to ./sklearn_eval_plots_use_prev_pred/metrics_core.csv")
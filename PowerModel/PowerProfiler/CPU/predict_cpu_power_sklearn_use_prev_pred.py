"""
step3: 製作一個可以用 sklearn 模型來預測 CPU 功率的腳本


sklearn 推論腳本（支援 1D/多特徵兩種模型）
- 會自動從 joblib 讀出 estimator；若是 dict，預設挑 'ridge'（可用 --algo 選 'gbdt'）
- 1D 模式：只需 --cpu-usage
- 多特徵模式：需 --u 與 --f（以逗號分隔），以及 --prev-pred（前一筆預測值）
特徵順序與 plot_cpu_power_eval.py 相同：u_k, f_k, u_k*f_k, on_k, ac_k, [U,A,C,umax,uvar,stdU1s], prev_pred


============
指令：
sklearn 推論（支援你保存成 dict 的 joblib：model_ridge / model_gbdt / feature_info）
用法（多特徵）：

# 用 ridge（建議，因為你觀察到它較穩）
python predict_cpu_power_sklearn_use_prev_pred.py \
--model cpu_power_predictor_use_prev_pred.joblib --algo ridge \
--u 51.1,26.2,27.4 --f 2202,2202,730 --prev-pred 2.0

# 換 gbdt 看看
python predict_cpu_power_sklearn_use_prev_pred.py \
--model cpu_power_predictor_use_prev_pred.joblib --algo gbdt \
--u 51.1,26.2,27.4 --f 2202,2202,730 --prev-pred 2.0


python predict_cpu_power_sklearn_use_prev_pred.py \
--model cpu_power_predictor_use_prev_pred.joblib --algo ridge \
--u 1.575,23.425,0.000 --f 730,2202,730 --prev-pred 2.992869

python predict_cpu_power_sklearn_use_prev_pred.py \
--model cpu_power_predictor_use_prev_pred.joblib --algo ridge \
--u 0,75,0 --f 2202,2202,2202 --prev-pred 0.0
"""

import os, argparse, numpy as np, joblib, sys

def parse_list(s: str, dtype=float):
    return np.asarray([dtype(x) for x in s.split(",")], dtype=dtype) if s else None

def load_from_joblib(path: str, algo: str):
    obj = joblib.load(path)
    if hasattr(obj, "predict"):         # 直接是 estimator/pipeline
        return obj, None
    if isinstance(obj, dict):
        # 明確鍵名（你的檔就是這兩個）
        key = "ridge_model" if (algo or "ridge").lower() == "ridge" else "model_gbdt"
        if key in obj and hasattr(obj[key], "predict"):
            return obj[key], obj.get("feature_info", None)
        # 後備：找第一個能 predict 的
        for k, v in obj.items():
            if hasattr(v, "predict"):
                return v, obj.get("feature_info", None)
        raise ValueError("字典裡找不到可用的 estimator。鍵有：" + ",".join(obj.keys()))
    raise TypeError(f"不支援的模型型態：{type(obj)}")

def build_feature_row(u_pct, f_mhz, prev_pred, ck=None, stdU1s=0.0):
    """
    與 plot_cpu_power_eval.py 相同的單筆特徵：
      [u_k]*K, [f_k]*K, [u*f]_k*K, [on]_k*K, [ac]_k*K, U, A, C, u_max, u_var, stdU1s, prev_pred
    """
    u = np.asarray(u_pct, float).ravel()
    f = np.asarray(f_mhz, float).ravel()
    K = len(u)
    if ck is None: ck = np.ones(K)*4.0
    ck = np.asarray(ck, float).ravel()
    if len(f)!=K or len(ck)!=K:
        raise ValueError("u, f, ck 長度需一致")

    on = (u > 3.0).astype(float)
    ac = np.round(u/100.0 * ck)

    feats = []
    feats += u.tolist()
    feats += f.tolist()
    feats += (u*f).tolist()
    feats += on.tolist()
    feats += ac.tolist()
    U = float(u.sum()); A=float(ac.sum()); C=float(on.sum())
    feats += [U, A, C, float(u.max()), float(u.var()), float(stdU1s)]
    feats += [float(prev_pred)]
    return np.asarray(feats, float).reshape(1, -1)

def main():
    ap = argparse.ArgumentParser(description="Predict CPU power (W) with sklearn model saved as dict.")
    ap.add_argument("--model", "-m", default="cpu_power_predictor.joblib", help="joblib 模型路徑")
    ap.add_argument("--algo", choices=["ridge","gbdt"], default="ridge", help="選 ridge 或 gbdt（對應字典鍵）")
    ap.add_argument("--u",  required=True, help="每叢集 util%，逗號分隔，如: 50,25,25")
    ap.add_argument("--f",  required=True, help="每叢集 freq MHz，逗號分隔，如: 2202,2202,730")
    ap.add_argument("--ck", default="4,4,4",  help="每叢集核心數，逗號分隔，預設全為4，如: 4,4,4")
    ap.add_argument("--prev-pred", type=float, default=0.0, help="前一筆預測值 (W)，預設為 0.0")
    ap.add_argument("--stdU1s", type=float, default=0.0, help="allU 的 1秒移動標準差（單點通常給 0）")
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    est, finfo = load_from_joblib(model_path, args.algo)

    u = parse_list(args.u, float)
    f = parse_list(args.f, float)
    ck = parse_list(args.ck, float) if args.ck else None

    # 用 feature_info['K'] 做長度檢查（若存在）
    if finfo and "K" in finfo:
        K_expected = int(finfo["K"])
        if len(u) != K_expected or len(f) != K_expected or (ck is not None and len(ck)!=K_expected):
            sys.exit(f"輸入長度與模型期望不符：期望 K={K_expected}，收到 u={len(u)}, f={len(f)}, ck={len(ck) if ck is not None else 'None'}")

    X = build_feature_row(u_pct=u, f_mhz=f, prev_pred=args.prev_pred, ck=ck, stdU1s=args.stdU1s)
    y_pred = float(est.predict(X).ravel()[0])
    y_pred = max(0.0, y_pred)  # 將負值截斷為 0
    print(f"[{os.path.basename(model_path)} | {args.algo}] Predicted Power: {y_pred:.3f} W")

if __name__ == "__main__":
    main()
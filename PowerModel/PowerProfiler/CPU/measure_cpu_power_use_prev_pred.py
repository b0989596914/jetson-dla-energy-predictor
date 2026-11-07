#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4: 確認 cpu 功耗預測器在實際資料上的表現


measure_cpu_power_use_prev_pred.py
讀取 CPU/功耗 CSV 檔案（或日誌檔案），逐筆呼叫 predict_cpu_power_sklearn_use_prev_pred.py，
使用前一筆預測值 (prev_pred) 作為輸入參數，收集預測值，與實測功耗比較並畫圖。
****最後我要採用的版本****

用法範例：
python measure_cpu_power_use_prev_pred.py \
    --log cpu_power_data.csv \
    --log-type csv \
    --predict_script predict_cpu_power_sklearn_use_prev_pred.py \
    --model cpu_power_predictor_use_prev_pred.joblib \
    --algo ridge \
    --policies policy0,policy4,policy8 \
    --out-csv measure_use_prev_pred/eval_results.csv \
    --plot measure_use_prev_pred/eval_plot.png \
    --python-exec /home/sslab_agx/archiconda3/envs/test2/bin/python \
    # --max_samples 100

===結果===
2025-09-27 20:17:58,493 [INFO] MAE=0.3608 W, RMSE=0.4714 W

"""

import re
import os
import sys
import argparse
import subprocess
import shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# 創建 measure_use_prev_pred 資料夾
output_dir = "measure_use_prev_pred"
os.makedirs(output_dir, exist_ok=True)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, 'measure_cpu_power.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_csv(file_path: str, policy_order: list = None):
    """
    讀取 CSV 檔案，回傳解析後的資料。
    回傳: list of dict, 每筆包含 {time, all_util, policies: {name: (util, freq)}, power}
    """
    logger.info(f"開始解析 CSV 檔案: {file_path}")
    try:
        # 讀取 CSV
        df = pd.read_csv(file_path, sep=',', skipinitialspace=True)
        logger.info(f"CSV 檔案包含 {len(df)} 行數據")
        
        # 檢查必要欄位
        required_cols = ['Timestamp', 'Total_Utilization', 'Power_W']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV 缺少必要欄位: {required_cols}")

        # 自動偵測 policy 欄位或使用指定的 policy_order
        if policy_order is None:
            policy_cols = [col for col in df.columns if col.startswith('policy') and ('_util' in col or '_freq' in col)]
            policy_names = sorted(set(col.split('_')[0] for col in policy_cols))
            logger.info(f"自動偵測到 policy: {policy_names}")
        else:
            policy_names = policy_order
            logger.info(f"使用指定的 policy 順序: {policy_names}")

        # 檢查 policy 欄位是否齊全
        for p in policy_names:
            if f"{p}_util" not in df.columns or f"{p}_freq" not in df.columns:
                raise ValueError(f"CSV 缺少 {p} 的 util 或 freq 欄位")

        rows = []
        for idx, row in df.iterrows():
            # 處理時間戳：將 timestamp 轉為 HH:MM:SS 格式
            timestamp = pd.to_datetime(row['Timestamp'], unit='s').strftime('%H:%M:%S')
            power = row['Power_W'] if not pd.isna(row['Power_W']) else None
            all_util = row['Total_Utilization']
            
            # 收集 policy 數據，按指定順序提取 util 和 freq
            policies = {}
            for p in policy_names:
                util = row[f"{p}_util"]
                freq = row[f"{p}_freq"]
                policies[p] = (float(util), float(freq))
            
            # 略過 power 為 NaN 或 0.0 的行
            if power is None or power == 0.0:
                logger.debug(f"略過第 {idx} 行: Power_W={power}")
                continue
                
            rows.append({
                "time": timestamp,
                "all_util": float(all_util),
                "policies": policies,
                "power": float(power)
            })
        
        logger.info(f"解析到 {len(rows)} 筆有效數據")
        return rows, policy_names
    except Exception as e:
        logger.error(f"解析 CSV 檔案失敗: {e}")
        raise

def parse_log(file_path: str, policy_order: list = None):
    """
    讀取日誌檔案，回傳解析後的資料。
    回傳: list of dict, 每筆包含 {time, all_util, policies: {name: (util, freq)}, power}
    """
    logger.info(f"開始解析日誌檔案: {file_path}")
    line_re = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\]\s+ALL=([\d.]+)%\s+(.+?)\s+Power=([\d.]+)W"
    )
    policy_re = re.compile(r"(\w+):\s*util=([\d.]+)%\s*freq=([\d.]+)MHz")

    rows = []
    policy_names = policy_order if policy_order else []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                m = line_re.match(line)
                if not m:
                    logger.debug(f"略過第 {idx} 行: 格式不符")
                    continue

                time_str, all_util, policies_str, power = m.groups()
                all_util = float(all_util)
                power = float(power)

                if power == 0.0:
                    logger.debug(f"略過第 {idx} 行: Power_W={power}")
                    continue

                policies = {}
                for pm in policy_re.finditer(policies_str):
                    p_name, p_util, p_freq = pm.groups()
                    policies[p_name] = (float(p_util), float(p_freq))

                if not policy_order:
                    for p_name in policies:
                        if p_name not in policy_names:
                            policy_names.append(p_name)
                    policy_names.sort()
                    logger.info(f"自動偵測到 policy: {policy_names}")

                if not all(p in policies for p in policy_names):
                    logger.debug(f"略過第 {idx} 行: 缺少 policy {policy_names}")
                    continue

                rows.append({
                    "time": time_str,
                    "all_util": all_util,
                    "policies": policies,
                    "power": power
                })

        if not rows:
            raise ValueError("日誌檔案無有效數據行")
        
        logger.info(f"解析到 {len(rows)} 筆有效數據")
        return rows, policy_names
    except Exception as e:
        logger.error(f"解析日誌檔案失敗: {e}")
        raise

def run_predict(python_exec: str, predict_script: str, model: str, algo: str,
                u_vals, f_vals, prev_pred: float = 0.0):
    """
    呼叫 predict_cpu_power_sklearn_use_prev_pred.py 並回傳預測值(float)。
    使用前一筆預測值 (prev_pred) 作為輸入參數。
    """
    logger.info(f"執行預測: u_vals={u_vals}, f_vals={f_vals}, prev_pred={prev_pred}")
    u_str = ",".join(f"{x:.4f}".rstrip("0").rstrip(".") for x in u_vals)
    f_str = ",".join(f"{x:.0f}" for x in f_vals)
    cmd = [
        python_exec, predict_script,
        "--model", model, "--algo", algo,
        "--u", u_str, "--f", f_str,
        "--prev-pred", f"{prev_pred:.4f}",
    ]
    logger.debug(f"執行命令: {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"Predicted Power:\s*([0-9.-]+)", out)
        if proc.returncode != 0 or not m:
            logger.error(f"predict 腳本執行失敗或無法解析輸出:\ncmd={' '.join(shlex.quote(c) for c in cmd)}\n---OUTPUT---\n{out}")
            raise RuntimeError(f"predict 腳本執行失敗或無法解析輸出")
        predicted = float(m.group(1))
        predicted = max(0.0, predicted)
        logger.info(f"預測成功: Predicted Power={predicted}")
        return predicted, out.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"predict 腳本執行超時: cmd={' '.join(shlex.quote(c) for c in cmd)}")
        raise
    except Exception as e:
        logger.error(f"predict 腳本執行失敗: {e}\ncmd={' '.join(shlex.quote(c) for c in cmd)}\n---OUTPUT---\n{out}")
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="輸入的檔案（例如 cpu_power_data.csv 或 cpu_profiler_predict_test.txt）")
    ap.add_argument("--log-type", choices=["csv", "log"], default="csv", help="輸入檔案類型：csv 或 log")
    ap.add_argument("--predict_script", default="predict_cpu_power_sklearn_no_prev.py",
                    help="predict 腳本路徑（預設同資料夾）")
    ap.add_argument("--model", default="cpu_power_predictor_no_prev.joblib", help="joblib 模型路徑")
    ap.add_argument("--algo", choices=["ridge", "gbdt"], default="ridge", help="選擇 predict 腳本內的模型鍵")
    ap.add_argument("--policies", default=None,
                    help="使用的 policy 順序，逗號分隔（例：policy0,policy4,policy8）。未提供則自動從檔案欄位偵測。")
    ap.add_argument("--python-exec", default=sys.executable, help="執行 predict 腳本的 Python 路徑")
    ap.add_argument("--out-csv", default=os.path.join(output_dir, "eval_results.csv"), help="輸出評估結果 CSV")
    ap.add_argument("--plot", default=os.path.join(output_dir, "eval_plot.png"), help="輸出比較圖檔名")
    ap.add_argument("--max_samples", type=int, default=None, help="最多處理筆數（除錯用）")
    args = ap.parse_args()

    logger.info("程式啟動")
    logger.info(f"參數: {vars(args)}")

    if not os.path.exists(args.log):
        logger.error(f"輸入檔案不存在: {args.log}")
        sys.exit(1)

    try:
        if args.policies:
            policy_order = [p.strip() for p in args.policies.split(",") if p.strip()]
        else:
            policy_order = None
        
        if args.log_type == "csv":
            rows_raw, policy_order = parse_csv(args.log, policy_order)
        else:
            rows_raw, policy_order = parse_log(args.log, policy_order)
    except Exception as e:
        logger.error(f"無法解析檔案: {e}")
        sys.exit(1)

    if not rows_raw:
        logger.error("解析不到任何有效行，請確認檔案格式")
        sys.exit(1)

    filtered = [r for r in rows_raw if r["power"] is not None and all(p in r["policies"] for p in policy_order)]
    logger.info(f"過濾後剩餘 {len(filtered)} 筆有效數據")

    if len(filtered) < 2:
        logger.error("可用樣本不足（需要至少 2 筆含 power 的行）")
        sys.exit(1)

    if args.max_samples is not None:
        filtered = filtered[:args.max_samples]
        logger.info(f"限制最多處理 {args.max_samples} 筆數據")

    recs = []
    prev_pred = 0.0  # 初始預測值為 0
    for idx, r in enumerate(filtered):
        u_vals = [r["policies"][p][0] for p in policy_order]
        f_vals = [r["policies"][p][1] for p in policy_order]

        try:
            y_pred, raw_out = run_predict(args.python_exec, args.predict_script,
                                          args.model, args.algo, u_vals, f_vals, prev_pred)
        except Exception as e:
            logger.warning(f"第 {idx} 筆呼叫 predict 失敗，略過。原因：{e}")
            continue

        recs.append({
            "index": idx,
            "time": r["time"],
            "power_actual_W": r["power"],
            "power_pred_W": y_pred,
            **{f"u_{p}": r["policies"][p][0] for p in policy_order},
            **{f"f_{p}_MHz": r["policies"][p][1] for p in policy_order},
        })
        prev_pred = y_pred  # 更新前一筆預測值

    if not recs:
        logger.error("沒有成功的預測記錄")
        sys.exit(1)

    try:
        df = pd.DataFrame(recs)
        df.to_csv(args.out_csv, index=False)
        logger.info(f"已輸出 CSV: {args.out_csv} (rows={len(df)})")
    except Exception as e:
        logger.error(f"儲存 CSV 失敗: {e}")
        sys.exit(1)

    y_true = df["power_actual_W"].to_numpy()
    y_pred = df["power_pred_W"].to_numpy()
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    logger.info(f"MAE={mae:.4f} W, RMSE={rmse:.4f} W")

    try:
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual (W)")
        plt.plot(y_pred, label="Predicted (W)")
        plt.xlabel("Sample Index")
        plt.ylabel("Power (W)")
        plt.title(f"Actual vs Predicted (MAE={mae:.3f}W, RMSE={rmse:.3f}W)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        plt.close()
        logger.info(f"已輸出圖檔: {args.plot}")
    except Exception as e:
        logger.error(f"儲存圖檔 {args.plot} 失敗: {e}")
        sys.exit(1)

    try:
        detailed_plot = os.path.join(output_dir, "detailed_" + os.path.basename(args.plot))
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Actual (W)", marker='o')
        plt.plot(y_pred, label="Predicted (W)", marker='x')
        plt.xlabel("Sample Index")
        plt.ylabel("Power (W)")
        plt.title(f"Actual vs Predicted Power (MAE={mae:.3f}W, RMSE={rmse:.3f}W)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(detailed_plot, dpi=150)
        plt.close()
        logger.info(f"已輸出詳細圖檔: {detailed_plot}")
    except Exception as e:
        logger.error(f"儲存詳細圖檔 {detailed_plot} 失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# eval 結果畫圖
'''
python eval_plot_energy_results.py \
--dir ./profile_data_test/with_blockopt/VGG16/bs16 \
--engine VGG16_imagenet100_input224_with_blockopt_bs16


python eval_plot_energy_results.py \
--dir ./profile_data/with_blockopt/VGG16/bs16 \
--engine VGG16_imagenet100_input224_with_blockopt_bs16 \
--output_dir_name input224

-----
實際使用：
python eval_plot_energy_results.py \
--dir ./profile_data/with_blockopt/MobileNetV2/bs16 \
--engine MobileNetV2_imagenet100_input331_with_blockopt_bs16 \
--output_dir_name input331
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import math
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 無視窗環境也能存圖
import matplotlib.pyplot as plt


def find_engine_paths(root_dir: str, engine_name: str = None):
    """
    回傳 (power_results_csv, monitor_csv, engine_name_base)
    """
    if engine_name:
        base = os.path.join(root_dir, f"{engine_name}")
        power_csv = base + "_power_results.csv"
        mon_csv   = base + "_monitor.csv"
        if not os.path.isfile(power_csv):
            raise FileNotFoundError(f"找不到檔案: {power_csv}")
        # monitor 可能還沒產出，允許不存在
        return power_csv, (mon_csv if os.path.isfile(mon_csv) else None), engine_name

    # auto-discover
    candidates = sorted(glob.glob(os.path.join(root_dir, "*_power_results.csv")))
    if not candidates:
        raise FileNotFoundError(f"目錄下找不到 *_power_results.csv：{root_dir}")
    power_csv = candidates[0]
    engine_name_base = os.path.basename(power_csv).replace("_power_results.csv", "")
    mon_csv = os.path.join(root_dir, f"{engine_name_base}_monitor.csv")
    return power_csv, (mon_csv if os.path.isfile(mon_csv) else None), engine_name_base


def parse_power_results(power_csv: str):
    """
    解析 <engine>_power_results.csv
    - layer_rows: DataFrame with columns [layer, avg_time_ms, avg_energy_mJ, round?]
      （每回合都會寫一次 layer 平均，通常可跨回合平均）
    - round_rows: DataFrame columns [round, metric, value]
      metric in {
        AVG_TOTAL_POWER_W, AVG_CPU_PRED_POWER_W,
        AVERAGE_SINGLE_INFERENCE_DLA_ENERGY_mJ, TEST_ACCURACY
      }
    """
    layers = []
    rounds = []
    round_pat = re.compile(r"^ROUND_(\d+)_(.+)$")

    with open(power_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # CSV 很簡單，用逗號 split
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue

            key = parts[0]

            # ROUND 行
            m = round_pat.match(key)
            if m:
                r = int(m.group(1))
                metric_key = m.group(2)
                # value 在第三欄 (index=2)
                val = None
                if len(parts) >= 3:
                    try:
                        val = float(parts[2])
                    except Exception:
                        val = np.nan
                rounds.append({"round": r, "metric": metric_key, "value": val})
                continue

            # Layer 行："{ForeignNode[...]}","time_ms","energy_mJ"
            if key.startswith("{ForeignNode"):
                try:
                    time_ms = float(parts[1])
                except Exception:
                    time_ms = np.nan
                try:
                    energy_mJ = float(parts[2])
                except Exception:
                    energy_mJ = np.nan

                layers.append({
                    "layer": key,
                    "avg_time_ms": time_ms,
                    "avg_energy_mJ": energy_mJ
                })
                continue

            # 其它行忽略

    layer_df = pd.DataFrame(layers)
    round_df = pd.DataFrame(rounds)
    # ★ 新增：同一個 round/metric 多次寫入時，先彙整（平均）
    if not round_df.empty:
        round_df = (round_df
                    .groupby(["round", "metric"], as_index=False)
                    .agg(value=("value", "mean")))
    return layer_df, round_df


def plot_round_dla_energy(round_df: pd.DataFrame, out_png: str):
    """
    畫每回合 AVERAGE_SINGLE_INFERENCE_DLA_ENERGY_mJ 折線圖
    """
    df = round_df[round_df["metric"] == "AVERAGE_SINGLE_INFERENCE_DLA_ENERGY_mJ"].copy()
    if df.empty:
        print("[WARN] 此檔案沒有 ROUND_*_AVERAGE_SINGLE_INFERENCE_DLA_ENERGY_mJ")
        return
    df = df.sort_values("round")
    x = df["round"].values
    y = df["value"].values

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Single Inference DLA Energy (mJ)")
    plt.title("Per-round DLA Energy (single inference)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] 存圖：{out_png}")


def plot_round_power_compare(round_df: pd.DataFrame, out_png: str):
    """
    畫每回合 Avg Total Power vs Avg CPU Pred Power（兩條線）
    """
    df_tot = round_df[round_df["metric"] == "AVG_TOTAL_POWER_W"].copy()
    df_cpu = round_df[round_df["metric"] == "AVG_CPU_PRED_POWER_W"].copy()
    if df_tot.empty or df_cpu.empty:
        print("[WARN] 缺 AVG_TOTAL_POWER_W 或 AVG_CPU_PRED_POWER_W，略過此圖")
        return

    df_tot = df_tot.sort_values("round")
    df_cpu = df_cpu.sort_values("round")

    # 對齊 rounds
    common_rounds = sorted(set(df_tot["round"]).intersection(set(df_cpu["round"])))
    df_tot = df_tot[df_tot["round"].isin(common_rounds)]
    df_cpu = df_cpu[df_cpu["round"].isin(common_rounds)]

    x = common_rounds
    y1 = df_tot["value"].values
    y2 = df_cpu["value"].values

    plt.figure()
    plt.plot(x, y1, marker="o", label="Avg Total Power (W)")
    plt.plot(x, y2, marker="o", label="Avg CPU Pred Power (W)")
    plt.xlabel("Round")
    plt.ylabel("Power (W)")
    plt.title("Per-round Total vs CPU Pred Power")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] 存圖：{out_png}")


def plot_layer_energy_topk(layer_df: pd.DataFrame, out_png: str, topk: int = 15):
    """
    取 layer 平均能耗（mJ）Top-K 畫橫條圖。
    注意：CSV 每回合都寫一次 layer 平均，所以先 groupby layer 取均值。
    """
    if layer_df.empty:
        print("[WARN] 沒有 layer 資料，略過 layer 圖")
        return

    agg = (layer_df.groupby("layer", as_index=False)
           .agg(avg_time_ms=("avg_time_ms", "mean"),
                avg_energy_mJ=("avg_energy_mJ", "mean")))

    agg = agg.sort_values("avg_energy_mJ", ascending=False).head(topk)

    plt.figure(figsize=(8, max(4, int(0.3*len(agg)))))
    plt.barh(agg["layer"], agg["avg_energy_mJ"])
    plt.xlabel("Avg Energy per Layer (mJ)")
    plt.title(f"Top-{topk} Layer Energy (mean across rounds)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] 存圖：{out_png}")


def parse_monitor_csv(mon_csv: str):
    """
    讀取 monitor.csv，回傳 DataFrame（含 ts, total_power_W, cpu_pred_power_W, dla_power_W）
    """
    df = pd.read_csv(mon_csv)
    # 欄位名可能有空白，先 normalize
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["ts", "total_power_W", "cpu_pred_power_W"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"monitor.csv 缺欄位：{c}")

    # 轉型
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["total_power_W"] = pd.to_numeric(df["total_power_W"], errors="coerce")
    df["cpu_pred_power_W"] = pd.to_numeric(df["cpu_pred_power_W"], errors="coerce")
    df = df.dropna(subset=["ts"])

    # 推估 DLA
    df["dla_power_W"] = df["total_power_W"] - df["cpu_pred_power_W"]
    return df


def downsample_df(df: pd.DataFrame, max_points: int = 4000):
    """
    時間序列太長時等距抽樣，以免圖太大
    """
    n = len(df)
    if n <= max_points or max_points <= 0:
        return df
    step = math.ceil(n / max_points)
    return df.iloc[::step, :].reset_index(drop=True)


def plot_monitor_total_vs_cpu(df_mon: pd.DataFrame, out_png: str):
    df = df_mon.dropna(subset=["ts", "total_power_W", "cpu_pred_power_W"]).copy()
    if df.empty:
        print("[WARN] monitor.csv 沒有可用資料（total/cpu_pred），略過此圖")
        return

    df = downsample_df(df, max_points=4000)

    plt.figure()
    plt.plot(df["ts"].values, df["total_power_W"].values, label="Total Power (W)")
    plt.plot(df["ts"].values, df["cpu_pred_power_W"].values, label="CPU Pred Power (W)")
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Power (W)")
    plt.title("Monitor Timeline: Total vs CPU Pred")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] 存圖：{out_png}")


def plot_monitor_dla(df_mon: pd.DataFrame, out_png: str):
    df = df_mon.dropna(subset=["ts", "dla_power_W"]).copy()
    if df.empty:
        print("[WARN] monitor.csv 沒有可用 DLA 資料，略過此圖")
        return

    df = downsample_df(df, max_points=4000)

    plt.figure()
    plt.plot(df["ts"].values, df["dla_power_W"].values)
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Estimated DLA Power (W)")
    plt.title("Monitor Timeline: Estimated DLA Power")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] 存圖：{out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="包含 CSV 的資料夾（例如 ./profile_data/with_blockopt/VGG16/bs16）")
    ap.add_argument("--engine", default=None, help="engine 檔名前綴（例如 VGG16_imagenet100_input224_with_blockopt_bs16）")
    ap.add_argument("--topk", type=int, default=15, help="Layer 能耗圖顯示前幾名")
    
    ap.add_argument("--output_dir_name", required=True, help="所有圖形要存放的子資料夾名稱（例如 1, 2, 3 等數字）")

    args = ap.parse_args()

    output_dir = os.path.join(args.dir, args.output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 正在將圖形儲存至資料夾：{output_dir}")
    
    power_csv, mon_csv, eng = find_engine_paths(args.dir, args.engine)
    print(f"[INFO] power_results: {power_csv}")
    print(f"[INFO] monitor_csv  : {mon_csv if mon_csv else '(not found)'}")

    layer_df, round_df = parse_power_results(power_csv)

    # === 圖 1：每回合 DLA mJ ===
    out1 = os.path.join(output_dir, f"{eng}_round_dla_mJ.png")
    plot_round_dla_energy(round_df, out1)

    # === 圖 2：每回合 Total vs CPU Pred Power ===
    out2 = os.path.join(output_dir, f"{eng}_round_total_vs_cpu_W.png")
    plot_round_power_compare(round_df, out2)

    # === 圖 3：Layer Top-K 能耗 ===
    out3 = os.path.join(output_dir, f"{eng}_layer_energy_top{args.topk}.png")
    plot_layer_energy_topk(layer_df, out3, topk=args.topk)

    # === 圖 4/5：monitor 時間序列 ===
    if mon_csv:
        df_mon = parse_monitor_csv(mon_csv)
        out4 = os.path.join(output_dir, f"{eng}_monitor_total_vs_cpu_W.png")
        plot_monitor_total_vs_cpu(df_mon, out4)

        out5 = os.path.join(output_dir, f"{eng}_monitor_dla_W.png")
        plot_monitor_dla(df_mon, out5)
    else:
        print("[WARN] 找不到 monitor.csv，略過時間序列圖。")

    # 額外印出簡單統計
    df_dla = round_df[round_df["metric"] == "AVERAGE_SINGLE_INFERENCE_DLA_ENERGY_mJ"]
    if not df_dla.empty:
        v = df_dla["value"].dropna().values
        if len(v) > 0:
            print(f"[STAT] DLA energy (mJ) over rounds: mean={np.mean(v):.2f}, std={np.std(v, ddof=1):.2f}, n={len(v)}")


if __name__ == "__main__":
    main()

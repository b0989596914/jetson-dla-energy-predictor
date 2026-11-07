'''
step1-1 (單純紀錄一個之後可以驗證的檔案)
監測並紀錄CPU使用率與功耗數據
製作cpu_profiler_predict_test.txt、cpu_power_data.txt ->  這是用來驗證cpu predictor足夠準確
(不加入npz檔案訓練，純粹用來驗證預測器)

----------------------------------指令---------------------------------------------

sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python cpu_profiler_to_txt_for_validate_predictor.py


stress-ng --cpu 12 --cpu-load 90 --timeout 10s

stress-ng --cpu 12 --cpu-load 30 --timeout 10s
stress-ng --cpu 12 --cpu-load 60 --timeout 10s
stress-ng --cpu 12 --cpu-load 80 --timeout 10s
stress-ng --cpu 11 --cpu-load 90 --timeout 10s

stress-ng --cpu 12 --cpu-load 50 --timeout 10s
stress-ng --cpu 5 --cpu-load 55 --timeout 10s
stress-ng --cpu 7 --cpu-load 67 --timeout 10s
stress-ng --cpu 10 --cpu-load 75 --timeout 10s



'''

from pathlib import Path
import os, time, psutil, numpy as np, threading, argparse

SAVE_PATH = "cpu_power_data.csv"  # 改為 csv 檔案
recording_flag = False  # 控制監測是否啟動
# LOG_FILE_DEFAULT = "cpu_profiler_predict_test.txt"

def write_log(msg, log_path):
    print(msg, flush=True)  # 照常印到螢幕
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"[log error] {e}", flush=True)

# ───────── 探測 cluster(policy) 與對應 CPU 清單 ─────────
def discover_policies():
    """
    回傳 (policy_names(固定順序), cluster_map)
    cluster_map: { 'policy0':[0,1,2,3], 'policy4':[4,5,6,7], ... }
    """
    base = Path("/sys/devices/system/cpu/cpufreq")
    names, cmap = [], {}
    for pol in sorted(base.glob("policy*")):
        rc = pol / "related_cpus"
        if not rc.exists(): 
            continue
        cpus = list(map(int, rc.read_text().split()))
        if not cpus:
            continue
        names.append(pol.name)
        cmap[pol.name] = cpus
    if not names:
        names = ["cpu0"]
        cmap["cpu0"] = [0]
    return names, cmap

# 讀目前 policy 頻率
def read_policy_cur_freq_mhz(policy_name):
    base = Path("/sys/devices/system/cpu/cpufreq") / policy_name
    f = base / "scaling_cur_freq"
    try:
        return int(f.read_text().strip()) / 1000.0
    except Exception:
        return float("nan")

def read_all_policy_freqs_mhz(policy_names):
    return [read_policy_cur_freq_mhz(p) for p in policy_names]

# ───────── 將 per-CPU usage 轉成各 policy 的平均 usage ─────────
def avg_util_by_policy(percpu_util, cluster_map, policy_names):
    vals = []
    for p in policy_names:
        idx = cluster_map[p]
        if len(idx) == 0:
            vals.append(float("nan"))
        else:
            vals.append(float(np.mean([percpu_util[i] for i in idx])))
    return vals

# ───────── hwmon 取功耗數據 ─────────
def get_power_from_hwmon():
    try:
        hwmon_base = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
        hwmon_dirs = [d for d in os.listdir(hwmon_base) if d.startswith('hwmon')]
        if not hwmon_dirs:
            print("No hwmon directory found!")
            return None
        hwmon_path = os.path.join(hwmon_base, hwmon_dirs[0])
        with open(os.path.join(hwmon_path, "in2_input"), "r") as f:
            voltage = int(f.read().strip()) / 1000.0  # 轉換為伏特(V)
        with open(os.path.join(hwmon_path, "curr2_input"), "r") as f:
            current = int(f.read().strip()) / 1000.0  # 轉換為安培(A)
        return voltage * current  # 計算功耗(W=V*I)
    except Exception as e:
        print(f"[hwmon error] {e}")
        return None

# ───────── 資料收集迴圈 ─────────
recording_flag = False
log_ts = []
log_power = []
log_util_all = []
log_util_by_policy = []  # 2D: [n_samples, n_policies]
log_freq_by_policy = []  # 2D: [n_samples, n_policies]
policy_names, cluster_map = discover_policies()  # 啟動時固定下來

# cpu 與功耗監測函式
def cpu_power_logger(interval=0.2):
    global log_ts, log_power, log_util_all, log_util_by_policy, log_freq_by_policy
    psutil.cpu_percent(interval=None, percpu=True)  # 觸發基準讀取
    while recording_flag:
        timestamp = time.time()
        util_all = psutil.cpu_percent(interval=None, percpu=False)
        util_percpu = psutil.cpu_percent(interval=None, percpu=True)
        util_p = avg_util_by_policy(util_percpu, cluster_map, policy_names)
        freq_p = read_all_policy_freqs_mhz(policy_names)
        power = get_power_from_hwmon()
        if power is not None:
            log_ts.append(timestamp)
            log_power.append(power)
            log_util_all.append(util_all)
            log_util_by_policy.append(util_p)
            log_freq_by_policy.append(freq_p)
            pairs = " | ".join([f"{name}: util={u:5.1f}% freq={f:4.0f}MHz"
                                for name, u, f in zip(policy_names, util_p, freq_p)])
            line = f"[{time.strftime('%H:%M:%S')}] ALL={util_all:5.1f}%  {pairs}  Power={power:6.4f}W"
            # print(line, flush=True)
            # write_log(line, LOG_FILE_DEFAULT)
        time.sleep(interval)

# 開始監測
def start_monitoring(interval=0.2):
    global recording_flag
    recording_flag = True
    thread = threading.Thread(target=cpu_power_logger, args=(interval,), daemon=True)
    thread.start()
    return thread

# 停止監測
def stop_monitoring():
    global recording_flag
    recording_flag = False
    time.sleep(1)

# 存檔到 txt
def save_to_file():
    if len(log_ts) == 0:
        print("[Error] No data collected.")
        return

    # 將數據寫入 txt 檔案
    try:
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            # 寫入標頭
            header = "Timestamp,Total_Utilization,Power_W," + ",".join([f"{p}_util,{p}_freq" for p in policy_names])
            f.write(header + "\n")
            
            # 寫入數據
            for ts, util_all, power, util_p, freq_p in zip(log_ts, log_util_all, log_power, log_util_by_policy, log_freq_by_policy):
                interleaved_data = ",".join([f"{u:5.1f},{f:4.0f}" for u, f in zip(util_p, freq_p)])
                line = f"{ts:.6f},{util_all:5.1f},{power:6.4f},{interleaved_data}"
                f.write(line + "\n")
            
        print(f"[✅ Saved] {SAVE_PATH}  samples={len(log_ts)}  policies={len(policy_names)}")
        print("policy_names:", policy_names)
        print("util_by_policy[0]:", log_util_by_policy[0] if log_util_by_policy else [])
        print("freq_by_policy[0]:", log_freq_by_policy[0] if log_freq_by_policy else [])
    except Exception as e:
        print(f"[Error] Failed to save to {SAVE_PATH}: {e}")

# ===== Main entry =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=50, help="Measurement duration in seconds.")
    parser.add_argument('--interval', type=float, default=0.2, help="Sampling interval in seconds.")
    # parser.add_argument('--log', type=str, default=LOG_FILE_DEFAULT, help="Path to append plain-text logs.")
    args = parser.parse_args()
    
    try:
        open(args.log, "w", encoding="utf-8").close()
    except Exception as e:
        print(f"[log init error] {e}", flush=True)

    print("Policies discovered:", discover_policies()[0])
    print(f"⏳ Monitoring CPU & power for {args.duration}s... Launch stress-ng now.")
    
    thread = start_monitoring(args.interval)
    time.sleep(args.duration)
    stop_monitoring()
    thread.join()
    save_to_file()
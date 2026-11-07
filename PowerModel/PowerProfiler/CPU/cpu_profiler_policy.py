'''
cpu step1: 先取CPU使用率與功耗數據

生成 cpu_power_data_freq.npz(現在用的，有各個policy) 檔案
-------------------------------------------------------------------------------
監測並紀錄CPU使用率與功耗數據
把結果存成 npz 檔案(numpy檔案)

----------------------------------指令---------------------------------------------

sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python cpu_profiler_policy.py

# stress-ng 是一個壓力測試工具，可以用來模擬CPU負載
# 用以模擬 12 個 CPU 核心的 90% 負載 -> 測試高負載下的性能 & 功耗
stress-ng --cpu 12 --cpu-load 90 --timeout 50s
stress-ng --cpu 12 --cpu-load 100 --timeout 50s
stress-ng --cpu 12 --cpu-load 30 --timeout 10s
stress-ng --cpu 12 --cpu-load 60 --timeout 10s
stress-ng --cpu 12 --cpu-load 80 --timeout 10s
stress-ng --cpu 11 --cpu-load 90 --timeout 10s

stress-ng --cpu 12 --cpu-load 50 --timeout 10s
stress-ng --cpu 5 --cpu-load 55 --timeout 10s
stress-ng --cpu 7 --cpu-load 67 --timeout 10s
stress-ng --cpu 10 --cpu-load 75 --timeout 10s


stress-ng --cpu 12 --cpu-load 50 --timeout 1s
stress-ng --cpu 5 --cpu-load 55 --timeout 1s
stress-ng --cpu 12 --cpu-load 90 --timeout 1s
stress-ng --cpu 12 --cpu-load 30 --timeout 1s
stress-ng --cpu 7 --cpu-load 67 --timeout 1s
stress-ng --cpu 12 --cpu-load 60 --timeout 1s
stress-ng --cpu 11 --cpu-load 90 --timeout 1s
stress-ng --cpu 10 --cpu-load 75 --timeout 1s
stress-ng --cpu 12 --cpu-load 80 --timeout 1s



'''

from pathlib import Path
import os, time, psutil, numpy as np, threading, argparse

SAVE_PATH = "cpu_power_data_freq.npz"
recording_flag = False  # 控制監測是否啟動
power_log = []  # 每筆為 (timestamp, usage, power) -> 儲存的列表

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
    # 若沒偵測到 policy（極少見），備援用 cpu0
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
        # 硬體監控路徑 -> 獲取電壓和電流 -> 計算功耗
        hwmon_base = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
        
        hwmon_dirs = [d for d in os.listdir(hwmon_base) if d.startswith('hwmon')]
        if not hwmon_dirs:
            print("No hwmon directory found!")
            return None
        hwmon_path = os.path.join(hwmon_base, hwmon_dirs[0])
        
        # 獲取電壓 (in2_input，單位：mV) 和電流 (curr2_input，單位：mA)
        with open(os.path.join(hwmon_path, "in2_input"), "r") as f:
            voltage = int(f.read().strip()) / 1000.0                    # 轉換為伏特(V)
        with open(os.path.join(hwmon_path, "curr2_input"), "r") as f:
            current = int(f.read().strip()) / 1000.0                    # 轉換為安培(A)
        return voltage * current                                        # 計算功耗(W=V*I)
    except Exception as e:
        print(f"[hwmon error] {e}")
        return None


# ───────── 資料收集迴圈 ─────────
recording_flag = False
log_ts = []
log_power = []
log_util_all = []
log_util_by_policy = []      # 2D: [n_samples, n_policies]
log_freq_by_policy = []      # 2D: [n_samples, n_policies]
policy_names, cluster_map = discover_policies()  # 啟動時固定下來


# cpu 與功耗監測函式
def cpu_power_logger(interval=0.2):
    # global power_log
    global log_ts, log_power, log_util_all, log_util_by_policy, log_freq_by_policy
    # 觸發 psutil 的第一個基準讀取（避免第一次 0% 現象）
    psutil.cpu_percent(interval=None, percpu=True)
    
    while recording_flag: # flag = true 時，週期性監測 CPU 使用率和功耗

        # 獲取當前時間戳
        timestamp = time.time()
        
        # 全機 & 每核心使用率
        util_all = psutil.cpu_percent(interval=None, percpu=False) # 獲取 CPU 當前 12 個 CPU 的"平均"使用率(百分比)
        util_percpu = psutil.cpu_percent(interval=None, percpu=True)
        
        # 各 policy 的平均使用率 & 目前頻率
        util_p = avg_util_by_policy(util_percpu, cluster_map, policy_names)
        freq_p = read_all_policy_freqs_mhz(policy_names)
        
        # 獲取當前功耗
        power = get_power_from_hwmon()

        if power is not None:
            log_ts.append(timestamp)
            log_power.append(power)
            log_util_all.append(util_all)
            log_util_by_policy.append(util_p)
            log_freq_by_policy.append(freq_p)

            # 印出一行摘要
            pairs = " | ".join([f"{name}: util={u:5.1f}% freq={f:4.0f}MHz"
                                for name, u, f in zip(policy_names, util_p, freq_p)])
            print(f"[{time.strftime('%H:%M:%S')}] ALL={util_all:5.1f}%  {pairs}  Power={power:6.4f}W")
            
        time.sleep(interval) # 取樣間隔時間

# 開始監測 : 管理監測執行緒
def start_monitoring(interval=0.2):
    global recording_flag       # 設置全域變數flag
    recording_flag = True
    
    # 啟用新執行緒運行 cpu_power_logger，設為 daemon 執行緒(會自動隨主程式結束而結束)
    thread = threading.Thread(target=cpu_power_logger, args=(interval,), daemon=True)  
    thread.start() 
    return thread

# 停止監測
def stop_monitoring():
    global recording_flag
    recording_flag = False  # 關閉旗標
    time.sleep(1)           # 等待1秒確保執行緒結束

# 存檔
def save_to_file():
    if len(log_ts) == 0:
        print("[Error] No data collected."); return

    ts = np.array(log_ts, dtype=float)
    power = np.array(log_power, dtype=float)
    util_all = np.array(log_util_all, dtype=float)
    util_mat = np.array(log_util_by_policy, dtype=float)     # shape = (N, P)
    freq_mat = np.array(log_freq_by_policy, dtype=float)     # shape = (N, P)
    pol_arr  = np.array(policy_names)                        # shape = (P,)
    
    # 若已存在檔案，只有在「policy_names 一致且 P 尺寸相同」時才做 append
    if os.path.exists(SAVE_PATH):
        try:
            with np.load(SAVE_PATH, allow_pickle=False) as z:
                prev_pol = z["policy_names"]
                if prev_pol.shape == pol_arr.shape and np.all(prev_pol == pol_arr):
                    ts   = np.concatenate([z["ts"], ts])
                    power= np.concatenate([z["powers"], power])
                    util_all = np.concatenate([z["util_all"], util_all])
                    util_mat = np.vstack([z["util_by_policy"], util_mat])
                    freq_mat = np.vstack([z["freq_by_policy"], freq_mat])
                    print(f"[Info] Appended to {SAVE_PATH}")
                else:
                    # policy 組合不同 → 另存新檔避免維度衝突
                    new_name = time.strftime("cpu_power_data_%Y%m%d_%H%M%S.npz")
                    np.savez(new_name,
                            ts=ts, powers=power, util_all=util_all,
                            util_by_policy=util_mat, freq_by_policy=freq_mat,
                            policy_names=pol_arr)
                    print(f"[Warn] Policy set changed. Saved to {new_name}"); return
        except Exception as e:
            print(f"[Warn] Failed to read old file ({e}). Will overwrite.")

    np.savez(SAVE_PATH,
            ts=ts, powers=power, util_all=util_all,
            util_by_policy=util_mat, freq_by_policy=freq_mat,
            policy_names=pol_arr)
    print(f"[✅ Saved] {SAVE_PATH}  samples={len(ts)}  policies={len(policy_names)}")
    print("policy_names:", policy_names)
    print("util_by_policy[0]:", util_mat[0] if len(util_mat)>0 else [])
    print("freq_by_policy[0]:", freq_mat[0] if len(freq_mat)>0 else [])

# ===== Main entry =====
if __name__ == "__main__":
    # 分析指令
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=50, help="Measurement duration in seconds.")  # 監測持續時間(預設50秒)
    parser.add_argument('--interval', type=float, default=0.2, help="Sampling interval in seconds.")  # 取樣間隔(預設0.2秒) -> 每0.2秒採樣一次
    args = parser.parse_args()
    
    print("Policies discovered:", discover_policies()[0])  # 檢視一次
    print(f"⏳ Monitoring CPU & power for {args.duration}s... Launch stress-ng now.")
    
    thread = start_monitoring(args.interval)
    time.sleep(args.duration)
    stop_monitoring()
    thread.join()
    save_to_file()

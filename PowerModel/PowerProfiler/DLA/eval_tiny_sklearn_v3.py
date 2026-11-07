"""
This file is used to test the accuracy of the engine on the tiny-imagenet dataset and to profile its power consumption. 10000 pics

without_blokopt:
VGG16:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG16/bs16/VGG16_tiny_without_blockopt_bs16.engine --num_runs=30
Inference Time: 184.78 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG16/bs8/VGG16_tiny_without_blockopt_bs8.engine --num_runs=60
Inference Time: 191.51 ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG16/bs4/VGG16_tiny_without_blockopt_bs4.engine --num_runs=100
Inference Time: 172.96 ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG16/bs1/VGG16_tiny_without_blockopt_bs1.engine --num_runs=150
Inference Time:  ms

VGG19:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG19/bs16/VGG19_tiny_without_blockopt_bs16.engine --num_runs=30
Inference Time: 230.85 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG19/bs8/VGG19_tiny_without_blockopt_bs8.engine --num_runs=60
Inference Time: 236.67 ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG19/bs4/VGG19_tiny_without_blockopt_bs4.engine --num_runs=100
Inference Time: 210.03 ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/VGG19/bs1/VGG19_tiny_without_blockopt_bs1.engine --num_runs=150
Inference Time:  ms

ResNet50:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/ResNet50/bs16/ResNet50_tiny_without_blockopt_bs16.engine --num_runs=30
Inference Time: 190.78 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/ResNet50/bs8/ResNet50_tiny_without_blockopt_bs8.engine --num_runs=60
Inference Time: 200.49 ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/ResNet50/bs4/ResNet50_tiny_without_blockopt_bs4.engine --num_runs=100
Inference Time: 191.86 ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/ResNet50/bs1/ResNet50_tiny_without_blockopt_bs1.engine --num_runs=100
Inference Time:  ms

MobileNetV2:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/MobileNetV2/bs16/MobileNetV2_tiny_without_blockopt_bs16.engine --num_runs=30
Inference Time: 192.92 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/MobileNetV2/bs8/MobileNetV2_tiny_without_blockopt_bs8.engine --num_runs=60
Inference Time: 188.39 ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/MobileNetV2/bs4/MobileNetV2_tiny_without_blockopt_bs4.engine --num_runs=100
Inference Time: 173.43 ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/without_blockopt/MobileNetV2/bs1/MobileNetV2_tiny_without_blockopt_bs1.engine --num_runs=100
Inference Time:  ms

with_blokopt:
VGG16:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG16/bs16/VGG16_tiny_input64_with_blockopt_bs16.engine --num_runs=30
Inference Time: 198.62 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG16/bs8/VGG16_tiny_with_blockopt_bs8.engine --num_runs=60
Inference Time:  ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG16/bs4/VGG16_tiny_with_blockopt_bs4.engine --num_runs=100
Inference Time:  ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG16/bs1/VGG16_tiny_with_blockopt_bs1.engine --num_runs=150
Inference Time:  ms


VGG19:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG19/bs16/VGG19_tiny_with_blockopt_bs16.engine --num_runs=30
Inference Time: 240.62 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG19/bs8/VGG19_tiny_with_blockopt_bs8.engine --num_runs=60
Inference Time:  ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG19/bs4/VGG19_tiny_with_blockopt_bs4.engine --num_runs=100
Inference Time:  ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/VGG19/bs1/VGG19_tiny_with_blockopt_bs1.engine --num_runs=150
Inference Time:  ms

ResNet50:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/ResNet50/bs16/ResNet50_tiny_with_blockopt_bs16.engine --num_runs=30
Inference Time: 285.88 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/ResNet50/bs8/ResNet50_tiny_with_blockopt_bs8.engine --num_runs=60
Inference Time:  ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/ResNet50/bs4/ResNet50_tiny_with_blockopt_bs4.engine --num_runs=100
Inference Time:  ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/ResNet50/bs1/ResNet50_tiny_with_blockopt_bs1.engine --num_runs=100
Inference Time:  ms

MobileNetV2:
bs16:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/MobileNetV2/bs16/MobileNetV2_tiny_with_blockopt_bs16.engine --num_runs=30
Inference Time: 215.11 ms

bs8:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/MobileNetV2/bs8/MobileNetV2_tiny_with_blockopt_bs8.engine --num_runs=60
Inference Time:  ms

bs4:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/MobileNetV2/bs4/MobileNetV2_tiny_with_blockopt_bs4.engine --num_runs=100
Inference Time:  ms

bs1:
sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn.py ./engine/with_blockopt/MobileNetV2/bs1/MobileNetV2_tiny_with_blockopt_bs1.engine --num_runs=100
Inference Time:  ms


這次嘗試

sudo -E /home/sslab_agx/archiconda3/envs/test2/bin/python eval_tiny_sklearn_v3.py ./engine/with_blockopt/VGG16/bs16/preset_B/VGG16_tiny_input64_with_blockopt_bs16_preset_B.engine --num_runs=30


"""

# -*- coding: utf-8 -*-
import argparse
import os
import re
import time
import csv
import hashlib
import threading
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import joblib
import psutil
import torch
import torchvision
import torchvision.transforms as transforms
import tensorrt as trt
from torch.utils.data import Subset

# ------------------------- argparse -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('engine', type=str, help='Path to the optimized TensorRT engine')
parser.add_argument("--num_dataset_chunks", type=int, default=5, help="Number of test dataset chunk")
parser.add_argument("--num_runs", type=int, default=20, help="Number of inference runs")
parser.add_argument("--num_total_profile_rounds", type=int, default=10, help="Number of total profiling rounds")
parser.add_argument("--monitor_interval", type=float, default=0.01, help="Sampling interval (seconds) for power/CPU monitor")

# sklearn CPU 功耗模型
parser.add_argument("--sk-model", default="cpu_power_predictor_use_prev_pred.joblib", help="joblib 模型路徑")
parser.add_argument("--algo", choices=["ridge", "gbdt"], default="ridge", help="選 ridge 或 gbdt（對應字典鍵）")
parser.add_argument("--u-thresh-on", type=float, default=3.0, help="on_k 判定門檻（%）")

# 暖機
parser.add_argument("--warmup_batches", type=int, default=3, help="每 round 正式量測前暖機的 batch 數")
parser.add_argument("--warmup_runs", type=int, default=3, help="暖機每個 batch 重複次數；None=沿用 num_runs")

parser.add_argument("--write-monitor", action="store_true", help="若指定，將輸出 <engine>_monitor.csv（含 total/cpu_pred/util/freq 等監測資料）")

args = parser.parse_args()

# ============================================================
# 工具：CPU policy 與頻率
# ============================================================
def discover_policies():
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

def read_policy_cur_freq_mhz(policy_name):
    base = Path("/sys/devices/system/cpu/cpufreq") / policy_name
    f = base / "scaling_cur_freq"
    try:
        return int(f.read_text().strip()) / 1000.0
    except Exception:
        return float("nan")

def read_all_policy_freqs_mhz(policy_names):
    return [read_policy_cur_freq_mhz(p) for p in policy_names]

def avg_util_by_policy(percpu_util, cluster_map, policy_names):
    vals = []
    for p in policy_names:
        idxs = cluster_map[p]
        vals.append(float(np.mean([percpu_util[i] for i in idxs])) if idxs else float("nan"))
    return vals

# ============================================================
# 解析 engine 檔名（支援 input<size>）
# ============================================================
def parse_engine_path(engine_path):
    engine_name = os.path.splitext(os.path.basename(engine_path))[0]

    # 允許 preset_X，也允許沒有 preset
    m = re.match(
        r"(?P<model>.+)_(?P<dataset>cifar10|tiny|imagenet100)_input(?P<input_size>\d+)_(?P<blockopt>with_blockopt|without_blockopt)_bs(?P<bs>\d+)(?:_preset_(?P<preset>[A-Z]))?",
        engine_name
    )
    if not m:
        raise ValueError(f"Engine name format error: {engine_name}")

    model      = m.group("model")
    dataset    = m.group("dataset")
    input_size = int(m.group("input_size"))
    blockopt   = m.group("blockopt")
    batch_size = int(m.group("bs"))
    preset     = m.group("preset") or "default"

    dataset_path = {
        "cifar10": "../../../Model/Data_cifar10/cifar10",
        "tiny": "../../../Model/Data_tiny_Imagenet/tiny-imagenet-200/val/sorted",
        "imagenet100": "../../../Model/Data_imagenet100/val"
    }[dataset]

    # preset 目錄支援
    output_dir = os.path.join("./profile_data", blockopt, model, f"bs{batch_size}", f"preset_{preset}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{engine_name}_power_results.csv")

    return model, dataset, dataset_path, batch_size, input_size, output_dir, csv_path, engine_name, preset

def image_sha256(image_tensor):
    return hashlib.sha256(image_tensor.numpy().tobytes()).hexdigest()

# ============================================================
# 載入 sklearn 模型（支援 dict 打包）
# ============================================================
def load_from_joblib(path: str, algo: str):
    obj = joblib.load(path)
    if hasattr(obj, "predict"):
        return obj, None
    if isinstance(obj, dict):
        key = "ridge_model" if (algo or "ridge").lower() == "ridge" else "model_gbdt"
        if key in obj and hasattr(obj[key], "predict"):
            return obj[key], obj.get("feature_info", None)
        for k, v in obj.items():
            if hasattr(v, "predict"):
                return v, obj.get("feature_info", None)
        raise ValueError("字典裡找不到可用的 estimator。鍵有：" + ",".join(obj.keys()))
    raise TypeError(f"不支援的模型型態：{type(obj)}")

def build_feature_row(u_pct, f_mhz, prev_pred, ck=None, stdU1s=0.0, u_thresh_on=3.0):
    u = np.asarray(u_pct, float).ravel()
    f = np.asarray(f_mhz, float).ravel()
    K = len(u)
    ck = np.ones(K) * 4.0 if ck is None else np.asarray(ck, float).ravel()
    if len(f) != K or len(ck) != K:
        raise ValueError("u, f, ck 長度需一致")
    on = (u > u_thresh_on).astype(float)
    ac = np.round(u/100.0 * ck)

    feats = []
    feats += u.tolist()
    feats += f.tolist()
    feats += (u*f).tolist()
    feats += on.tolist()
    feats += ac.tolist()
    U = float(u.sum()); A = float(ac.sum()); C = float(on.sum())
    feats += [U, A, C, float(u.max()), float(u.var()), float(stdU1s)]
    feats += [float(prev_pred)]
    X = np.asarray(feats, float).reshape(1, -1)
    expected = 5*len(u) + 7
    assert X.shape[1] == expected, f"feature length {X.shape[1]} != expected {expected}"
    return X

est, finfo = load_from_joblib(os.path.abspath(args.sk_model), args.algo)
print(f"[SKMODEL] path={os.path.abspath(args.sk_model)} algo={args.algo}  estimator={est.__class__.__name__}")

# ============================================================
# Power/CPU 監測
# ============================================================
recording_flag = True
# (timestamp, total_power, util_all, util_policies, freq_policies)
power_log = []

policy_names, cluster_map = discover_policies()
ck_per_policy = np.array([len(cluster_map[p]) for p in policy_names], dtype=float)

def get_power_from_hwmon():
    try:
        base = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
        dlist = [d for d in os.listdir(base) if d.startswith("hwmon")]
        if not dlist:
            return None
        hw = os.path.join(base, dlist[0])
        with open(os.path.join(hw, "in2_input")) as f:
            v = int(f.read().strip())/1000.0
        with open(os.path.join(hw, "curr2_input")) as f:
            c = int(f.read().strip())/1000.0
        return v*c
    except Exception:
        return None

def power_cpu_monitor_thread(interval=0.15):
    psutil.cpu_percent(interval=None, percpu=True)  # prime
    while recording_flag:
        ts = time.time()
        total_power   = get_power_from_hwmon()
        util_all      = psutil.cpu_percent(interval=None, percpu=False)
        util_percpu   = psutil.cpu_percent(interval=None, percpu=True)
        util_policies = avg_util_by_policy(util_percpu, cluster_map, policy_names)
        freq_policies = read_all_policy_freqs_mhz(policy_names)
        power_log.append((ts, total_power, util_all, util_policies, freq_policies))
        time.sleep(interval)

def start_power_cpu_monitoring():
    global recording_flag
    recording_flag = True
    t = threading.Thread(target=power_cpu_monitor_thread, daemon=True, args=(args.monitor_interval,))
    t.start()
    return t

def stop_power_cpu_monitoring():
    global recording_flag
    recording_flag = False

# ============================================================
# 準備資料集與 TensorRT
# ============================================================
(model_name, dataset, dataset_path,
    batch_size, input_size, output_dir,
    csv_output_path, engine_name, preset_name) = parse_engine_path(args.engine)

print(f"[INFO] Model={model_name}, Dataset={dataset}, Input={input_size}, BS={batch_size}, Preset={preset_name}")

# 監測 CSV
mon_csv_output_path = os.path.join(output_dir, f"{engine_name}_monitor.csv")

args.batch_size = batch_size
args.input_size = input_size
args.dataset_path = dataset_path
args.output_dir = output_dir

transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Change here to use the correct dataset
test_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)


layer_times = defaultdict(list)
class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.enabled = True
    def report_layer_time(self, layer_name, ms):
        if not self.enabled:
            return
        ln = layer_name.lower()
        if "foreignnode" not in ln: return
        if "copy" in ln or "reformatting" in ln: return
        layer_times[layer_name].append(ms)

def _interp_value(series, t):
    if not series:
        return float("nan")
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    for (t0, v0), (t1, v1) in zip(series, series[1:]):
        if t0 <= t <= t1:
            dt = (t1 - t0)
            if dt <= 0:
                return v0
            r = (t - t0) / dt
            return v0 + r * (v1 - v0)
    return float("nan")

def integrate_trapezoid(series, start, end):
    if not series or not (end > start):
        return float("nan")
    series = sorted(series, key=lambda x: x[0])
    pts = [(ts, val) for ts, val in series if start < ts < end]
    v_start = _interp_value(series, start)
    v_end   = _interp_value(series, end)
    pts = [(start, v_start)] + pts + [(end, v_end)]
    area = 0.0
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if np.isfinite(v0) and np.isfinite(v1) and (t1 > t0):
            area += 0.5 * (v0 + v1) * (t1 - t0)
    return area

logger = trt.Logger()
runtime = trt.Runtime(logger)
with open(args.engine, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
if context is None:
    raise RuntimeError("Failed to create execution context ...")

layer_prof = LayerProfiler()
context.profiler = layer_prof

# 綁定 I/O
input_binding_idx = engine.get_binding_index('input')
output_binding_idx = engine.get_binding_index('output')
input_shape = (args.batch_size, 3, args.input_size, args.input_size)
# Change here to reflect the correct number of classes for tiny-imagenet
output_shape = (args.batch_size, 200)
context.set_binding_shape(input_binding_idx, input_shape)
input_buffer  = torch.zeros(input_shape , dtype=torch.float32, device='cuda')
output_buffer = torch.zeros(output_shape, dtype=torch.float32, device='cuda')
bindings = [None, None]
bindings[input_binding_idx] = input_buffer.data_ptr()
bindings[output_binding_idx] = output_buffer.data_ptr()

print("Policies discovered:", policy_names)
for p in policy_names:
    print(f"  {p} -> CPUs {cluster_map[p]} (ck={len(cluster_map[p])})")
print(f"⏳ Start monitoring & inference. Sampling interval = {args.monitor_interval}s  |  "
      f"sklearn={args.algo}  model={os.path.basename(args.sk_model)}")

round_slices = []

for profile_round in range(args.num_total_profile_rounds):
    print(f"\n=== Profiling Round {profile_round+1}/{args.num_total_profile_rounds} ===")
    time_ranges = []
    inference_times = []
    total_correct = 0
    total_samples = 0

    total_size = len(test_dataset)
    if total_size == 0:
        print("⚠️  Dataset is empty. Skip inference.")
        continue
    chunk_size = max(1, total_size // max(1, args.num_dataset_chunks))
    print("Starting TensorRT inference with test dataset...")

    # 暖機：關閉 layer 記錄
    # ===== Warm-up =====  這邊是1125新修改的暖機方法
    if args.warmup_batches > 0:
        print(f"🔥 Warm-up: {args.warmup_batches} batches")

        dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size, 
                                device='cuda', dtype=torch.float32)

        # 綁定 input buffer（一次）
        input_buffer.copy_(dummy_input)

        layer_prof.enabled = False  # 暖機不記錄 layer time

        for _ in range(args.warmup_batches):
            for _ in range(args.warmup_runs):
                context.execute_async_v2(
                    bindings,
                    torch.cuda.current_stream().cuda_stream
                )

        # 最後一次同步，避免影響量測 -> 原本就有了不用重新跑
        torch.cuda.current_stream().synchronize()
        layer_prof.enabled = True

        print("✅ Warm-up done.")

    # 監測開始（暖機後）
    round_start_idx = len(power_log)
    mon_thread = start_power_cpu_monitoring()

    # 正式推論
    for i in range(args.num_dataset_chunks):
        print(f"\n=== Run {i+1}/{args.num_dataset_chunks} ===")
        start_idx = i * chunk_size
        end_idx = total_size if i == args.num_dataset_chunks - 1 else (i + 1) * chunk_size
        subset_dataset = Subset(test_dataset, list(range(start_idx, end_idx)))
        test_loader = torch.utils.data.DataLoader(
            subset_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )

        correct = 0
        total = 0

        for batch_idx, (image, label) in enumerate(test_loader):
            if batch_idx == 0:
                print(f"Run {i+1} - First batch first image hash: {image_sha256(image[0])}")

            actual_batch_size = int(image.shape[0])
            image_gpu = image.cuda(non_blocking=True)
            input_buffer[0:actual_batch_size].copy_(image_gpu)
            # torch.cuda.synchronize()

            start_time = time.time()
            for _ in range(args.num_runs):
                context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
            
            torch.cuda.current_stream().synchronize()
            end_time = time.time()

            output = output_buffer[0:actual_batch_size]
            predictions = output.argmax(dim=-1)
            correct += int(torch.sum(predictions == label.cuda()))
            total += actual_batch_size

            if batch_idx == 0:
                print("\n🔍 First few predictions vs ground truth:")
                for idx in range(min(5, actual_batch_size)):
                    print(f"Image {idx+1}: Pred = {predictions[idx].item()}, Label = {label[idx].item()}")

            time_ranges.append((start_time, end_time))
            inference_times.append((end_time - start_time) * 1000.0)

        acc = (correct / total) if total else 0.0
        total_correct += correct
        total_samples += total
        if inference_times:
            print(f"Run {i+1}/{args.num_dataset_chunks} - Last batch latency: {inference_times[-1]:.2f} ms - Acc: {acc:.4f}")
        else:
            print(f"Run {i+1}/{args.num_dataset_chunks} - (no batches)")

    # 監測結束並 join
    stop_power_cpu_monitoring()
    mon_thread.join(timeout=1.0)

    round_end_idx = len(power_log)
    round_slices.append((round_start_idx, round_end_idx))

    # 本 round 監測資料
    round_power_log = power_log[round_start_idx:round_end_idx]
    if not time_ranges:
        print("⚠️  This round has no time windows. Skip energy computation and CSV writing for this round.")
        continue

    # 以 sklearn 模型預測 CPU 功耗並扣出 DLA
    one_sec_buf = deque()
    prev_pred = 0.0
    cpu_pred_series = []
    for (ts, total_power, util_all, util_policies, freq_policies) in round_power_log:
        one_sec_buf.append((ts, util_all))
        while one_sec_buf and (ts - one_sec_buf[0][0] > 1.0):
            one_sec_buf.popleft()
        stdU1s = float(np.std([u for _, u in one_sec_buf])) if len(one_sec_buf) >= 2 else 0.0
        try:
            X = build_feature_row(
                u_pct=util_policies,
                f_mhz=freq_policies,
                prev_pred=prev_pred,
                ck=ck_per_policy,
                stdU1s=stdU1s,
                u_thresh_on=args.u_thresh_on
            )
            cpu_pred = float(est.predict(X).ravel()[0])
            cpu_pred = max(0.0, cpu_pred)
        except Exception:
            cpu_pred = 0.0
        cpu_pred_series.append((ts, cpu_pred))
        prev_pred = cpu_pred

    total_series = [(ts, tot) for (ts, tot, *_rest) in round_power_log if tot is not None]
    dla_power_series = []
    for j, (ts, tot, *_rest) in enumerate(round_power_log):
        if tot is None:
            continue
        cpu = cpu_pred_series[j][1]
        dla_power_series.append((ts, max(0.0, tot - cpu)))
    dla_power_series = sorted(dla_power_series, key=lambda x: x[0])

    # 對每個 batch 視窗做梯形積分 → 單次推論能耗（mJ）
    batch_summary = []
    for idx, (s, e) in enumerate(time_ranges, 1):
        total_Energy_J = integrate_trapezoid(dla_power_series, s, e)
        if np.isfinite(total_Energy_J) and e > s:
            avg_dla_W = total_Energy_J / (e - s)
            per_run_mJ = (total_Energy_J / args.num_runs) * 1000.0
        else:
            avg_dla_W = float("nan")
            per_run_mJ = float("nan")
        batch_summary.append((idx, (e - s) * 1000.0, avg_dla_W, per_run_mJ))

    # 以 layer time 比例分攤能耗
    avg_layer_time_map = {}
    total_dla_time = 0.0
    for lname, times in layer_times.items():
        if times:
            avg_t = sum(times) / len(times)
            avg_layer_time_map[lname] = avg_t
            total_dla_time += avg_t

    d_layer_total_energy = defaultdict(float)
    d_layer_count = defaultdict(int)
    for _, _, _, per_run_mJ in batch_summary:
        if not np.isfinite(per_run_mJ):
            continue
        for lname, avg_t in avg_layer_time_map.items():
            portion = (avg_t / total_dla_time) if total_dla_time > 0 else 0.0
            d_layer_total_energy[lname] += portion * per_run_mJ
            d_layer_count[lname] += 1

    overall_avg_single_run_mJ = float(np.nanmean([row[3] for row in batch_summary])) if batch_summary else float("nan")
    # Change here to reflect the correct number of samples for tiny-imagenet
    test_acc_overall = (total_correct / total_samples) if total_samples else float("nan")

    # 額外 round 統計
    def _avg_in_window2(start, end, series):
        vals = [v for ts, v in series if (ts is not None and start <= ts <= end)]
        return (sum(vals)/len(vals)) if vals else float("nan")

    avg_total_power_W = float(np.nanmean([_avg_in_window2(s, e, total_series) for s, e in time_ranges])) if time_ranges else float("nan")
    avg_cpu_pred_W = float(np.nanmean([_avg_in_window2(s, e, cpu_pred_series) for s, e in time_ranges])) if time_ranges else float("nan")

    # 寫入 CSV（layer 平均時間與估能 + round 指標）
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_output_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for lname in avg_layer_time_map:
            avg_time_ms = avg_layer_time_map[lname]
            avg_energy_mJ = (d_layer_total_energy[lname] / max(1, d_layer_count[lname])) if d_layer_count[lname] else float("nan")
            writer.writerow([lname, f"{avg_time_ms:.2f}", f"{avg_energy_mJ:.6f}"])
        # Round-level summary（不再用 ROUND_x 前綴）
        writer.writerow([
            "AVG_TOTAL_POWER_W", "",
            f"{(avg_total_power_W if np.isfinite(avg_total_power_W) else float('nan')):.4f}"
        ])
        writer.writerow([
            "AVG_CPU_PRED_POWER_W", "",
            f"{(avg_cpu_pred_W if np.isfinite(avg_cpu_pred_W) else float('nan')):.4f}"
        ])
        writer.writerow([
            "AVERAGE_SINGLE_INFERENCE_DLA_ENERGY_mJ", "",
            f"{(overall_avg_single_run_mJ if np.isfinite(overall_avg_single_run_mJ) else float('nan')):.6f}"
        ])
        writer.writerow([
            "TEST_ACCURACY", "",
            f"{(test_acc_overall if np.isfinite(test_acc_overall) else float('nan')):.4f}"
        ])
        writer.writerow([])  # 空行分隔不同 round
if args.write_monitor:
    # ============================================================
    # 監測資料寫成一個 CSV：<engine>_monitor.csv（僅在 --write-monitor 時）
    # ============================================================
    monitor_headers = ["ts", "total_power_W", "cpu_pred_power_W", "util_all_percent"]
    monitor_headers += [f"util_{p}" for p in policy_names]
    monitor_headers += [f"freq_{p}_MHz" for p in policy_names]

    cpu_pred_all_series = []
    slices = round_slices if round_slices else [(0, len(power_log))]
    for (sidx, eidx) in slices:
        one_sec_buf_all = deque()
        prev_pred_all = 0.0
        for (ts, total_power, util_all, util_policies, freq_policies) in power_log[sidx:eidx]:
            one_sec_buf_all.append((ts, util_all))
            while one_sec_buf_all and (ts - one_sec_buf_all[0][0] > 1.0):
                one_sec_buf_all.popleft()
            stdU1s = float(np.std([u for _, u in one_sec_buf_all])) if len(one_sec_buf_all) >= 2 else 0.0
            try:
                X = build_feature_row(
                    u_pct=util_policies,
                    f_mhz=freq_policies,
                    prev_pred=prev_pred_all,
                    ck=ck_per_policy,
                    stdU1s=stdU1s,
                    u_thresh_on=args.u_thresh_on
                )
                cpu_pred = float(est.predict(X).ravel()[0])
                cpu_pred = max(0.0, cpu_pred)
            except Exception:
                cpu_pred = 0.0
            cpu_pred_all_series.append((ts, cpu_pred))
            prev_pred_all = cpu_pred

    cpu_pred_map = dict(cpu_pred_all_series)
    with open(mon_csv_output_path, mode='w', newline='') as fmon:
        w = csv.writer(fmon)
        w.writerow(monitor_headers)
        for (ts, total_power, util_all, util_policies, freq_policies) in power_log:
            tot = float(total_power) if total_power is not None else float("nan")
            cpu_pred = cpu_pred_map.get(ts, float("nan"))
            row = [f"{ts:.6f}", f"{tot:.6f}", f"{cpu_pred:.6f}", f"{util_all:.3f}"]
            row += [f"{(u if np.isfinite(u) else float('nan')):.3f}" for u in util_policies]
            row += [f"{(f if np.isfinite(f) else float('nan')):.3f}" for f in freq_policies]
            w.writerow(row)
if args.write_monitor:
    print(f"✅ Monitor CSV saved to: {mon_csv_output_path}")
else:
    print("ℹ️ Monitor CSV disabled (未指定 --write-monitor)。")

"""
This file is used to convert the model, transforming a PyTorch model into ONNX format.
Current experiment use batch size: 1, 4, 8, 16 ,32, 64

# 先確認進入 PowerModel/PowerProfiler/DLA/，再執行指令

# 此檔案作用 : 將權重檔案轉為 ONNX 檔案
# 此py檔案的output : "./onnx/VGG16/bs16/VGG16_cifar10_bs16.onnx" -> 之後用於 build_engine_general.py 轉為tensorRT engine

指令 -> run_all_export_model.sh
"""

import argparse
import torch
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

from Model.models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='Name of the model defined in MODELS')
parser.add_argument('output', type=str, help='Output ONNX file path')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--num_classes', type=int, default=10)   # 類別數
parser.add_argument('--input_size', type=int, required=True)
args = parser.parse_args()

# Initialize dummy input
dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size).cuda()

# Initialize model with input size awareness (if applicable)
try:
    model = MODELS[args.model_name](num_classes=args.num_classes, input_size=args.input_size).cuda().eval()
except TypeError:
    # Backward compatibility for models that do not accept input_size
    model = MODELS[args.model_name](num_classes=args.num_classes).cuda().eval()

# Load checkpoint if provided -> 有提供檢查點就載入訓練好的權重，若沒有 則使用隨機初始化model
if args.checkpoint_path is not None:
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    # opset_version=11
    opset_version=11 # 或 14, 15, 16 都可以嘗試
)

print(f"✅ Successfully exported {args.model_name} to {args.output}")

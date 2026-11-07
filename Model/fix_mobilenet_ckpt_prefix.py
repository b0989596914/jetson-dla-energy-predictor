import torch, os, sys

src = './checkpoints/MobileNetV2_tiny_bs64.pth'  # 舊檔
dst = './checkpoints/MobileNetV2_tiny_bs64.fixed.pth'  # 轉好的新檔

if len(sys.argv) >= 2: src = sys.argv[1]
if len(sys.argv) >= 3: dst = sys.argv[2]

assert os.path.isfile(src), f'not found: {src}'
ckpt = torch.load(src, map_location='cpu')

# 兼容 {'state_dict': ...} 或直接 state_dict
state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
new_state = {}

def strip_prefix(k, p):
    return k[len(p):] if k.startswith(p) else k

for k, v in state.items():
    k = strip_prefix(k, 'module.')
    # 目前模型要的是 'model.*' 前綴
    if not k.startswith('model.'):
        if k.startswith('features.') or k.startswith('classifier.') or k.startswith('features') or k.startswith('classifier'):
            k = 'model.' + k
    new_state[k] = v

# 輸出格式盡量與原檔一致
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt['state_dict'] = new_state
    torch.save(ckpt, dst)
else:
    torch.save(new_state, dst)

print(f'✅ wrote: {dst}  (keys: {len(new_state)})')

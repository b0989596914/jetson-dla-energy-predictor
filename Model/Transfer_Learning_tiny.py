# '''
# python Transfer_Learning_tiny.py VGG16 \
#     --dataset_path /home/sslab_agx/Desktop/DeepGreen/Model/Data_tiny_Imagenet/tiny-imagenet-200 \
#     --checkpoint_path ./checkpoints/VGG16_tiny_bs64.pth \
#     --batch_size 64 \
#     --epochs 20 \
#     --lr 3e-4

# '''#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Strong tiny-ImageNet trainer
# - 安全載入舊 checkpoint（自動忽略 shape 不相容的層，避免 Missing/Unexpected/size mismatch）
# - AMP (mixed precision) 可開關
# - CosineAnnealingLR 與 AdamW / SGD 選擇
# - Label Smoothing
# - 自動偵測 val 或 val/sorted
# - 保存 best 與 last
# """

# import os
# import argparse
# from pathlib import Path

# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from tqdm import tqdm

# from models import MODELS


# # -------------------- argparse --------------------
# def get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('model_name', type=str, help='Model name defined in MODELS')
#     p.add_argument('--batch_size', type=int, default=64)
#     p.add_argument('--lr', type=float, default=1e-4)
#     p.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'adamw', 'sgd'])
#     p.add_argument('--momentum', type=float, default=0.9)
#     p.add_argument('--weight_decay', type=float, default=1e-4)
#     p.add_argument('--epochs', type=int, default=200)
#     p.add_argument('--patience', type=int, default=20)
#     p.add_argument('--dataset_path', type=str, default='/home/sslab_agx/Desktop/DeepGreen/Model/Data_tiny_Imagenet/tiny-imagenet-200')
#     p.add_argument('--checkpoint_path', type=str, default=None, help='路徑存在時嘗試部分載入；若不存在則當作保存路徑')
#     p.add_argument('--num_workers', type=int, default=4)
#     p.add_argument('--label_smoothing', type=float, default=0.0)
#     p.add_argument('--amp', action='store_true', help='啟用 AMP 混合精度')
#     p.add_argument('--cosine', action='store_true', help='使用 CosineAnnealingLR')
#     return p.parse_args()


# # -------------------- utils --------------------
# def default_ckpt_paths(model_name: str, bs: int):
#     ckpt_dir = Path('./checkpoints'); ckpt_dir.mkdir(parents=True, exist_ok=True)
#     base = f"{model_name}_tiny_bs{bs}"
#     return ckpt_dir / f"{base}.pth", ckpt_dir / f"{base}_best.pth", ckpt_dir / f"{base}_last.pth"


# @torch.no_grad()
# def safe_partial_load(model: nn.Module, path: str):
#     """僅載入鍵存在且 shape 完全相符的權重；回傳統計資訊。"""
#     if not path or not os.path.isfile(path):
#         return {"loaded": 0, "total": 0, "dropped": 0, "used": False}

#     print(f"📦 Try partial load from: {path}")
#     ckpt = torch.load(path, map_location='cpu')
#     if isinstance(ckpt, dict) and 'state_dict' in ckpt:
#         ckpt = ckpt['state_dict']

#     msd = model.state_dict()
#     filtered = {k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}
#     dropped = len(ckpt) - len(filtered)
#     msd.update(filtered)
#     model.load_state_dict(msd, strict=False)
#     print(f"🧩 loaded {len(filtered)}/{len(ckpt)} tensors; dropped {dropped} (classifier 不相容屬正常)")
#     return {"loaded": len(filtered), "total": len(ckpt), "dropped": dropped, "used": True}


# def build_dataloaders(root: str, bs: int, nw: int):
#     # 官方 tiny-ImageNet 統計（也可用 ImageNet 統計）
#     mean = [0.4802, 0.4481, 0.3975]; std = [0.2302, 0.2265, 0.2262]

#     transform_train = transforms.Compose([
#         transforms.RandomResizedCrop(64),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])
#     transform_val = transforms.Compose([
#         transforms.Resize(64),
#         transforms.CenterCrop(64),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])

#     root = Path(root)
#     train_dir = root / 'train'
#     # 自動容錯：val 或 val/sorted 都可
#     val_dir = root / 'val' / 'sorted'
#     if not val_dir.is_dir():
#         val_dir = root / 'val'

#     train_dataset = ImageFolder(str(train_dir), transform=transform_train)
#     val_dataset = ImageFolder(str(val_dir), transform=transform_val)

#     train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
#                               num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
#     val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
#                             num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
#     print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)} | Classes: {len(train_dataset.classes)}")
#     return train_loader, val_loader, len(train_dataset), len(val_dataset)


# def build_optimizer(params, name: str, lr: float, momentum: float, wd: float):
#     name = name.lower()
#     if name == 'sgd':
#         return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
#     if name == 'adam':
#         return torch.optim.Adam(params, lr=lr, weight_decay=wd)
#     if name == 'adamw':
#         return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
#     raise ValueError(f'Unknown optimizer: {name}')


# # -------------------- main --------------------
# def main():
#     args = get_args()
#     NUM_CLASSES = 200

#     # 準備模型
#     model = MODELS[args.model_name](num_classes=NUM_CLASSES, input_size=64).cuda()

#     # 保存路徑
#     default_pth, best_pth, last_pth = default_ckpt_paths(args.model_name, args.batch_size)
#     save_path = args.checkpoint_path or str(default_pth)

#     # 嘗試安全載入（若路徑存在）
#     stats = safe_partial_load(model, save_path) if os.path.isfile(save_path) else {"used": False}
#     if not stats["used"]:
#         print("🚀 No compatible checkpoint loaded. Training from scratch / incompatible head will be re-trained.")
#     print(f"💾 Will save best to: {best_pth}")

#     # dataloader
#     train_loader, val_loader, n_train, n_val = build_dataloaders(args.dataset_path, args.batch_size, args.num_workers)

#     # 損失（可選 label smoothing）
#     criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()

#     # optimizer / scheduler
#     optimizer = build_optimizer(model.parameters(), args.optimizer, args.lr, args.momentum, args.weight_decay)
#     if args.cosine:
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/100)
#     else:
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

#     scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

#     # 早停
#     best_acc, patience_counter = 0.0, 0

#     for epoch in range(args.epochs):
#         # -------------------- train --------------------
#         model.train()
#         total_loss, correct = 0.0, 0
#         pbar = tqdm(train_loader, desc=f"Train {epoch+1:03d}/{args.epochs}", leave=False)
#         for x, y in pbar:
#             x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)
#             with torch.cuda.amp.autocast(enabled=args.amp):
#                 out = model(x)
#                 loss = criterion(out, y)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item() * x.size(0)
#             correct += (out.argmax(dim=1) == y).sum().item()
#             pbar.set_postfix(loss=total_loss / max(1, len(train_loader.dataset)), acc=correct / max(1, n_train))

#         train_loss = total_loss / n_train
#         train_acc = correct / n_train

#         # -------------------- val --------------------
#         model.eval()
#         val_loss, val_correct = 0.0, 0
#         with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp):
#             for x, y in tqdm(val_loader, desc="Val", leave=False):
#                 x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
#                 out = model(x)
#                 val_loss += F.cross_entropy(out, y, reduction='sum').item()
#                 val_correct += (out.argmax(dim=1) == y).sum().item()

#         val_loss /= n_val
#         val_acc = val_correct / n_val

#         scheduler.step()

#         print(f"[Epoch {epoch+1:03d}] "
#               f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
#               f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
#               f"LR: {scheduler.get_last_lr()[0]:.2e}")

#         # 保存 best / last
#         torch.save(model.state_dict(), last_pth)
#         if val_acc > best_acc:
#             torch.save(model.state_dict(), best_pth)
#             best_acc = val_acc
#             patience_counter = 0
#             print(f"✅ Improved! best_acc={best_acc:.4f} → saved to {best_pth}")
#         else:
#             patience_counter += 1
#             print(f"🕗 No improvement. Patience: {patience_counter}/{args.patience}")
#             if patience_counter >= args.patience:
#                 print("⛔ Early stopping triggered.")
#                 break

#     print(f"\n🎉 Finished. Best Val Acc: {best_acc:.4f}")
#     print(f"Best: {best_pth}\nLast: {last_pth}")


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transfer Learning on Tiny-ImageNet (mobileNet 風格)
- 與你的 train_tiny_imagenet.py 對齊：單一 checkpoint、同樣的 transforms/loop/早停
- MobileNetV2 轉移學習：ImageNet1k 預訓練、凍結骨幹（可解凍最後 N 個 blocks）、替換分類頭(200 類)
- 安全 partial load：只載入 shape 相符權重，避免分類頭不相容報錯
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# 你的自家模型（VGG/ResNet...）
try:
    from models import MODELS
except Exception:
    MODELS = {}

# torchvision MobileNetV2（支援舊/新 API）
from torchvision.models import mobilenet_v2
try:
    from torchvision.models import MobileNet_V2_Weights
    _HAS_WEIGHTS_ENUM = True
except Exception:
    MobileNet_V2_Weights = None
    _HAS_WEIGHTS_ENUM = False


# -------------------- argparse --------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('model_name', type=str, help='模型名稱（MobileNetV2 或 MODELS 中已註冊名稱）')

    # 與 train_tiny_imagenet.py 對齊的超參數
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=30)

    p.add_argument('--dataset_path', type=str,
                   default='/home/sslab_agx/Desktop/DeepGreen/Model/Data_tiny_Imagenet/tiny-imagenet-200')
    p.add_argument('--checkpoint_path', type=str, default=None)

    # dataloader
    p.add_argument('--num_workers', type=int, default=4)

    # 轉移學習選項（僅對 MobileNetV2 有效）
    p.add_argument('--pretrained', action='store_true', help='使用 ImageNet1k 預訓練權重')
    p.add_argument('--freeze_backbone', action='store_true', help='凍結 features 參數')
    p.add_argument('--unfreeze_last_n', type=int, default=0, help='解凍最後 N 個 features block')
    p.add_argument('--head_dropout', type=float, default=0.2, help='分類頭 Dropout 機率')
    return p.parse_args()


# -------------------- utils --------------------
def get_default_checkpoint_path(args):
    filename = f"{args.model_name}_tiny_bs{args.batch_size}.pth"
    Path("./checkpoints").mkdir(parents=True, exist_ok=True)
    return os.path.join("./checkpoints", filename)


@torch.no_grad()
def safe_partial_load(model: nn.Module, path: str):
    """僅載入鍵存在且 shape 完全相符的權重；忽略不相容層（如分類頭）。"""
    if not path or not os.path.isfile(path):
        return {"loaded": 0, "total": 0, "dropped": 0, "used": False}

    print(f"📦 Try partial load from: {path}")
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']

    msd = model.state_dict()
    filtered = {k: v for k, v in ckpt.items() if k in msd and v.shape == msd[k].shape}
    dropped = len(ckpt) - len(filtered)
    msd.update(filtered)
    model.load_state_dict(msd, strict=False)
    print(f"🧩 loaded {len(filtered)}/{len(ckpt)} tensors; dropped {dropped} (分類頭不相容屬正常)")
    return {"loaded": len(filtered), "total": len(ckpt), "dropped": dropped, "used": True}


def build_dataloaders(root: str, bs: int, nw: int):
    # tiny-ImageNet 統計
    mean = [0.4802, 0.4481, 0.3975]
    std  = [0.2302, 0.2265, 0.2262]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    root = Path(root)
    train_dir = root / 'train'
    val_dir = root / 'val' / 'sorted'
    if not val_dir.is_dir():
        val_dir = root / 'val'

    train_dataset = ImageFolder(str(train_dir), transform=transform_train)
    val_dataset   = ImageFolder(str(val_dir),   transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def build_optimizer(params, name: str, lr: float, momentum: float):
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f'Unknown optimizer: {name}')


# -------------------- model factory --------------------
def make_model(args, num_classes=200):
    name = args.model_name.lower()

    # MobileNetV2（轉移學習）
    if name in ('mobilenetv2', 'mobilenet_v2', 'mobilenet-v2'):
        if _HAS_WEIGHTS_ENUM:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if args.pretrained else None
            model = mobilenet_v2(weights=weights)
        else:
            model = mobilenet_v2(pretrained=bool(args.pretrained))

        in_dim = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=args.head_dropout),
            nn.Linear(in_dim, num_classes)
        )

        if args.freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
            if args.unfreeze_last_n > 0:
                # 解凍最後 N 個子模組
                for m in list(model.features.children())[-args.unfreeze_last_n:]:
                    for p in m.parameters():
                        p.requires_grad = True
        return model

    # 其餘（VGG/ResNet…）走你自家的 MODELS
    if name in {k.lower() for k in MODELS.keys()}:
        real_key = next(k for k in MODELS.keys() if k.lower() == name)
        return MODELS[real_key](num_classes=num_classes, input_size=64)

    raise ValueError(f"未知模型：{args.model_name}，請用 MobileNetV2 或在 MODELS 註冊。")


# -------------------- main (與你的 loop 對齊) --------------------
def main():
    args = get_args()
    NUM_CLASSES = 200

    # 準備模型
    model = make_model(args, num_classes=NUM_CLASSES).cuda()

    # checkpoint 路徑（單一檔案）
    if args.checkpoint_path is None:
        args.checkpoint_path = get_default_checkpoint_path(args)
        print(f"💾 Saving to: {args.checkpoint_path}")

    # 若存在則嘗試部分載入（避免分類頭不相容）
    if os.path.isfile(args.checkpoint_path):
        print(f"📦 Loading checkpoint from {args.checkpoint_path}")
        safe_partial_load(model, args.checkpoint_path)
    else:
        print("🚀 No checkpoint found or not specified. Training from scratch.")

    # dataloader
    train_loader, val_loader, n_train, n_val = build_dataloaders(
        args.dataset_path, args.batch_size, args.num_workers
    )

    # optimizer（只訓練 requires_grad=True 的參數，若凍結骨幹則只訓練頭/已解凍層）
    optimizer = build_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.optimizer, args.lr, args.momentum
    )

    # ===== 訓練流程（複製你原本腳本的節奏與輸出）=====
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct = 0.0, 0

        print(f"\n[Epoch {epoch+1:03d}] Training...")
        for x, y in tqdm(train_loader, desc="Train", leave=False):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()

        train_acc = correct / n_train
        train_loss = total_loss / max(1, len(train_loader))

        # --------- Validation ---------
        model.eval()
        val_loss, val_correct = 0.0, 0
        print(f"[Epoch {epoch+1:03d}] Validating...")
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val", leave=False):
                x, y = x.cuda(), y.cuda()
                out = model(x)
                val_loss += F.cross_entropy(out, y).item()
                val_correct += (out.argmax(dim=1) == y).sum().item()

        val_acc = val_correct / n_val
        val_loss /= max(1, len(val_loader))

        print(f"[Epoch {epoch+1:03d}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 單一 checkpoint：只存 best
        if val_acc > best_accuracy:
            print(f"✅ Saving checkpoint to {args.checkpoint_path}")
            torch.save(model.state_dict(), args.checkpoint_path)
            best_accuracy = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"🕗 No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("⛔ Early stopping triggered.")
                break

    print(f"\n🎉 Finished. Best Val Acc: {best_accuracy:.4f}")
    print(f"Best checkpoint: {args.checkpoint_path}")


if __name__ == '__main__':
    main()

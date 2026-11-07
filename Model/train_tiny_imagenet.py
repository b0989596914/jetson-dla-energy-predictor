"""
python train_tiny_imagenet.py VGG16  --batch_size 64  / acc: 0.6422  -> 重新train好了

python train_tiny_imagenet.py VGG19  --batch_size 64  / acc:0.6530

python train_tiny_imagenet.py ResNet50  --batch_size 64  / acc:0.6508

python train_tiny_imagenet.py MobileNetV2  --batch_size 64  / acc:0.5764 -> 重新train好了 0.5697
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from models import MODELS
from tqdm import tqdm

# -------------------- argparse --------------------
parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='Model name defined in MODELS')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--dataset_path', type=str, default='/home/sslab_agx/Desktop/DeepGreen/Model/Data_tiny_Imagenet/tiny-imagenet-200')
parser.add_argument('--checkpoint_path', type=str, default=None)
args = parser.parse_args()

# -------------------- checkpoint path --------------------
def get_default_checkpoint_path(args):
    filename = f"{args.model_name}_tiny_bs{args.batch_size}.pth"
    return os.path.join("./checkpoints", filename)

# -------------------- model --------------------
NUM_CLASSES = 200
model = MODELS[args.model_name](num_classes=NUM_CLASSES, input_size=64).cuda()

if args.checkpoint_path is None:
    os.makedirs("./checkpoints", exist_ok=True)
    args.checkpoint_path = get_default_checkpoint_path(args)
    print(f"💾 Saving to: {args.checkpoint_path}")

# Resume from checkpoint if exists
if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
    print(f"📦 Loading checkpoint from {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path))
else:
    print("🚀 No checkpoint found or not specified. Training from scratch.")
# -------------------- optimizer --------------------
optimizer = {
    'sgd': lambda: torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum),
    'adam': lambda: torch.optim.Adam(model.parameters(), lr=args.lr)
}[args.optimizer]()

# -------------------- transforms --------------------
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
])

transform_test = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]),
])

# -------------------- dataset & dataloader --------------------
train_dataset = ImageFolder(os.path.join(args.dataset_path, 'train'), transform=transform_train)
val_dataset = ImageFolder(os.path.join(args.dataset_path, 'val/sorted'), transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# -------------------- training loop --------------------
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

    train_acc = correct / len(train_dataset)
    train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss, val_correct = 0.0, 0
    print(f"[Epoch {epoch+1:03d}] Validating...")
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Val", leave=False):
            x, y = x.cuda(), y.cuda()
            out = model(x)
            val_loss += F.cross_entropy(out, y).item()
            val_correct += (out.argmax(dim=1) == y).sum().item()

    val_acc = val_correct / len(val_dataset)
    val_loss /= len(val_loader)

    print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

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

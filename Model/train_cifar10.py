"""
# 此檔案作用 : 訓練 cifar-10的權重檔
# 此py檔案的output : 權重檔案"VGG16_cifar10_bs64.pth" -> 之後用於 export_model.py 轉換為 ONNX 檔案 (後續轉為tensorRT engine)

建立權重檔指令 / 後面是目前訓練到最好的acc

python train_cifar10.py VGG16  --batch_size 64 / acc:0.92

python train_cifar10.py VGG19  --batch_size 64 / acc:0.90

python train_cifar10.py ResNet50  --batch_size 64 / acc:0.90

python train_cifar10.py MobileNetV2  --batch_size 64 / acc:0.8618 -> 重新train
epoch 31 -> test acc: 0.8524
"""
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from models import MODELS
from tqdm import tqdm

# -------------------- argparse --------------------
parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--dataset_path', type=str, default='./Data_cifar10/cifar10')
args = parser.parse_args()

# -------------------- checkpoint path(檢查點路徑生成) --------------------
def get_default_checkpoint_path(args):
    filename = f"{args.model_name}_cifar10_bs{args.batch_size}.pth"
    return os.path.join("./checkpoints", filename)

# -------------------- Optimizer 選擇 --------------------
OPTIMIZERS = {
    'sgd': lambda params: torch.optim.SGD(params, lr=args.lr, momentum=args.momentum),
    'adam': lambda params: torch.optim.Adam(params, lr=args.lr)
}

# -------------------- 模型 --------------------
model = MODELS[args.model_name](num_classes=10, input_size=32).cuda()
optimizer = OPTIMIZERS[args.optimizer](model.parameters())

# 若 checkpoint_path 未指定，則自動生成
if args.checkpoint_path is None:
    os.makedirs("./checkpoints", exist_ok=True)
    args.checkpoint_path = get_default_checkpoint_path(args)
    print(f"💾 Saving to: {args.checkpoint_path}")

# 若檢查點存在，載入權重
# Resume from checkpoint if exists
if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
    print(f"📦 Loading checkpoint from {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path))
else:
    print("🚀 No checkpoint found or not specified. Training from scratch.")
    
# 數據預處理和增強
# -------------------- Transforms --------------------
# train data 隨機剪裁、隨機水平翻轉、正規化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# test data 僅正規化，無數據增強
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 數據載入
# -------------------- Dataset --------------------
train_dataset = torchvision.datasets.CIFAR10(
    root=args.dataset_path, train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root=args.dataset_path, train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)  # num_worker=4：使用 4 個子進程來加速數據載入
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# -------------------- 訓練迴圈 --------------------
# 含訓練、驗證、保存、早停
best_accuracy = 0.0
patience_counter = 0

for epoch in range(args.epochs):
    model.train()
    train_loss, train_correct = 0.0, 0

    print(f"\n[Epoch {epoch+1:03d}] Training...")
    for images, labels in tqdm(train_loader, desc="Train", leave=False):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)
    train_loss /= len(train_loader)

    model.eval()
    test_loss, test_correct = 0.0, 0

    print(f"[Epoch {epoch+1:03d}] Validating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test", leave=False):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()

    test_acc = test_correct / len(test_dataset)
    test_loss /= len(test_loader)

    print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    if test_acc > best_accuracy:
        print(f"✅ Saving checkpoint to {args.checkpoint_path}")
        torch.save(model.state_dict(), args.checkpoint_path)
        best_accuracy = test_acc
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"🕗 No improvement. Patience: {patience_counter}/{args.patience}")
        if patience_counter >= args.patience:
            print("⛔ Early stopping triggered.")
            break

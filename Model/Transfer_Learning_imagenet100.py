'''
python train_imagenet100.py VGG19  --batch_size 64 --input_size 256
python train_imagenet100.py VGG19  --batch_size 64 --input_size 331

python Transfer_Learning_imagenet100.py

'''


import torch

import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm

# --- 參數設定 (您可以根據需求修改) ---
# 1. 新模型的參數
TARGET_INPUT_SIZE = 331
NUM_CLASSES = 100
BATCH_SIZE = 32 # 根據您的 VRAM 調整
NUM_EPOCHS = 10   # 微調訓練的週期數
LEARNING_RATE = 0.001

# 2. 權重與資料集路徑
# ❗ 請確保以下兩個路徑是正確的
PRETRAINED_CHECKPOINT_PATH = 'checkpoints/VGG19_imagenet100_bs64_input256.pth'
DATASET_PATH = '/home/sslab_agx/Desktop/DeepGreen/Model/Data_imagenet100'


# 為了讓此腳本能獨立執行，我先從您之前的程式碼複製一份模型定義過來
# 在您的專案中，可以直接 from models import ModelVGG19
from torchvision.models import vgg19_bn
class ModelVGG19(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        self.model = vgg19_bn(pretrained=True)
        if input_size == 224:
            self.model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            flattened_size = 512 * 7 * 7
            classifier = nn.Sequential(nn.Linear(flattened_size, 4096),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(4096, 4096),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(4096, num_classes))
        elif input_size > 224:
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flattened_size = 512 * 1 * 1
            classifier = nn.Sequential(nn.Linear(flattened_size, 4096),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(4096, 4096),nn.ReLU(True),nn.Dropout(0.5),nn.Linear(4096, num_classes))
        else:
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flattened_size = 512
            classifier = nn.Sequential(nn.Linear(flattened_size, 256),nn.BatchNorm1d(256),nn.ReLU(True),nn.Dropout(0.4),nn.Linear(256, num_classes))
        self.model.classifier = classifier
    def forward(self, x):
        return self.model(x)
# --- 模型定義結束 ---

# 1. 建立一個目標為 input_size=256 的新模型實體
# 這會讓模型內部使用 (1, 1) 的 avgpool 和 512 維輸入的 classifier
model = ModelVGG19(num_classes=NUM_CLASSES, input_size=TARGET_INPUT_SIZE).cuda()

# 2. 載入您用 input_size=224 訓練好的權重檔案
checkpoint_pretrained = torch.load(PRETRAINED_CHECKPOINT_PATH)
model_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in checkpoint_pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(filtered_state_dict)
model.load_state_dict(model_dict, strict=False)
print("✅ 特徵提取層 (features) 的權重已成功載入！")
print("⚠️ 分類層 (classifier) 保持隨機初始化，需要重新訓練。")


# 凍結 features 層
for param in model.model.features.parameters():
    param.requires_grad = False
    

# --- ✅ DataLoader 載入部分 (從的 train_imagenet100.py 複製而來) ---
print("🚚 Preparing DataLoaders...")
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(TARGET_INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(TARGET_INPUT_SIZE),
    transforms.CenterCrop(TARGET_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(os.path.join(DATASET_PATH, 'train'), transform=transform_train)
val_dataset = ImageFolder(os.path.join(DATASET_PATH, 'val'), transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")

# --- 訓練準備 (優化器、損失函數) ---
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()


# 3. 執行訓練迴圈 (此處為示意程式碼)
num_epochs = 10 # 假設訓練 10 個 epoch
# dataloader = ... (您需要定義您的資料載入器)

# --- 完整訓練與驗證迴圈 ---
best_accuracy = 0.0
new_checkpoint_path = f"./VGG19_finetuned_input{TARGET_INPUT_SIZE}.pth"

for epoch in range(NUM_EPOCHS):
    model.train()
    print(f"\n[Epoch {epoch+1:03d}] Training...")
    for inputs, labels in tqdm(train_loader, desc="Train"):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # --- 驗證 ---
    model.eval()
    val_correct = 0
    print(f"[Epoch {epoch+1:03d}] Validating...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Val"):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    print(f"[Epoch {epoch+1:03d}] Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_accuracy:
        print(f"✅ Accuracy improved! Saving checkpoint to {new_checkpoint_path}")
        torch.save(model.state_dict(), new_checkpoint_path)
        best_accuracy = val_acc

print("\n🎉 Fine-tuning finished!")
print(f"Best validation accuracy: {best_accuracy:.4f}")
print(f"Final model saved to: {new_checkpoint_path}")

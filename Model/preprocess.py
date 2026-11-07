import os
import shutil

# 設定資料路徑（請依實際路徑修改）
val_dir = '/home/sslab_agx/Desktop/DeepGreen/Model/Data_tiny_Imagenet/tiny-imagenet-200/val'
img_dir = os.path.join(val_dir, 'images')
ann_file = os.path.join(val_dir, 'val_annotations.txt')
output_dir = os.path.join(val_dir, 'sorted')  # 整理後輸出這裡

# 建立 sorted 資料夾
os.makedirs(output_dir, exist_ok=True)

# 解析 val_annotations.txt
with open(ann_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        filename, wnid = parts[0], parts[1]

        # 為每個類別建立子資料夾
        class_dir = os.path.join(output_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(img_dir, filename)
        dst = os.path.join(class_dir, filename)

        if os.path.exists(src):
            shutil.copyfile(src, dst)

print(f"✅ 成功整理驗證資料至：{output_dir}")
print("請在你的程式中這樣讀入驗證資料集：")
print(f"val_dataset = ImageFolder('{output_dir}', transform=transform_test)")

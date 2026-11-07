#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
由 DLA engine 的 log 檔案中，萃取每個 DLA block 的特徵，輸出為 CSV 檔案
-> 這邊只是解析 log 檔案

feature_extractor_tina_v3.py
複製1107版本
擴充版：從 engine log 解析 block 特徵，加入記憶體與運算密度等延伸特徵。

12/5 現在只測試VGG16 v3版本(有preset ABCD)：

VGG16:
cifar10 :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_A/VGG16_cifar10_input32_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_A/VGG16_cifar10_input32_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_A/VGG16_cifar10_input32_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_A/VGG16_cifar10_input32_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_A/VGG16_cifar10_input32_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_A/VGG16_cifar10_input32_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_B/VGG16_cifar10_input32_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_B/VGG16_cifar10_input32_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_B/VGG16_cifar10_input32_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_B/VGG16_cifar10_input32_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_B/VGG16_cifar10_input32_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_B/VGG16_cifar10_input32_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_C/VGG16_cifar10_input32_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_C/VGG16_cifar10_input32_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_C/VGG16_cifar10_input32_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_C/VGG16_cifar10_input32_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_C/VGG16_cifar10_input32_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_C/VGG16_cifar10_input32_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_D/VGG16_cifar10_input32_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_D/VGG16_cifar10_input32_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_D/VGG16_cifar10_input32_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_D/VGG16_cifar10_input32_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_D/VGG16_cifar10_input32_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_D/VGG16_cifar10_input32_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_D/

------------------------
tiny :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_A/VGG16_tiny_input64_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_A/VGG16_tiny_input64_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_A/VGG16_tiny_input64_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_A/VGG16_tiny_input64_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_A/VGG16_tiny_input64_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_A/VGG16_tiny_input64_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_B/VGG16_tiny_input64_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_B/VGG16_tiny_input64_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_B/VGG16_tiny_input64_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_B/VGG16_tiny_input64_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_B/VGG16_tiny_input64_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_B/VGG16_tiny_input64_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_C/VGG16_tiny_input64_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_C/VGG16_tiny_input64_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_C/VGG16_tiny_input64_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_C/VGG16_tiny_input64_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_C/VGG16_tiny_input64_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_C/VGG16_tiny_input64_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_D/VGG16_tiny_input64_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_D/VGG16_tiny_input64_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_D/VGG16_tiny_input64_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_D/VGG16_tiny_input64_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_D/VGG16_tiny_input64_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_D/VGG16_tiny_input64_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_D/

---------------
imagenet100(224) :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_A/VGG16_imagenet100_input224_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_A/VGG16_imagenet100_input224_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_A/VGG16_imagenet100_input224_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_A/VGG16_imagenet100_input224_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_A/VGG16_imagenet100_input224_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_A/VGG16_imagenet100_input224_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_B/VGG16_imagenet100_input224_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_B/VGG16_imagenet100_input224_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_B/VGG16_imagenet100_input224_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_B/VGG16_imagenet100_input224_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_B/VGG16_imagenet100_input224_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_B/VGG16_imagenet100_input224_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_C/VGG16_imagenet100_input224_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_C/VGG16_imagenet100_input224_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_C/VGG16_imagenet100_input224_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_C/VGG16_imagenet100_input224_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_C/VGG16_imagenet100_input224_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_C/VGG16_imagenet100_input224_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_D/VGG16_imagenet100_input224_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_D/VGG16_imagenet100_input224_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_D/VGG16_imagenet100_input224_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_D/VGG16_imagenet100_input224_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_D/VGG16_imagenet100_input224_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_D/VGG16_imagenet100_input224_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_D/

-------------------
imagenet100(256) :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_A/VGG16_imagenet100_input256_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_A/VGG16_imagenet100_input256_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_A/VGG16_imagenet100_input256_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_A/VGG16_imagenet100_input256_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_A/VGG16_imagenet100_input256_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_A/VGG16_imagenet100_input256_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_B/VGG16_imagenet100_input256_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_B/VGG16_imagenet100_input256_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_B/VGG16_imagenet100_input256_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_B/VGG16_imagenet100_input256_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_B/VGG16_imagenet100_input256_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_B/VGG16_imagenet100_input256_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_C/VGG16_imagenet100_input256_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_C/VGG16_imagenet100_input256_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_C/VGG16_imagenet100_input256_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_C/VGG16_imagenet100_input256_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_C/VGG16_imagenet100_input256_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_C/VGG16_imagenet100_input256_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_D/VGG16_imagenet100_input256_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_D/VGG16_imagenet100_input256_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_D/VGG16_imagenet100_input256_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_D/VGG16_imagenet100_input256_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_D/VGG16_imagenet100_input256_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_D/VGG16_imagenet100_input256_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_D/

-------
imagenet100(331) :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_A/VGG16_imagenet100_input331_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_A/VGG16_imagenet100_input331_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_A/VGG16_imagenet100_input331_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_A/VGG16_imagenet100_input331_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_A/VGG16_imagenet100_input331_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_A/VGG16_imagenet100_input331_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_B/VGG16_imagenet100_input331_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_B/VGG16_imagenet100_input331_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_B/VGG16_imagenet100_input331_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_B/VGG16_imagenet100_input331_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_B/VGG16_imagenet100_input331_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_B/VGG16_imagenet100_input331_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_C/VGG16_imagenet100_input331_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_C/VGG16_imagenet100_input331_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_C/VGG16_imagenet100_input331_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_C/VGG16_imagenet100_input331_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_C/VGG16_imagenet100_input331_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_C/VGG16_imagenet100_input331_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs64/preset_D/VGG16_imagenet100_input331_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs32/preset_D/VGG16_imagenet100_input331_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs16/preset_D/VGG16_imagenet100_input331_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs8/preset_D/VGG16_imagenet100_input331_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs4/preset_D/VGG16_imagenet100_input331_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG16/bs1/preset_D/VGG16_imagenet100_input331_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG16/bs1/preset_D/

----------
12/27 新增測ResNet50:
cifar10 :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_A/ResNet50_cifar10_input32_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_A/ResNet50_cifar10_input32_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_A/ResNet50_cifar10_input32_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_A/ResNet50_cifar10_input32_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_A/ResNet50_cifar10_input32_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_A/ResNet50_cifar10_input32_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_B/ResNet50_cifar10_input32_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_B/ResNet50_cifar10_input32_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_B/ResNet50_cifar10_input32_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_B/ResNet50_cifar10_input32_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_B/ResNet50_cifar10_input32_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_B/ResNet50_cifar10_input32_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_C/ResNet50_cifar10_input32_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_C/ResNet50_cifar10_input32_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_C/ResNet50_cifar10_input32_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_C/ResNet50_cifar10_input32_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_C/ResNet50_cifar10_input32_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_C/ResNet50_cifar10_input32_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_D/ResNet50_cifar10_input32_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_D/ResNet50_cifar10_input32_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_D/ResNet50_cifar10_input32_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_D/ResNet50_cifar10_input32_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_D/ResNet50_cifar10_input32_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_D/ResNet50_cifar10_input32_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_D/

tiny :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_A/ResNet50_tiny_input64_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_A/ResNet50_tiny_input64_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_A/ResNet50_tiny_input64_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_A/ResNet50_tiny_input64_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_A/ResNet50_tiny_input64_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_A/ResNet50_tiny_input64_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_B/ResNet50_tiny_input64_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_B/ResNet50_tiny_input64_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_B/ResNet50_tiny_input64_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_B/ResNet50_tiny_input64_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_B/ResNet50_tiny_input64_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_B/ResNet50_tiny_input64_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_C/ResNet50_tiny_input64_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_C/ResNet50_tiny_input64_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_C/ResNet50_tiny_input64_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_C/ResNet50_tiny_input64_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_C/ResNet50_tiny_input64_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_C/ResNet50_tiny_input64_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_D/ResNet50_tiny_input64_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_D/ResNet50_tiny_input64_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_D/ResNet50_tiny_input64_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_D/ResNet50_tiny_input64_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_D/ResNet50_tiny_input64_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_D/ResNet50_tiny_input64_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_D/

imagenet100(224) :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_A/ResNet50_imagenet100_input224_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_A/ResNet50_imagenet100_input224_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_A/ResNet50_imagenet100_input224_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_A/ResNet50_imagenet100_input224_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_A/ResNet50_imagenet100_input224_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_A/ResNet50_imagenet100_input224_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_B/ResNet50_imagenet100_input224_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_B/ResNet50_imagenet100_input224_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_B/ResNet50_imagenet100_input224_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_B/ResNet50_imagenet100_input224_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_B/ResNet50_imagenet100_input224_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_B/ResNet50_imagenet100_input224_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_C/ResNet50_imagenet100_input224_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_C/ResNet50_imagenet100_input224_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_C/ResNet50_imagenet100_input224_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_C/ResNet50_imagenet100_input224_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_C/ResNet50_imagenet100_input224_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_C/ResNet50_imagenet100_input224_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_D/ResNet50_imagenet100_input224_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_D/ResNet50_imagenet100_input224_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_D/ResNet50_imagenet100_input224_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_D/ResNet50_imagenet100_input224_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_D/ResNet50_imagenet100_input224_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_D/ResNet50_imagenet100_input224_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_D/

imagenet100(256) :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_A/ResNet50_imagenet100_input256_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_A/ResNet50_imagenet100_input256_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_A/ResNet50_imagenet100_input256_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_A/ResNet50_imagenet100_input256_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_A/ResNet50_imagenet100_input256_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_A/ResNet50_imagenet100_input256_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_B/ResNet50_imagenet100_input256_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_B/ResNet50_imagenet100_input256_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_B/ResNet50_imagenet100_input256_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_B/ResNet50_imagenet100_input256_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_B/ResNet50_imagenet100_input256_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_B/ResNet50_imagenet100_input256_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_C/ResNet50_imagenet100_input256_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_C/ResNet50_imagenet100_input256_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_C/ResNet50_imagenet100_input256_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_C/ResNet50_imagenet100_input256_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_C/ResNet50_imagenet100_input256_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_C/ResNet50_imagenet100_input256_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_D/ResNet50_imagenet100_input256_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_D/ResNet50_imagenet100_input256_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_D/ResNet50_imagenet100_input256_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_D/ResNet50_imagenet100_input256_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_D/ResNet50_imagenet100_input256_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_D/ResNet50_imagenet100_input256_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_D/

imagenet100(331) :
preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_A/ResNet50_imagenet100_input331_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_A/ResNet50_imagenet100_input331_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_A/ResNet50_imagenet100_input331_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_A/ResNet50_imagenet100_input331_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_A/ResNet50_imagenet100_input331_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_A/ResNet50_imagenet100_input331_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_A/

preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_B/ResNet50_imagenet100_input331_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_B/ResNet50_imagenet100_input331_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_B/ResNet50_imagenet100_input331_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_B/ResNet50_imagenet100_input331_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_B/ResNet50_imagenet100_input331_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_B/ResNet50_imagenet100_input331_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_B/

preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_C/ResNet50_imagenet100_input331_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_C/ResNet50_imagenet100_input331_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_C/ResNet50_imagenet100_input331_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_C/ResNet50_imagenet100_input331_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_C/ResNet50_imagenet100_input331_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_C/ResNet50_imagenet100_input331_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_C/

preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs64/preset_D/ResNet50_imagenet100_input331_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs32/preset_D/ResNet50_imagenet100_input331_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs16/preset_D/ResNet50_imagenet100_input331_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs8/preset_D/ResNet50_imagenet100_input331_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs4/preset_D/ResNet50_imagenet100_input331_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/ResNet50/bs1/preset_D/ResNet50_imagenet100_input331_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/ResNet50/bs1/preset_D/

-------------VGG19-----------------
# preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_A/VGG19_cifar10_input32_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_A/VGG19_cifar10_input32_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_A/VGG19_cifar10_input32_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_A/VGG19_cifar10_input32_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_A/VGG19_cifar10_input32_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_A/VGG19_cifar10_input32_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_A/

# preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_B/VGG19_cifar10_input32_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_B/VGG19_cifar10_input32_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_B/VGG19_cifar10_input32_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_B/VGG19_cifar10_input32_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_B/VGG19_cifar10_input32_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_B/VGG19_cifar10_input32_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_B/

# preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_C/VGG19_cifar10_input32_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_C/VGG19_cifar10_input32_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_C/VGG19_cifar10_input32_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_C/VGG19_cifar10_input32_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_C/VGG19_cifar10_input32_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_C/VGG19_cifar10_input32_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_C/

# preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_D/VGG19_cifar10_input32_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_D/VGG19_cifar10_input32_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_D/VGG19_cifar10_input32_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_D/VGG19_cifar10_input32_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_D/VGG19_cifar10_input32_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_D/VGG19_cifar10_input32_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_D/

# VGG19: tiny
# preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_A/VGG19_tiny_input64_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_A/VGG19_tiny_input64_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_A/VGG19_tiny_input64_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_A/VGG19_tiny_input64_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_A/VGG19_tiny_input64_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_A/VGG19_tiny_input64_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_A/

# preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_B/VGG19_tiny_input64_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_B/VGG19_tiny_input64_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_B/VGG19_tiny_input64_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_B/VGG19_tiny_input64_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_B/VGG19_tiny_input64_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_B/VGG19_tiny_input64_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_B/

# preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_C/VGG19_tiny_input64_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_C/VGG19_tiny_input64_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_C/VGG19_tiny_input64_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_C/VGG19_tiny_input64_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_C/VGG19_tiny_input64_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_C/VGG19_tiny_input64_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_C/

# preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_D/VGG19_tiny_input64_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_D/VGG19_tiny_input64_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_D/VGG19_tiny_input64_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_D/VGG19_tiny_input64_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_D/VGG19_tiny_input64_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_D/VGG19_tiny_input64_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_D/

# VGG19: imagenet100(224)
# preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_A/VGG19_imagenet100_input224_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_A/VGG19_imagenet100_input224_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_A/VGG19_imagenet100_input224_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_A/VGG19_imagenet100_input224_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_A/VGG19_imagenet100_input224_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_A/VGG19_imagenet100_input224_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_A/

# preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_B/VGG19_imagenet100_input224_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_B/VGG19_imagenet100_input224_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_B/VGG19_imagenet100_input224_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_B/VGG19_imagenet100_input224_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_B/VGG19_imagenet100_input224_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_B/VGG19_imagenet100_input224_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_B/

# preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_C/VGG19_imagenet100_input224_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_C/VGG19_imagenet100_input224_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_C/VGG19_imagenet100_input224_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_C/VGG19_imagenet100_input224_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_C/VGG19_imagenet100_input224_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_C/VGG19_imagenet100_input224_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_C/

# preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_D/VGG19_imagenet100_input224_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_D/VGG19_imagenet100_input224_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_D/VGG19_imagenet100_input224_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_D/VGG19_imagenet100_input224_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_D/VGG19_imagenet100_input224_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_D/VGG19_imagenet100_input224_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_D/

# VGG19: imagenet100(256)
# preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_A/VGG19_imagenet100_input256_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_A/VGG19_imagenet100_input256_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_A/VGG19_imagenet100_input256_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_A/VGG19_imagenet100_input256_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_A/VGG19_imagenet100_input256_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_A/VGG19_imagenet100_input256_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_A/

# preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_B/VGG19_imagenet100_input256_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_B/VGG19_imagenet100_input256_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_B/VGG19_imagenet100_input256_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_B/VGG19_imagenet100_input256_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_B/VGG19_imagenet100_input256_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_B/VGG19_imagenet100_input256_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_B/

# preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_C/VGG19_imagenet100_input256_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_C/VGG19_imagenet100_input256_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_C/VGG19_imagenet100_input256_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_C/VGG19_imagenet100_input256_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_C/VGG19_imagenet100_input256_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_C/VGG19_imagenet100_input256_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_C/

# preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_D/VGG19_imagenet100_input256_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_D/VGG19_imagenet100_input256_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_D/VGG19_imagenet100_input256_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_D/VGG19_imagenet100_input256_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_D/VGG19_imagenet100_input256_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_D/VGG19_imagenet100_input256_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_D/

# VGG19: imagenet100(331)
# preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_A/VGG19_imagenet100_input331_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_A/VGG19_imagenet100_input331_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_A/VGG19_imagenet100_input331_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_A/VGG19_imagenet100_input331_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_A/VGG19_imagenet100_input331_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_A/VGG19_imagenet100_input331_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_A/

# preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_B/VGG19_imagenet100_input331_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_B/VGG19_imagenet100_input331_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_B/VGG19_imagenet100_input331_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_B/VGG19_imagenet100_input331_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_B/VGG19_imagenet100_input331_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_B/VGG19_imagenet100_input331_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_B/

# preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_C/VGG19_imagenet100_input331_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_C/VGG19_imagenet100_input331_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_C/VGG19_imagenet100_input331_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_C/VGG19_imagenet100_input331_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_C/VGG19_imagenet100_input331_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_C/VGG19_imagenet100_input331_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_C/

# preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs64/preset_D/VGG19_imagenet100_input331_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs32/preset_D/VGG19_imagenet100_input331_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs16/preset_D/VGG19_imagenet100_input331_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs8/preset_D/VGG19_imagenet100_input331_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs4/preset_D/VGG19_imagenet100_input331_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/VGG19/bs1/preset_D/VGG19_imagenet100_input331_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/VGG19/bs1/preset_D/

----------------MobilerNetV2---------------
# MobileNetV2: cifar10
# preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_A/MobileNetV2_cifar10_input32_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_A/MobileNetV2_cifar10_input32_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_A/MobileNetV2_cifar10_input32_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_A/MobileNetV2_cifar10_input32_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_A/MobileNetV2_cifar10_input32_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_A/MobileNetV2_cifar10_input32_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_A/

# preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_B/MobileNetV2_cifar10_input32_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_B/MobileNetV2_cifar10_input32_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_B/MobileNetV2_cifar10_input32_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_B/MobileNetV2_cifar10_input32_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_B/MobileNetV2_cifar10_input32_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_B/MobileNetV2_cifar10_input32_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_B/

# preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_C/MobileNetV2_cifar10_input32_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_C/MobileNetV2_cifar10_input32_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_C/MobileNetV2_cifar10_input32_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_C/MobileNetV2_cifar10_input32_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_C/MobileNetV2_cifar10_input32_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_C/MobileNetV2_cifar10_input32_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_C/

# preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_D/MobileNetV2_cifar10_input32_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_D/MobileNetV2_cifar10_input32_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_D/MobileNetV2_cifar10_input32_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_D/MobileNetV2_cifar10_input32_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_D/MobileNetV2_cifar10_input32_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_D/MobileNetV2_cifar10_input32_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_D/

# MobileNetV2: tiny
# preset A：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_A/MobileNetV2_tiny_input64_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_A/MobileNetV2_tiny_input64_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_A/MobileNetV2_tiny_input64_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_A/MobileNetV2_tiny_input64_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_A/MobileNetV2_tiny_input64_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_A/MobileNetV2_tiny_input64_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_A/

# preset B：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_B/MobileNetV2_tiny_input64_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_B/MobileNetV2_tiny_input64_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_B/MobileNetV2_tiny_input64_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_B/MobileNetV2_tiny_input64_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_B/MobileNetV2_tiny_input64_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_B/MobileNetV2_tiny_input64_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_B/

# preset C：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_C/MobileNetV2_tiny_input64_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_C/MobileNetV2_tiny_input64_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_C/MobileNetV2_tiny_input64_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_C/MobileNetV2_tiny_input64_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_C/MobileNetV2_tiny_input64_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_C/MobileNetV2_tiny_input64_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_C/

# preset D：
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_D/MobileNetV2_tiny_input64_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_D/MobileNetV2_tiny_input64_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_D/MobileNetV2_tiny_input64_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_D/MobileNetV2_tiny_input64_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_D/MobileNetV2_tiny_input64_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_D/MobileNetV2_tiny_input64_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_D/

# Preset A
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_A/MobileNetV2_imagenet100_input224_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_A/MobileNetV2_imagenet100_input224_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_A/MobileNetV2_imagenet100_input224_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_A/MobileNetV2_imagenet100_input224_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_A/MobileNetV2_imagenet100_input224_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_A/MobileNetV2_imagenet100_input224_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_A/

# Preset B
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_B/MobileNetV2_imagenet100_input224_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_B/MobileNetV2_imagenet100_input224_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_B/MobileNetV2_imagenet100_input224_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_B/MobileNetV2_imagenet100_input224_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_B/MobileNetV2_imagenet100_input224_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_B/MobileNetV2_imagenet100_input224_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_B/

# Preset C
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_C/MobileNetV2_imagenet100_input224_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_C/MobileNetV2_imagenet100_input224_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_C/MobileNetV2_imagenet100_input224_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_C/MobileNetV2_imagenet100_input224_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_C/MobileNetV2_imagenet100_input224_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_C/MobileNetV2_imagenet100_input224_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_C/

# Preset D
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_D/MobileNetV2_imagenet100_input224_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_D/MobileNetV2_imagenet100_input224_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_D/MobileNetV2_imagenet100_input224_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_D/MobileNetV2_imagenet100_input224_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_D/MobileNetV2_imagenet100_input224_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_D/MobileNetV2_imagenet100_input224_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_D/

# Preset A
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_A/MobileNetV2_imagenet100_input256_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_A/MobileNetV2_imagenet100_input256_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_A/MobileNetV2_imagenet100_input256_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_A/MobileNetV2_imagenet100_input256_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_A/MobileNetV2_imagenet100_input256_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_A/MobileNetV2_imagenet100_input256_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_A/

# Preset B
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_B/MobileNetV2_imagenet100_input256_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_B/MobileNetV2_imagenet100_input256_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_B/MobileNetV2_imagenet100_input256_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_B/MobileNetV2_imagenet100_input256_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_B/MobileNetV2_imagenet100_input256_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_B/MobileNetV2_imagenet100_input256_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_B/

# Preset C
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_C/MobileNetV2_imagenet100_input256_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_C/MobileNetV2_imagenet100_input256_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_C/MobileNetV2_imagenet100_input256_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_C/MobileNetV2_imagenet100_input256_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_C/MobileNetV2_imagenet100_input256_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_C/MobileNetV2_imagenet100_input256_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_C/

# Preset D
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_D/MobileNetV2_imagenet100_input256_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_D/MobileNetV2_imagenet100_input256_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_D/MobileNetV2_imagenet100_input256_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_D/MobileNetV2_imagenet100_input256_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_D/MobileNetV2_imagenet100_input256_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_D/MobileNetV2_imagenet100_input256_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_D/

# Preset A
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_A/MobileNetV2_imagenet100_input331_with_blockopt_bs64_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_A/MobileNetV2_imagenet100_input331_with_blockopt_bs32_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_A/MobileNetV2_imagenet100_input331_with_blockopt_bs16_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_A/MobileNetV2_imagenet100_input331_with_blockopt_bs8_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_A/MobileNetV2_imagenet100_input331_with_blockopt_bs4_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_A/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_A/MobileNetV2_imagenet100_input331_with_blockopt_bs1_preset_A.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_A/

# Preset B
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_B/MobileNetV2_imagenet100_input331_with_blockopt_bs64_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_B/MobileNetV2_imagenet100_input331_with_blockopt_bs32_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_B/MobileNetV2_imagenet100_input331_with_blockopt_bs16_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_B/MobileNetV2_imagenet100_input331_with_blockopt_bs8_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_B/MobileNetV2_imagenet100_input331_with_blockopt_bs4_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_B/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_B/MobileNetV2_imagenet100_input331_with_blockopt_bs1_preset_B.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_B/

# Preset C
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_C/MobileNetV2_imagenet100_input331_with_blockopt_bs64_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_C/MobileNetV2_imagenet100_input331_with_blockopt_bs32_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_C/MobileNetV2_imagenet100_input331_with_blockopt_bs16_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_C/MobileNetV2_imagenet100_input331_with_blockopt_bs8_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_C/MobileNetV2_imagenet100_input331_with_blockopt_bs4_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_C/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_C/MobileNetV2_imagenet100_input331_with_blockopt_bs1_preset_C.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_C/

# Preset D
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs64/preset_D/MobileNetV2_imagenet100_input331_with_blockopt_bs64_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs64/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs32/preset_D/MobileNetV2_imagenet100_input331_with_blockopt_bs32_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs32/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs16/preset_D/MobileNetV2_imagenet100_input331_with_blockopt_bs16_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs16/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs8/preset_D/MobileNetV2_imagenet100_input331_with_blockopt_bs8_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs8/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs4/preset_D/MobileNetV2_imagenet100_input331_with_blockopt_bs4_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs4/preset_D/
python feature_extractor_tina_v3.py --log_path=./engine/with_blockopt/log/MobileNetV2/bs1/preset_D/MobileNetV2_imagenet100_input331_with_blockopt_bs1_preset_D.log --output_dir=./feature_v3/with_blockopt/MobileNetV2/bs1/preset_D/


"""
import onnx
import re
import csv
import argparse
from pathlib import Path
from statistics import mean
import os
import numpy as np  # ← 新增，用於 log() 和記憶體計算

# ===== 新增：判斷「資料搬移層」的輔助函式 =====
DATA_MOVE_LAYER_TYPES = {
    'SHUFFLE',
    'RESHAPE',
    'SLICE',
    'CONCATENATION',
    'GATHER',
    'SCATTER',
    'RESIZE',
    'TRANSPOSE',
    'REFORMAT',
    'COPY',
}


def parse_shape(s):
    try:
        return list(map(int, s.split(',')))
    except:
        return []

def compute_resolution(shape):
    return shape[2] * shape[3] if len(shape) >= 4 else 0

# ===== 新增：判斷「資料搬移層」的輔助函式 =====
DATA_MOVE_LAYER_TYPES = {
    'SHUFFLE',
    'RESHAPE',
    'SLICE',
    'CONCATENATION',
    'GATHER',
    'SCATTER',
    'RESIZE',
    'TRANSPOSE',
    'REFORMAT',
    'COPY',
}

def is_data_movement_layer(layer_type: str, raw_line: str) -> bool:
    """
    粗略判斷該 layer 是否主要做資料搬移／重排。
    依據 layer type + 關鍵字來決定。
    """
    lt = (layer_type or '').upper()
    if lt in DATA_MOVE_LAYER_TYPES:
        return True

    low = (raw_line or '').lower()
    # 常見關鍵字：copy / reformat / host<->device
    if 'copy' in low or 'reformat' in low:
        return True
    if 'device to host' in low or 'host to device' in low:
        return True

    return False

def compute_memory_MB(shape, dtype_bytes=1):
    """計算張量記憶體使用量 (MB)"""
    # 假設所有數據類型 (Int8, FP16/32) 平均為 4 bytes 以確保一致性，
    # 實際上 Int8 應該是 1 byte，但保留原代碼的通用邏輯，若需要精確 Int8，需修改 dtype_bytes=1
    if not shape:
        return 0.0
    return np.prod(shape) * dtype_bytes / 1e6

def estimate_macs_flops(layer_type, input_shape, output_shape, weight_shape=None, kernel_shape=None, group=None, op_kind=None):
    """
    估算單個 layer 的 MACs 與 FLOPs 數量（每個 MAC = 2 FLOPs）。
    """
    layer_type_u = (layer_type or '').upper()
    try:
        # 將 [N, C] 自動擴展為 [N, C, 1, 1]
        if len(input_shape) == 2:
            input_shape = [input_shape[0], input_shape[1], 1, 1]
        if len(output_shape) == 2:
            output_shape = [output_shape[0], output_shape[1], 1, 1]
        O_N, O_C, O_H, O_W = map(int, output_shape)
        I_N, I_C, I_H, I_W = map(int, input_shape)
    except Exception as e:
        # print(f"[WARN] Invalid input/output shape: {e}")
        return 0, 0

    output_size = O_N * O_C * O_H * O_W

    if layer_type == 'CONVOLUTION':
        if not weight_shape:
            return 0, 0
        
        # 處理權重維度
        if len(weight_shape) == 2:
            weight_shape = weight_shape + [1, 1]
        elif len(weight_shape) == 3:
            weight_shape = weight_shape + [1]
        if not kernel_shape:
            kernel_shape = [1, 1]

        K_H, K_W = map(int, kernel_shape)
        W_N, W_C, _, _ = map(int, weight_shape)

        # 判斷卷積類型
        is_depthwise = (group == I_C and W_N == I_C and W_C == 1) if group is not None else (W_C == 1 and W_N == I_C)
        is_gemm_like = (K_H == 1 and K_W == 1 and I_H == 1 and I_W == 1 and O_H == 1 and O_W == 1 and group == 1)

        if is_depthwise:
            macs = I_C * K_H * K_W * O_H * O_W * O_N
        elif is_gemm_like:
            macs = I_C * O_C * O_N
        else:
            macs = I_N * O_C * O_H * O_W * I_C * K_H * K_W
        
        return macs, macs * 2

    elif layer_type == 'POOLING':
        if not kernel_shape:
            return 0, 0
        K_H, K_W = map(int, kernel_shape)
        flops = output_size * K_H * K_W
        return 0, flops

    elif layer_type_u == 'ACTIVATION':
        k = (op_kind or '').lower()
        if k == 'sigmoid':
            return 0, 5 * output_size
        elif k == 'tanh':
            return 0, 6 * output_size
        return 0, output_size

    elif layer_type == 'ELEMENTWISE':
        return 0, output_size

    else:
        return 0, 0


# 從 log 檔案提取 DLA block 特徵
def extract_features(log_path):
    
    # 讀檔，將每行存成 list
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 找到所有 DLA block 的起始行
    dla_info_re = re.compile(r'(\{ForeignNode\[[^}]+\]\}): DLA Network Layer Information:')
    block_to_layers = {}
    current_block = None
    
    for line in lines:
        if (m := dla_info_re.search(line)):  
            current_block = m.group(1) 
            block_to_layers[current_block] = []
            continue
        
        if current_block:
            if line.startswith('Layer('): 
                layer_line = line.strip() 
                if ':' in layer_line:
                    layer_type = layer_line.split('Layer(')[1].split(')')[0] 
                    name_match = re.search(r'\): ([^,]+),', layer_line)
                    layer_name = name_match.group(1).strip() if name_match else 'Unknown'
                    block_to_layers[current_block].append((layer_type, layer_name, layer_line)) 
            else:
                current_block = None
    
    layer_details = {}
    parsing_node_re = re.compile(r'Parsing node: (.+?) \[(\w+)\]')
    output_mark_re = re.compile(r'Marking .* as output:')

    parsing_indices = [i for i, line in enumerate(lines) if parsing_node_re.search(line)] 
    
    for idx, start_idx in enumerate(parsing_indices):
        if idx + 1 < len(parsing_indices):
            end_idx = parsing_indices[idx + 1]
        else:
            end_idx = len(lines)
            for j in range(start_idx + 1, len(lines)):
                if output_mark_re.search(lines[j]):
                    end_idx = j + 1
                    break
                
        m = parsing_node_re.search(lines[start_idx])
        if m:
            lname, ltype = m.group(1), m.group(2)
            detail_block = lines[start_idx:end_idx]
            layer_details[lname] = ''.join(detail_block)

    
    features = []
    output_feature_keys = None  # 延後在第一個 block 決定
    
    for idx, (block, layers) in enumerate(block_to_layers.items()):
        
        # 重新初始化所有特徵欄位，包括平均參數
        f = {
            'block_index': idx,
            'block_name': block,
            'num_layers': sum(1 for l in layers if l[0] in ['CONVOLUTION', 'POOLING', 'ACTIVATION', 'ELEMENTWISE']),
            'layer_types': set(),
            'has_conv': 0,
            'has_pooling': 0,
            'has_activation': 0,
            'activation_types': set(),
            'input_tensor_shape': '',
            'output_tensor_shape': '',
            'input_resolution': 0,
            'output_resolution': 0,
            'input_channels': 0,
            'output_channels': 0,
            'batch_size': 0,
            'macs': 0, 
            'flops': 0,
            # ===== 新增：fallback / 資料搬移特徵（block level） =====
            'num_layers_total': 0,      # 該 DLA block 中 log 顯示的 layer 總數
            'num_layers_dla': 0,        # 判定在 DLA 執行的層數（預設視為 DLA）
            'num_layers_gpu': 0,        # 判定 fallback 至 GPU 的層數
            'fallback_ratio': 0.0,      # num_layers_gpu / num_layers_total
            'data_move_layers': 0,      # 資料搬移相關層的數量
            'data_move_ratio': 0.0,     # data_move_layers / num_layers_total
            
        }

        # 暫存變數
        conv_kernels, conv_strides, conv_pads, conv_outputs = [], [], [], [] # <-- 重新收集
        pool_kernels, pool_strides, pool_pads = [], [], []                   # <-- 重新收集
        
        for ltype, lname, raw in layers:
            f['layer_types'].add(ltype)
            if ltype == 'CONVOLUTION': f['has_conv'] = 1
            elif ltype == 'POOLING': f['has_pooling'] = 1
            elif ltype == 'ACTIVATION': f['has_activation'] = 1

            # ===== 新增：統計 fallback／資料搬移 =====
            f['num_layers_total'] += 1      # block 中的 layer 總數

            # 資料搬移層 -> 判斷資料搬移層
            if is_data_movement_layer(ltype, raw):
                f['data_move_layers'] += 1

            # 粗略判斷 GPU fallback：log 行內同時出現 "gpu" + "fallback"（不分大小寫）
            raw_lower = raw.lower()
            if ('gpu' in raw_lower) and ('fallback' in raw_lower):
                f['num_layers_gpu'] += 1
            else:
                # 在 DLA block 中，且沒有 fallback 標記 → 當作 DLA 執行
                f['num_layers_dla'] += 1
            
            # ====== 下面維持你原本的形狀與 MACs/FLOPs 解析邏輯 ======
            # 獲取 shape
            shape_match = re.findall(r'\[Int8\(\[(.*?)\]\)\]', raw)
            if shape_match:
                in_shape = shape_match[0].split(',') if len(shape_match) > 0 else []
                out_shape = shape_match[-1].split(',') if len(shape_match) > 1 else []
                
                # 紀錄 Block 級別的輸入輸出形狀
                if not f['input_tensor_shape'] and len(in_shape) >= 4:
                    f['input_tensor_shape'] = ','.join(in_shape)
                    f['batch_size'] = int(in_shape[0])
                    f['input_channels'] = int(in_shape[1])
                    f['input_resolution'] = int(in_shape[2]) * int(in_shape[3])
                if len(out_shape) >= 4:
                    f['output_tensor_shape'] = ','.join(out_shape) 
                    f['output_channels'] = int(out_shape[1])
                    f['output_resolution'] = int(out_shape[2]) * int(out_shape[3])

            # 獲取 Layer 詳細資訊和 MACs/FLOPs (以及平均參數所需的 k/s/p)
            detail_text = layer_details.get(lname.split(' ')[0], '')
            shape_match_detail = re.findall(r'\(([-\d,\s]+)\)', detail_text)
            in_shape_detail = list(map(int, shape_match_detail[0].split(','))) if len(shape_match_detail) > 0 else []
            if len(in_shape_detail) >= 1 and in_shape: in_shape_detail[0] = int(in_shape[0])
            out_shape_detail = list(map(int, shape_match_detail[-1].split(','))) if len(shape_match_detail) > 1 else []
            if len(out_shape_detail) >= 1 and out_shape: out_shape_detail[0] = int(out_shape[0])
            
            macs, flops = 0, 0
            group = 1
            kernel_shape_detail = [1, 1]
            weight_shape_detail = []
            op_kind = None
            
            if ltype == 'CONVOLUTION' and len(in_shape_detail) >= 2 and len(out_shape_detail) >= 2:
                # 嘗試從 ONNX 獲取 group/kernel/outputs/weights
                try:
                    # 嘗試從 ONNX 檔案讀取組、權重、內核形狀、步幅和填充
                    parts = str(log_path).split(os.sep)
                    model_name = parts[-3]
                    batch_folder = parts[-2]
                    log_filename = parts[-1]
                    onnx_filename = log_filename.replace('_with_blockopt', '').replace('.log', '.onnx')
                    onnx_path = os.path.join('.', 'onnx_optimized', model_name, batch_folder, onnx_filename)
                    model = onnx.load(onnx_path)
                    
                    for node in model.graph.node:
                        if node.name == lname:
                            for attr in node.attribute:
                                if attr.name == 'group':
                                    group = onnx.helper.get_attribute_value(attr)
                            if len(node.input) > 1:
                                w_name = node.input[1]
                                for initializer in model.graph.initializer:
                                    if initializer.name == w_name:
                                        weight_shape_detail = list(initializer.dims)
                                        break
                                
                    if 'Using kernel:' in detail_text:
                         k = re.search(r'kernel: \((\d+), (\d+)\)', detail_text)
                         s = re.search(r'strides: \((\d+), (\d+)\)', detail_text) # <-- 重新加入
                         pre_p = re.search(r'prepadding: \((\d+), (\d+)\)', detail_text) # <-- 重新加入
                         post_p = re.search(r'postpadding: \((\d+), (\d+)\)', detail_text) # <-- 重新加入
                         o = re.search(r'numOutputs: (\d+)', detail_text)
                         
                         if k:
                             kernel_shape_detail = [int(k.group(1)), int(k.group(2))]
                             conv_kernels.append(int(k.group(1)) * int(k.group(2))) 
                         if s:
                            conv_strides.append(int(s.group(1)) + int(s.group(2))) # <-- 收集步幅
                         if pre_p and post_p:
                             conv_pads.append(int(pre_p.group(1)) + int(pre_p.group(2)) + int(post_p.group(1)) + int(post_p.group(2))) # <-- 收集填充
                         if o:
                             conv_outputs.append(int(o.group(1))) 

                    macs, flops = estimate_macs_flops(ltype, in_shape_detail, out_shape_detail , weight_shape_detail , kernel_shape_detail, group)
                except:
                    # ONNX 讀取失敗時的處理與原始邏輯相同
                    if 'Using kernel:' in detail_text:
                         k = re.search(r'kernel: \((\d+), (\d+)\)', detail_text)
                         if k:
                             kernel_shape_detail = [int(k.group(1)), int(k.group(2))]
                         
                    if len(out_shape_detail) > 1 and len(in_shape_detail) > 1 and len(kernel_shape_detail) == 2:
                        weight_shape_detail = [out_shape_detail[1], in_shape_detail[1] // group, kernel_shape_detail[0], kernel_shape_detail[1]]
                    
                    macs, flops = estimate_macs_flops(ltype, in_shape_detail, out_shape_detail , weight_shape_detail , kernel_shape_detail, group)
                    
            elif ltype == 'POOLING' and len(in_shape_detail) >= 4 and len(out_shape_detail) >= 4:
                # 嘗試從 ONNX 讀取 pooling 參數
                try:
                    parts = str(log_path).split(os.sep)
                    model_name = parts[-3]
                    batch_folder = parts[-2]
                    log_filename = parts[-1]
                    onnx_filename = log_filename.replace('_with_blockopt', '').replace('.log', '.onnx')
                    onnx_path = os.path.join('.', 'onnx_optimized', model_name, batch_folder, onnx_filename)
                    model = onnx.load(onnx_path)
                    
                    for node in model.graph.node:
                        if node.name == lname:
                            for attr in node.attribute:
                                if attr.name == 'kernel_shape':
                                    kernel_shape_detail = onnx.helper.get_attribute_value(attr)
                                elif attr.name == 'strides':
                                    strides_detail = onnx.helper.get_attribute_value(attr)
                                elif attr.name == 'pads':
                                    pads_detail = onnx.helper.get_attribute_value(attr)
                                    
                    pool_kernels.append(kernel_shape_detail[0] * kernel_shape_detail[1]) # <-- 收集 pool kernel
                    pool_pads.append(pads_detail[0] + pads_detail[1] + pads_detail[2] + pads_detail[3]) # <-- 收集 pool pads
                    pool_strides.append(strides_detail[0] + strides_detail[1]) # <-- 收集 pool strides
                except:
                    # 如果讀取失敗，使用預設值或忽略
                    pass 
                
                macs, flops = estimate_macs_flops(ltype, in_shape_detail, out_shape_detail, weight_shape = None, kernel_shape = [1, 1] , group = None); 
            
            elif ltype == 'ELEMENTWISE' and len(in_shape_detail) >= 4 and len(out_shape_detail) >= 4:
                macs, flops = estimate_macs_flops(ltype, in_shape_detail, out_shape_detail, weight_shape = None, kernel_shape = None , group = None);

            elif ltype == 'ACTIVATION' and len(in_shape_detail) >= 4 and len(out_shape_detail) >= 4:
                # 確定 op_kind
                name_l = (lname + ' ' + layer_details.get(lname, '')).lower()
                if 'sigmoid' in name_l: op_kind = 'sigmoid'
                elif 'tanh' in name_l: op_kind = 'tanh'
                elif 'clip' in name_l: op_kind = 'clip'
                elif 'leakyrelu' in name_l or 'leaky' in name_l: op_kind = 'leakyrelu'
                else: op_kind = 'relu' 
                
                macs, flops = estimate_macs_flops(ltype, in_shape_detail, out_shape_detail, op_kind=op_kind)
                f['activation_types'].add(op_kind) 
                
            f['macs'] += macs
            f['flops'] += flops
        
        # 整理基礎特徵
        f['layer_types'] = ','.join(sorted(f['layer_types']))
        f['activation_types'] = ','.join(sorted(f['activation_types']))
        
        # === 計算所有衍伸特徵 (含 Memory, Density, Ratio) ===
        input_shape = parse_shape(f['input_tensor_shape'])
        output_shape = parse_shape(f['output_tensor_shape'])
        input_mem_MB = compute_memory_MB(input_shape, dtype_bytes=1) # DLA Int8 假設 1 byte
        output_mem_MB = compute_memory_MB(output_shape, dtype_bytes=1)
        total_mem_MB = input_mem_MB + output_mem_MB
        
        # 1. 記憶體特徵
        f['input_mem_MB'] = round(input_mem_MB, 6)
        f['output_mem_MB'] = round(output_mem_MB, 6)
        f['total_mem_MB'] = round(total_mem_MB, 6)
        f['log_total_mem_MB'] = np.log1p(total_mem_MB)
        
        # 2. 計算密度/強度特徵
        compute_density = f['macs'] / (total_mem_MB + 1e-9)
        compute_intensity = f['flops'] / (total_mem_MB + 1e-9)
        f['compute_density'] = round(compute_density, 6)
        f['compute_intensity'] = round(compute_intensity, 6)
        f['log_compute_density'] = np.log1p(compute_density)
        f['log_compute_intensity'] = np.log1p(compute_intensity)
        
        # 3. Log MACs/FLOPs 特徵
        f['log_macs'] = np.log1p(f['macs'])
        f['log_flops'] = np.log1p(f['flops'])

        # 4. Ratio 特徵
        depth_ratio = f['output_channels'] / (f['input_channels'] + 1e-9)
        aspect_ratio = (f['input_resolution'] + 1e-9) / (f['output_resolution'] + 1e-9)
        f['depth_ratio'] = round(depth_ratio, 6)
        f['aspect_ratio'] = round(aspect_ratio, 6)

        # === 新增：資料搬移量 / fallback 造成的搬移記憶體量 =====

        # 每一次搬移會搬 input + output tensor 的記憶體
        per_move_MB = input_mem_MB + output_mem_MB

        # 5. fallback / 資料搬移比例特徵
        if f['num_layers_total'] > 0:
            f['fallback_ratio'] = round(f['num_layers_gpu'] / (f['num_layers_total'] + 1e-9), 6)
            f['data_move_ratio'] = round(f['data_move_layers'] / (f['num_layers_total'] + 1e-9), 6)
        else:
            f['fallback_ratio'] = 0.0
            f['data_move_ratio'] = 0.0
        
        # === 清理特徵：刪除不需要輸出的內部計算輔助鍵 ===
        del f['input_resolution'] 
        del f['output_resolution']
        
        # 第一次迴圈決定輸出欄位順序
        if output_feature_keys is None:
            output_feature_keys = [
                'block_index', 'block_name', 'batch_size',
                'input_channels', 'output_channels', 'input_tensor_shape', 'output_tensor_shape',
                'num_layers', 'num_layers_total',
                'num_layers_dla', 'num_layers_gpu', 'fallback_ratio',
                'data_move_layers', 'data_move_ratio',
                'layer_types', 'has_conv', 'has_pooling', 'has_activation', 'activation_types',
                'macs', 'flops', 'log_macs', 'log_flops',
                'input_mem_MB', 'output_mem_MB', 'total_mem_MB', 'log_total_mem_MB',
                'compute_density', 'compute_intensity', 'log_compute_density', 'log_compute_intensity',
                'depth_ratio', 'aspect_ratio'
            ]


        # 只保留要輸出的欄位
        f_clean = {k: f.get(k, None) for k in output_feature_keys}
        features.append(f_clean)

    return features

def save_csv(results, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用第一個結果的 keys 來確保 fieldnames 順序和內容正確
    fieldnames = list(results[0].keys()) 
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description='Extract DLA block features from TensorRT log.')
    parser.add_argument('--log_path', type=str, required=True, help='Path to the TensorRT log file.')
    parser.add_argument('--output_dir', type=str, default=None, help='Power profile result directory')
    args = parser.parse_args()

    log_path = Path(args.log_path) 
    output_dir = Path(args.output_dir)

    output_csv = output_dir / (log_path.stem + '.csv')

    assert log_path.exists(), f"Log file does not exist: {log_path}"

    features = extract_features(log_path)
    save_csv(features, output_csv)
    print(f"Feature CSV saved to: {output_csv}")

if __name__ == '__main__':
    main()

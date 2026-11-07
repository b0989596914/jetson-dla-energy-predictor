"""
This file is used to convert a generated ONNX file into a TensorRT engine.


build_engine_general.py
(已加入 preset 支援，適合四種 block 切割模式)

# 此檔案作用 : 將 ONNX 檔案 轉為 TensorRT engine 檔案
# 此py檔案的output : "./engine/without_blockopt/VGG16/bs16/VGG16_cifar10_without_blockopt_bs16.engine" -> 之後用來優化推論
# 並將完整資訊寫入 log 檔案 -> 給feature_extractor.py 用來萃取特徵


The command we use is as follows:

without_blokopt:(未優化過後的onnx檔案)
VGG16:
bs16:
 python build_engine_general.py ./onnx/VGG16/bs16/VGG16_cifar10_bs16.onnx  --output=./engine/without_blockopt/VGG16/bs16/VGG16_cifar10_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs16/VGG16_cifar10_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=32 --output_class=10  --dataset=cifar10 --dataset_path=../../../Model/Data_cifar10/cifar10 --batch_size=16 --verbose > engine/without_blockopt/log/VGG16/bs16/VGG16_cifar10_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/VGG16/bs16/VGG16_tiny_bs16.onnx  --output=./engine/without_blockopt/VGG16/bs16/VGG16_tiny_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs16/VGG16_tiny_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=64 --output_class=200  --dataset=tiny-imagenet --batch_size=16 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG16/bs16/VGG16_tiny_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/VGG16/bs16/VGG16_imagenet100_bs16.onnx --output=./engine/without_blockopt/VGG16/bs16/VGG16_imagenet100_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs16/VGG16_imagenet100_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=16 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG16/bs16/VGG16_imagenet100_without_blockopt_bs16.log

bs8:
 python build_engine_general.py ./onnx/VGG16/bs8/VGG16_cifar10_bs8.onnx  --output=./engine/without_blockopt/VGG16/bs8/VGG16_cifar10_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs8/VGG16_cifar10_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=32 --output_class=10  --dataset=cifar10 --dataset_path=../../../Model/Data_cifar10/cifar10 --batch_size=8 --verbose > engine/without_blockopt/log/VGG16/bs8/VGG16_cifar10_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/VGG16/bs8/VGG16_tiny_bs8.onnx  --output=./engine/without_blockopt/VGG16/bs8/VGG16_tiny_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs8/VGG16_tiny_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=64 --output_class=200  --dataset=tiny-imagenet --batch_size=8 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG16/bs8/VGG16_tiny_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/VGG16/bs8/VGG16_imagenet100_bs8.onnx --output=./engine/without_blockopt/VGG16/bs8/VGG16_imagenet100_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs8/VGG16_imagenet100_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=8 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG16/bs8/VGG16_imagenet100_without_blockopt_bs8.log

bs4:
 python build_engine_general.py ./onnx/VGG16/bs4/VGG16_cifar10_bs4.onnx  --output=./engine/without_blockopt/VGG16/bs4/VGG16_cifar10_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs4/VGG16_cifar10_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=32 --output_class=10  --dataset=cifar10 --dataset_path=../../../Model/Data_cifar10/cifar10 --batch_size=4 --verbose > engine/without_blockopt/log/VGG16/bs4/VGG16_cifar10_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/VGG16/bs4/VGG16_tiny_bs4.onnx  --output=./engine/without_blockopt/VGG16/bs4/VGG16_tiny_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs4/VGG16_tiny_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=64 --output_class=200  --dataset=tiny-imagenet --batch_size=4 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG16/bs4/VGG16_tiny_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/VGG16/bs4/VGG16_imagenet100_bs4.onnx --output=./engine/without_blockopt/VGG16/bs4/VGG16_imagenet100_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs4/VGG16_imagenet100_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=4 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG16/bs4/VGG16_imagenet100_without_blockopt_bs4.log

bs1:
 python build_engine_general.py ./onnx/VGG16/bs1/VGG16_cifar10_bs1.onnx  --output=./engine/without_blockopt/VGG16/bs1/VGG16_cifar10_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs1/VGG16_cifar10_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=32 --output_class=10  --dataset=cifar10 --dataset_path=../../../Model/Data_cifar10/cifar10 --batch_size=1 --verbose > engine/without_blockopt/log/VGG16/bs1/VGG16_cifar10_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/VGG16/bs1/VGG16_tiny_bs1.onnx  --output=./engine/without_blockopt/VGG16/bs1/VGG16_tiny_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs1/VGG16_tiny_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=64 --output_class=200  --dataset=tiny-imagenet --batch_size=1 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG16/bs1/VGG16_tiny_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/VGG16/bs1/VGG16_imagenet100_bs1.onnx --output=./engine/without_blockopt/VGG16/bs1/VGG16_imagenet100_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG16/bs1/VGG16_imagenet100_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=1 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG16/bs1/VGG16_imagenet100_without_blockopt_bs1.log

VGG19:
bs16:
 python build_engine_general.py ./onnx/VGG19/bs16/VGG19_cifar10_bs16.onnx  --output=./engine/without_blockopt/VGG19/bs16/VGG19_cifar10_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs16/VGG19_cifar10_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=32 --output_class=10  --dataset=cifar10 --batch_size=16 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/VGG19/bs16/VGG19_cifar10_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/VGG19/bs16/VGG19_tiny_bs16.onnx  --output=./engine/without_blockopt/VGG19/bs16/VGG19_tiny_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs16/VGG19_tiny_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=64 --output_class=200  --dataset=tiny-imagenet --batch_size=16 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG19/bs16/VGG19_tiny_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/VGG19/bs16/VGG19_imagenet100_bs16.onnx --output=./engine/without_blockopt/VGG19/bs16/VGG19_imagenet100_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs16/VGG19_imagenet100_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=16 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG19/bs16/VGG19_imagenet100_without_blockopt_bs16.log

bs8:
 python build_engine_general.py ./onnx/VGG19/bs8/VGG19_cifar10_bs8.onnx --output=./engine/without_blockopt/VGG19/bs8/VGG19_cifar10_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs8/VGG19_cifar10_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=8 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/VGG19/bs8/VGG19_cifar10_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/VGG19/bs8/VGG19_tiny_bs8.onnx --output=./engine/without_blockopt/VGG19/bs8/VGG19_tiny_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs8/VGG19_tiny_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=8 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG19/bs8/VGG19_tiny_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/VGG19/bs8/VGG19_imagenet100_bs8.onnx --output=./engine/without_blockopt/VGG19/bs8/VGG19_imagenet100_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs8/VGG19_imagenet100_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=8 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG19/bs8/VGG19_imagenet100_without_blockopt_bs8.log

bs4:
 python build_engine_general.py ./onnx/VGG19/bs4/VGG19_cifar10_bs4.onnx --output=./engine/without_blockopt/VGG19/bs4/VGG19_cifar10_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs4/VGG19_cifar10_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=4 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/VGG19/bs4/VGG19_cifar10_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/VGG19/bs4/VGG19_tiny_bs4.onnx --output=./engine/without_blockopt/VGG19/bs4/VGG19_tiny_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs4/VGG19_tiny_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=4 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG19/bs4/VGG19_tiny_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/VGG19/bs4/VGG19_imagenet100_bs4.onnx --output=./engine/without_blockopt/VGG19/bs4/VGG19_imagenet100_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs4/VGG19_imagenet100_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=4 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG19/bs4/VGG19_imagenet100_without_blockopt_bs4.log


bs1:
 python build_engine_general.py ./onnx/VGG19/bs1/VGG19_cifar10_bs1.onnx --output=./engine/without_blockopt/VGG19/bs1/VGG19_cifar10_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs1/VGG19_cifar10_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=1 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/VGG19/bs1/VGG19_cifar10_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/VGG19/bs1/VGG19_tiny_bs1.onnx --output=./engine/without_blockopt/VGG19/bs1/VGG19_tiny_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs1/VGG19_tiny_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=1 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/VGG19/bs1/VGG19_tiny_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/VGG19/bs1/VGG19_imagenet100_bs1.onnx --output=./engine/without_blockopt/VGG19/bs1/VGG19_imagenet100_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/VGG19/bs1/VGG19_imagenet100_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=1 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/VGG19/bs1/VGG19_imagenet100_without_blockopt_bs1.log

ResNet50:
------------test--------------
 python build_engine_general.py ./onnx/ResNet50/bs16/ResNet50_imagenet100_input256_bs16.onnx --output=./engine/without_blockopt/ResNet50/bs16/ResNet50_imagenet100_input256_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs16/ResNet50_imagenet100_input256_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=256 --output_class=100 --dataset=imagenet100 --batch_size=16 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/ResNet50/bs16/ResNet50_imagenet100_input256_without_blockopt_bs16.log
------------test--------------

bs16:
 python build_engine_general.py ./onnx/ResNet50/bs16/ResNet50_cifar10_bs16.onnx  --output=./engine/without_blockopt/ResNet50/bs16/ResNet50_cifar10_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs16/ResNet50_cifar10_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=32 --output_class=10  --dataset=cifar10 --batch_size=16  --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/ResNet50/bs16/ResNet50_cifar10_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/ResNet50/bs16/ResNet50_tiny_bs16.onnx  --output=./engine/without_blockopt/ResNet50/bs16/ResNet50_tiny_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs16/ResNet50_tiny_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=64 --output_class=200  --dataset=tiny-imagenet --batch_size=16 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/ResNet50/bs16/ResNet50_tiny_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/ResNet50/bs16/ResNet50_imagenet100_bs16.onnx --output=./engine/without_blockopt/ResNet50/bs16/ResNet50_imagenet100_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs16/ResNet50_imagenet100_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=16 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/ResNet50/bs16/ResNet50_imagenet100_without_blockopt_bs16.log

bs8:
 python build_engine_general.py ./onnx/ResNet50/bs8/ResNet50_cifar10_bs8.onnx --output=./engine/without_blockopt/ResNet50/bs8/ResNet50_cifar10_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs8/ResNet50_cifar10_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=8 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/ResNet50/bs8/ResNet50_cifar10_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/ResNet50/bs8/ResNet50_tiny_bs8.onnx --output=./engine/without_blockopt/ResNet50/bs8/ResNet50_tiny_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs8/ResNet50_tiny_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=8 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/ResNet50/bs8/ResNet50_tiny_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/ResNet50/bs8/ResNet50_imagenet100_bs8.onnx --output=./engine/without_blockopt/ResNet50/bs8/ResNet50_imagenet100_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs8/ResNet50_imagenet100_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=8 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/ResNet50/bs8/ResNet50_imagenet100_without_blockopt_bs8.log

bs4:
 python build_engine_general.py ./onnx/ResNet50/bs4/ResNet50_cifar10_bs4.onnx --output=./engine/without_blockopt/ResNet50/bs4/ResNet50_cifar10_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs4/ResNet50_cifar10_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=4 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/ResNet50/bs4/ResNet50_cifar10_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/ResNet50/bs4/ResNet50_tiny_bs4.onnx --output=./engine/without_blockopt/ResNet50/bs4/ResNet50_tiny_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs4/ResNet50_tiny_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=4 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/ResNet50/bs4/ResNet50_tiny_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/ResNet50/bs4/ResNet50_imagenet100_bs4.onnx --output=./engine/without_blockopt/ResNet50/bs4/ResNet50_imagenet100_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs4/ResNet50_imagenet100_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=4 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/ResNet50/bs4/ResNet50_imagenet100_without_blockopt_bs4.log

bs1:
 python build_engine_general.py ./onnx/ResNet50/bs1/ResNet50_cifar10_bs1.onnx --output=./engine/without_blockopt/ResNet50/bs1/ResNet50_cifar10_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs1/ResNet50_cifar10_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=1 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/ResNet50/bs1/ResNet50_cifar10_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/ResNet50/bs1/ResNet50_tiny_bs1.onnx --output=./engine/without_blockopt/ResNet50/bs1/ResNet50_tiny_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs1/ResNet50_tiny_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=1 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/ResNet50/bs1/ResNet50_tiny_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/ResNet50/bs1/ResNet50_imagenet100_bs1.onnx --output=./engine/without_blockopt/ResNet50/bs1/ResNet50_imagenet100_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/ResNet50/bs1/ResNet50_imagenet100_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=1 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/ResNet50/bs1/ResNet50_imagenet100_without_blockopt_bs1.log

MobileNetV2:
bs16:
 python build_engine_general.py ./onnx/MobileNetV2/bs16/MobileNetV2_cifar10_bs16.onnx  --output=./engine/without_blockopt/MobileNetV2/bs16/MobileNetV2_cifar10_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs16/MobileNetV2_cifar10_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=32 --output_class=10  --dataset=cifar10 --batch_size=16  --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/MobileNetV2/bs16/MobileNetV2_cifar10_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/MobileNetV2/bs16/MobileNetV2_tiny_bs16.onnx  --output=./engine/without_blockopt/MobileNetV2/bs16/MobileNetV2_tiny_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs16/MobileNetV2_tiny_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=64 --output_class=200  --dataset=tiny-imagenet --batch_size=16 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/MobileNetV2/bs16/MobileNetV2_tiny_without_blockopt_bs16.log

 python build_engine_general.py ./onnx/MobileNetV2/bs16/MobileNetV2_imagenet100_bs16.onnx --output=./engine/without_blockopt/MobileNetV2/bs16/MobileNetV2_imagenet100_without_blockopt_bs16.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs16/MobileNetV2_imagenet100_without_blockopt_bs16.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=16 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/MobileNetV2/bs16/MobileNetV2_imagenet100_without_blockopt_bs16.log

bs8:
 python build_engine_general.py ./onnx/MobileNetV2/bs8/MobileNetV2_cifar10_bs8.onnx --output=./engine/without_blockopt/MobileNetV2/bs8/MobileNetV2_cifar10_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs8/MobileNetV2_cifar10_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=8 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/MobileNetV2/bs8/MobileNetV2_cifar10_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/MobileNetV2/bs8/MobileNetV2_tiny_bs8.onnx --output=./engine/without_blockopt/MobileNetV2/bs8/MobileNetV2_tiny_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs8/MobileNetV2_tiny_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=8 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/MobileNetV2/bs8/MobileNetV2_tiny_without_blockopt_bs8.log

 python build_engine_general.py ./onnx/MobileNetV2/bs8/MobileNetV2_imagenet100_bs8.onnx --output=./engine/without_blockopt/MobileNetV2/bs8/MobileNetV2_imagenet100_without_blockopt_bs8.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs8/MobileNetV2_imagenet100_without_blockopt_bs8.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=8 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/MobileNetV2/bs8/MobileNetV2_imagenet100_without_blockopt_bs8.log

bs4:
 python build_engine_general.py ./onnx/MobileNetV2/bs4/MobileNetV2_cifar10_bs4.onnx --output=./engine/without_blockopt/MobileNetV2/bs4/MobileNetV2_cifar10_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs4/MobileNetV2_cifar10_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=4 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/MobileNetV2/bs4/MobileNetV2_cifar10_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/MobileNetV2/bs4/MobileNetV2_tiny_bs4.onnx --output=./engine/without_blockopt/MobileNetV2/bs4/MobileNetV2_tiny_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs4/MobileNetV2_tiny_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=4 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/MobileNetV2/bs4/MobileNetV2_tiny_without_blockopt_bs4.log

 python build_engine_general.py ./onnx/MobileNetV2/bs4/MobileNetV2_imagenet100_bs4.onnx --output=./engine/without_blockopt/MobileNetV2/bs4/MobileNetV2_imagenet100_without_blockopt_bs4.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs4/MobileNetV2_imagenet100_without_blockopt_bs4.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=4 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/MobileNetV2/bs4/MobileNetV2_imagenet100_without_blockopt_bs4.log

bs1:
 python build_engine_general.py ./onnx/MobileNetV2/bs1/MobileNetV2_cifar10_bs1.onnx --output=./engine/without_blockopt/MobileNetV2/bs1/MobileNetV2_cifar10_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs1/MobileNetV2_cifar10_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=32 --output_class=10 --dataset=cifar10 --batch_size=1 --dataset_path=../../../Model/Data_cifar10/cifar10 --verbose > engine/without_blockopt/log/MobileNetV2/bs1/MobileNetV2_cifar10_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/MobileNetV2/bs1/MobileNetV2_tiny_bs1.onnx --output=./engine/without_blockopt/MobileNetV2/bs1/MobileNetV2_tiny_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs1/MobileNetV2_tiny_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=64 --output_class=200 --dataset=tiny-imagenet --batch_size=1 --dataset_path=../../../Model/Data_tiny_Imagenet/tiny-imagenet-200 --verbose > engine/without_blockopt/log/MobileNetV2/bs1/MobileNetV2_tiny_without_blockopt_bs1.log

 python build_engine_general.py ./onnx/MobileNetV2/bs1/MobileNetV2_imagenet100_bs1.onnx --output=./engine/without_blockopt/MobileNetV2/bs1/MobileNetV2_imagenet100_without_blockopt_bs1.engine --dla_core=0 --gpu_fallback --exportProfile=engine/without_blockopt/log/MobileNetV2/bs1/MobileNetV2_imagenet100_without_blockopt_bs1.json --int8 --separateProfileRun --input_size=224 --output_class=100 --dataset=imagenet100 --batch_size=1 --dataset_path=../../../Model/Data_imagenet100 --verbose > engine/without_blockopt/log/MobileNetV2/bs1/MobileNetV2_imagenet100_without_blockopt_bs1.log

"""


import argparse
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import tensorrt as trt
import time
from calibrator import DatasetCalibrator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import nvtx

parser = argparse.ArgumentParser()
parser.add_argument('onnx', type=str)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--max_workspace_size', type=int, default=1 << 28)
parser.add_argument('--int8', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--dla_core', type=int, default=None)
parser.add_argument('--gpu_fallback', action='store_true')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'tiny-imagenet', 'imagenet100'], default=None)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--input_size', type=int)
parser.add_argument('--output_class', type=int)
parser.add_argument('--exportProfile', type=str, default=None)
parser.add_argument('--separateProfileRun', action='store_true')
parser.add_argument('--verbose', action='store_true')

# ⭐⭐⭐ 新增 preset
parser.add_argument('--preset', type=str, choices=['A','B','C','D'], required=False,
                    help="block pattern preset type")

# ⭐⭐⭐ 新增 block config JSON（可選）
parser.add_argument('--block_json', type=str, default=None,
                    help="JSON generated from blockopt (optional)")

args = parser.parse_args()

# -----------------------------------------------------
# 1. Logger
# -----------------------------------------------------
logger_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
logger = trt.Logger(logger_level)

# -----------------------------------------------------
# 2. Dataset prepare
# -----------------------------------------------------
if args.dataset == 'cifar10':
    args.input_size = args.input_size or 32
    args.output_class = args.output_class or 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path, train=True, download=True, transform=transform)

elif args.dataset == 'tiny-imagenet':
    args.input_size = args.input_size or 64
    args.output_class = args.output_class or 200
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),(0.2302, 0.2265, 0.2262)),
    ])
    val_path = os.path.join(args.dataset_path, 'val', 'sorted')
    train_dataset = ImageFolder(val_path, transform=transform)

elif args.dataset == 'imagenet100':
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    val_path = os.path.join(args.dataset_path, 'val')
    train_dataset = ImageFolder(val_path, transform=transform)

# calibration batch
calib_data = torch.stack([
    train_dataset[i][0] for i in range(args.batch_size)
]).cuda()

# -----------------------------------------------------
# 3. Build TensorRT
# -----------------------------------------------------
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser_trt = trt.OnnxParser(network, logger)

with open(args.onnx, 'rb') as f:
    if not parser_trt.parse(f.read()):
        for i in range(parser_trt.num_errors):
            print(parser_trt.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

profile = builder.create_optimization_profile()
shape = (args.batch_size, 3, args.input_size, args.input_size)
profile.set_shape('input', shape, shape, shape)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.max_workspace_size)

if args.fp16:
    config.set_flag(trt.BuilderFlag.FP16)

if args.int8:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = DatasetCalibrator(calib_data, train_dataset)

if args.dla_core is not None:
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = args.dla_core

if args.gpu_fallback:
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

config.add_optimization_profile(profile)
config.set_calibration_profile(profile)

# ⭐⭐⭐ 把 preset / blocks 寫進 metadata
network_metadata = {
    "preset": args.preset,
    "block_json": args.block_json
}
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

engine = builder.build_serialized_network(network, config)

# -----------------------------------------------------
# Save engine
# -----------------------------------------------------
if args.output is not None:
    with open(args.output, 'wb') as f:
        f.write(engine)
    print(f"[OK] Engine saved → {args.output}")

# -----------------------------------------------------
# 4. Profiling
# -----------------------------------------------------
if args.exportProfile is not None:
    class MyProfiler(trt.IProfiler):
        def __init__(self):
            super(MyProfiler, self).__init__()
            self.layer_times = {}

        def report_layer_time(self, layer_name, ms):
            if "foreignnode" not in layer_name.lower():
                return
            if "copy" in layer_name.lower() or "reformatting" in layer_name.lower():
                return
            nvtx.annotate(message=layer_name, color="blue")(lambda: None)()
            self.layer_times[layer_name] = ms

    print("Running profiling...")

    runtime = trt.Runtime(logger)

    with runtime.deserialize_cuda_engine(engine) as engine:
        context = engine.create_execution_context()

        if context is None:
            raise RuntimeError(
                f"Engine saved to {args.output}, but FAILED to create execution context. "
                "Probably DLA resource overflow."
            )

        profiler = MyProfiler()
        context.profiler = profiler

        # prepare buffers
        input_binding_idx = engine.get_binding_index('input')
        output_binding_idx = engine.get_binding_index('output')

        input_shape = (args.batch_size, 3, args.input_size, args.input_size)
        output_shape = (args.batch_size, args.output_class)

        context.set_binding_shape(input_binding_idx, input_shape)

        input_buffer = torch.randn(input_shape, device='cuda')
        output_buffer = torch.zeros(output_shape, device='cuda')

        bindings = [None, None]
        bindings[input_binding_idx] = input_buffer.data_ptr()
        bindings[output_binding_idx] = output_buffer.data_ptr()

        # run once
        torch.cuda.synchronize()
        t0 = time.time()

        with nvtx.annotate("TRT Inference", color="green"):
            context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000

        # --------------------------------------------------
        # ⭐⭐⭐ profiling JSON output（加入 preset）
        # --------------------------------------------------
        profile_json = {
            "preset": args.preset,
            "block_json_path": args.block_json,
            "total_inference_time_ms": elapsed,
            "layers": [{"name": k, "time_ms": v}
                    for k, v in profiler.layer_times.items()]
        }

        # 若 block_json 存在，也一起寫入
        if args.block_json and os.path.exists(args.block_json):
            try:
                with open(args.block_json, 'r') as f:
                    profile_json["blockopt_info"] = json.load(f)
            except:
                profile_json["blockopt_info"] = "ERROR loading block_json"

        with open(args.exportProfile, 'w') as f:
            json.dump(profile_json, f, indent=4)

        print(f"[OK] Profiling JSON saved → {args.exportProfile}")

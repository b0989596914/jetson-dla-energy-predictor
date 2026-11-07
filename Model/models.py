"""
This file defines the model architectures.
The model architectures are dynamically adjusted based on input image size.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, vgg19_bn, resnet50, mobilenet_v2

__all__ = [
    'ModelVGG16',
    'ModelVGG19',
    'ModelResNet50',
    'ModelMobileNetV2',
    'MODELS'
]

class ModelVGG16(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        self.model = vgg16_bn(pretrained=True)

        if input_size == 224:
            self.model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            flattened_size = 512 * 7 * 7
            classifier = nn.Sequential(
                nn.Linear(flattened_size, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
        elif input_size > 224:  # 256 or 331
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flattened_size = 512 * 1 * 1
            classifier = nn.Sequential(
                nn.Linear(flattened_size, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
        else:
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flattened_size = 512
            classifier = nn.Sequential(
                nn.Linear(flattened_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        self.model.classifier = classifier

    def forward(self, x):
        return self.model(x)

class ModelVGG19(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        self.model = vgg19_bn(pretrained=True)

        if input_size == 224:
            self.model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            flattened_size = 512 * 7 * 7
            classifier = nn.Sequential(
                nn.Linear(flattened_size, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
        elif input_size > 224:  # 256 or 331
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flattened_size = 512 * 1 * 1
            classifier = nn.Sequential(
                nn.Linear(flattened_size, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
        else:
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flattened_size = 512
            classifier = nn.Sequential(
                nn.Linear(flattened_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )

        self.model.classifier = classifier

    def forward(self, x):
        return self.model(x)

class ModelResNet50(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        self.model = resnet50(pretrained=True)

        if input_size >= 224:
            classifier = nn.Sequential(
                nn.Linear(2048, 1000),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1000, num_classes)
            )
        else:
            classifier = nn.Sequential(
                nn.Linear(2048, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )

        self.model.fc = classifier

    def forward(self, x):
        return self.model(x)

class ModelMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, input_size=224):
        super().__init__()
        self.model = mobilenet_v2(pretrained=True)
    
        flattened_size = self.model.last_channel  # 1280
        
        # if input_size >= 224:
        #     self.pool = nn.Identity()
        # else:
        #     self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(flattened_size, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, num_classes)
        # )
        
        # 與 ckpt 一致：Dropout + Linear(1280 → 100)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(flattened_size, num_classes, bias=True),
        )
        

    def forward(self, x):
        # x = self.model.features(x)
        # x = self.pool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.classifier(x)
        x = self.model.features(x)  # 輸出形狀: (batch_size, 1280, H, W)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # GAP: (batch_size, 1280, 1, 1)
        x = torch.flatten(x, 1)  # 展平為 (batch_size, 1280)
        x = self.model.classifier(x)
        return x

MODELS = {
    'VGG16': ModelVGG16,
    'VGG19': ModelVGG19,
    'ResNet50': ModelResNet50,
    'MobileNetV2': ModelMobileNetV2
}
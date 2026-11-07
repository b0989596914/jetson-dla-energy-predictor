"""
This file defines the model architectures.
The model architectures are modified to support CIFAR-10 dataset.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, vgg19_bn, resnet50

__all__ = [
    'ModelVGG16',
    'ModelVGG19',
    'ModelResNet50',
    'MODELS'
]

class ModelVGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = vgg16_bn(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adjust for CIFAR-10
        self.model.classifier[0] = nn.Linear(512 * 1 * 1, 256)
        self.model.classifier[3] = nn.Linear(256, 256)
        self.model.classifier[6] = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.model(x)

class ModelVGG19(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = vgg19_bn(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adjust for CIFAR-10
        self.model.classifier[0] = nn.Linear(512 * 1 * 1, 256)
        self.model.classifier[3] = nn.Linear(256, 256)
        self.model.classifier[6] = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.model(x)

class ModelResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)  # Modify FC layer for CIFAR-10

    def forward(self, x):
        return self.model(x)

MODELS = {
    'VGG16': ModelVGG16,
    'VGG19': ModelVGG19,
    'ResNet50': ModelResNet50
}

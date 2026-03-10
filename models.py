"""
CIFAR용 ResNet 모델 정의
- ResNet-56: Teacher 모델 (각 group 9개 block)
- ResNet-20: Student 모델 (각 group 3개 block)

CIFAR용 ResNet은 ImageNet용과 구조가 다름:
- 첫 conv: 3x3 (ImageNet은 7x7), MaxPool 없음
- 3개 group, 필터 수: 16, 32, 64
- ResNet-N: 각 group에 (N-2)/6 개의 block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=100):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)   # 32x32
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)   # 16x16
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)   # 8x8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feature=False):
        """
        Args:
            x: 입력 이미지 (B, 3, 32, 32)
            return_feature: True이면 마지막 feature map도 반환
        Returns:
            logits: (B, num_classes)
            feature: (return_feature=True) 마지막 feature map (B, 64, 8, 8)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)  # feature map: (B, 64, 8, 8)
        feature = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        if return_feature:
            return logits, feature
        return logits


def resnet20(num_classes=100):
    """Student 모델"""
    return CIFARResNet([3, 3, 3], num_classes)

def resnet56(num_classes=100):
    """Teacher 모델"""
    return CIFARResNet([9, 9, 9], num_classes)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

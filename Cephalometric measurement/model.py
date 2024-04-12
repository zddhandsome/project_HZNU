import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck_coarse(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super(Bottleneck_coarse, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=True)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out
class ResNet_coarse(nn.Module):
    def __init__(self, block, num_blocks, num_classes=38):
        super(ResNet_coarse, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,64,kernel_size=5,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dilation=3)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dilation=5)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dilation=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dilation=dilation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.sigmoid(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 主网络
def ResNet_dilated():
    return ResNet_coarse(Bottleneck_coarse, [3, 4, 6, 3], num_classes=38)























# 这是一个使用了空洞卷积的ResNet网络，输入为单通道的图片，输出为一个38维的向量。该网络主要包含以下几个部分：
#
#
# 一个7x7的卷积层，步长为2，用于对输入图片进行特征提取。
#
# 4个残差块，每个块包含若干个空洞卷积层和恒等映射。其中第二个到第四个残差块使用了不同的空洞率(dilation rate)，以增加感受野。
#
# 一个全局平均池化层，用于将特征图降维为一个向量。
#
# 一个全连接层，将降维后的特征向量映射到38维，对应38个类别的概率。

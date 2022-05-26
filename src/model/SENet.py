import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(filter3,filter3//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3//16,filter3,kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)
        x1 = x1*x2
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x1 = x1 + x_shortcut
        x1 = self.relu(x1)
        return x1


class senet(nn.Module):
    def __init__(self, num, classes):
        super(senet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, (64, 64, 256), num[0],1)
        self.conv3 = self._make_layer(256, (128, 128, 512), num[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), num[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), num[3], 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self,in_channels, filters, num, stride=1):
        layers = []
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))
        return nn.Sequential(*layers)




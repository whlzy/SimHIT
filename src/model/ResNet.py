import torch.nn as nn
import torch
import torch.nn.functional as F



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        output = self.net(x)
        output += self.shortcut(x)
        output = F.relu(output)
        return output

class ResNet_18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_18, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 2, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channel, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_channel, channel, stride))
            else:
                layers.append(block(channel, channel, 1))
            self.in_channel = channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
import torch
import torch.nn as nn



class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=512, data_planes=1):
        block = BasicBlock
        super(SimpleCNN, self).__init__()
        self.in_planes = 64
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(data_planes, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return F.normalize(out)
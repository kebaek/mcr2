# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class ResNet50(torch.nn.Module):
    """ResNet50 with the softmax chopped off and the batchnorm frozen"""
    n_outputs = 2048

    def __init__(self, dropout=0.1, r=512, fd=512, num_classes=10):
        super(ResNet50, self).__init__()
        # self.network = torchvision.models.resnet18(pretrained=True)
        self.network = torchvision.models.resnet50(pretrained=True)
        self.freeze_bn()
        self.dropout = nn.Dropout(dropout)
        self.reshape = torch.nn.Sequential(
	    nn.Linear(self.n_outputs,self.n_outputs, bias=False),
            nn.BatchNorm1d(self.n_outputs),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_outputs, self.n_outputs, bias=True),
            nn.BatchNorm1d(self.n_outputs),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_outputs, fd, bias=True),
	)
        self.U = nn.Linear(fd, r)
        self.A = nn.Linear(num_classes, r)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)
        x = self.network.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.reshape(x)
        return F.normalize(x)
        # if self.fd != self.n_outputs:
        #     x = self.reshape(x)
        #     if self.hparams['norm'] == 1:
        #         return F.normalize(x)
        #     else:
        #         return x
        # else:
        #     return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, hparams):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.hparams = hparams
        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)
        self.reshape = torch.nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hparams['fd'], bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = x.mean(dim=(2,3))
        x = self.reshape(x)
        if self.hparams['norm'] == 1:
            return F.normalize(x)
        else:
            return x

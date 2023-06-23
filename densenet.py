import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.nn import Module


class DenseNet(Module):
    def __init__(self, num_classes=8):#1 for binary decision
        super(DenseNet, self).__init__()
        self.net = torchvision.models.densenet121(pretrained=True)
        # Feature extraction
        self.conv0 = self.net.features.conv0
        self.norm0 = self.net.features.norm0
        self.relu0 = self.net.features.relu0
        self.pool0 = self.net.features.pool0
        self.denseblock1 = self.net.features.denseblock1
        self.transition1 = self.net.features.transition1
        self.denseblock2 = self.net.features.denseblock2
        self.transition2 = self.net.features.transition2
        self.denseblock3 = self.net.features.denseblock3
        self.transition3 = self.net.features.transition3
        self.denseblock4 = self.net.features.denseblock4
        # Classification
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.classifier = nn.Sequential(nn.Linear(self.net.classifier.in_features, num_classes), nn.Sigmoid())

    def forward(self, inputs, phase="default"):
        # Feature extraction
        conv = self.conv0(inputs)
        norm = self.norm0(conv)
        relu = self.relu0(norm)
        pool = self.pool0(relu)
        deep1 = self.denseblock1(pool)
        trans1 = self.transition1(deep1)
        deep2 = self.denseblock2(trans1)
        trans2 = self.transition2(deep2)
        deep3 = self.denseblock3(trans2)
        trans3 = self.transition3(deep3)
        deep4 = self.denseblock4(trans3)
        # Classification
        out = self.avgpool(deep4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        if phase == 'deepfeatures':
            return deep4
        else:
            return out

def DenseNet121():
    """returns model"""
    return DenseNet()
    
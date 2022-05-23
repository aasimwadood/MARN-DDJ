import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch.nn.utils import weight_norm
import sys, math, l2proj


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, features, stride=1, weightnorm=None, shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = nn.Conv2d(inplanes, features, kernel_size=3, stride=stride,
                     padding=1, bias=True)

        self.batchnormal1 = nn.BatchNorm2d(num_features=features)
        self.relu1 = nn.PReLU(num_parameters=features, init=0.1)
        self.relu2 = nn.PReLU(num_parameters=features, init=0.1)
        self.relu3 = nn.PReLU(num_parameters=features, init=0.1)
        self.batchnormal2 = nn.BatchNorm2d(num_features=features)
        self.conv2 = nn.Conv2d(inplanes, features, kernel_size=3, stride=stride,
                     padding=1, bias=True)
        self.conv3 = nn.Conv2d(inplanes, features, kernel_size=3, stride=stride,padding=1, bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

    def forward(self, x):
        y = x
        out = self.relu3(x)
        # out = F.pad(out, (1, 1, 1, 1), 'reflect')
        out = self.conv1(out)
        out = out[:, :, :x.shape[2], :x.shape[3]]
        out = self.relu1(out)
        # out = F.pad(out, (1, 1, 1, 1), 'reflect')
        # out = self.batchnormal1(out) 
        out = self.conv2(out)
        out = out[:, :, :x.shape[2], :x.shape[3]]
        # out = self.batchnormal1(out)
        out = self.relu2(out)
        # out = F.pad(out, (1, 1, 1, 1), 'reflect')
        # out = self.batchnormal2(out) 
        out = self.conv3(out)
        # out = self.batchnormal2(out)
        out = out[:, :, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        return out
        
class NoiseRemoverBlock(nn.Module):

    def __init__(self, inplanes, features, stride=1, weightnorm=None, shortcut=True):
        super(NoiseRemoverBlock, self).__init__()
        kernel_size = 3
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=features, kernel_size=kernel_size,padding=1, bias=False)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.PReLU(num_parameters=features, init=0.1)

 

    def forward(self, x):
        y = x
        out = self.relu(self.bn(self.conv(x)))
        return y - out 

class MARNDD(nn.Module):
    def __init__(self, block, num_of_layers=17, kernel_size=3, color=True, weightnorm=None):
        self.inplanes = 64
        super(MARNDD, self).__init__()
        if color:
            channels = 3
        else:
            channels = 1

        self.startinplanes = 3
        self.inplanes = 64
        self.endinplanes = 3
        self.kernel_size = kernel_size
        self.padding = 1
        self.features = 64
        
        self.conv1 = nn.Conv2d(channels, self.features, kernel_size=5, stride=1, padding=2,
                               bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
        self.layer1 = self._make_layer(block, self.features, num_of_layers)
        self.conv_out = nn.ConvTranspose2d(self.features, channels, kernel_size=5, stride=1, padding=2,
                                           bias=True)
        self.conv2 = nn.Conv2d( self.features,channels, kernel_size=self.kernel_size, stride=1, padding=1,
                               bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)
        self.l2proj = l2proj.L2Proj()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights = np.sqrt(2 / (9. * 64)) * np.random.standard_normal(m.weight.data.shape)
                weights = np.random.normal(size=m.weight.data.shape,
                                           scale=np.sqrt(1. / m.weight.data.shape[1]))
                m.weight.data = torch.Tensor(weights)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.zeromean()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        # layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))

        for i in range(1, int(blocks // 2)) :
            layers.append(NoiseRemoverBlock(self.inplanes, planes, weightnorm=True, shortcut=True))
        for u in range(int(blocks // 2), blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        return nn.Sequential(*layers)

    def zeromean(self):

        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)

    def forward(self, x):
        y = x
        self.zeromean()
        out = F.pad(x, (1, 1, 1, 1), 'reflect')
        out = self.conv1(x)
        out = self.layer1(out)
        # out = self.conv2(out)
        out = self.conv_out(out)
        return out


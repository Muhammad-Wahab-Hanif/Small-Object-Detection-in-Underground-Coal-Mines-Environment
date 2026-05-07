"""
Common modules for MFF-YOLOv5s-4EL
Includes: SimAM attention, C3Ghost module, GhostConv
"""

import torch
import torch.nn as nn
import math


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None):
    """Auto-padding for convolution"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class GhostConv(nn.Module):
    """Ghost Convolution - cheaper alternative to standard convolution
    Source: GhostNet: More Features from Cheap Operations (CVPR 2020)
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Conv2d(c1, c_, k, s, padding=k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU() if act else nn.Identity()
        
        self.cheap_op = nn.Conv2d(c_, c_, 3, 1, padding=1, groups=c_, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        
    def forward(self, x):
        x1 = self.act(self.bn(self.conv(x)))
        x2 = self.act(self.bn2(self.cheap_op(x1)))
        return torch.cat([x1, x2], 1)


class C3Ghost(nn.Module):
    """C3 module with GhostBottleneck
    Replaces standard Bottleneck with GhostConv for efficiency
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck building block"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            depthwise_conv(c_, c_, 3, 1),
            GhostConv(c_, c2, 1, 1, act=False)
        )
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv(x) if self.shortcut else self.conv(x)


def depthwise_conv(c1, c2, k=3, s=1):
    """Depthwise convolution utility"""
    return nn.Conv2d(c1, c2, k, s, padding=k//2, groups=c1, bias=False)


class SimAM(nn.Module):
    """SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    Source: https://github.com/ZjjConan/SimAM
    Paper: SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks (ICML 2021)
    
    SimAM computes 3D attention weights (across both spatial and channel dimensions)
    without adding any trainable parameters.
    """
    def __init__(self, channels=None, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Compute mean and variance across spatial dimensions
        n = w * h - 1
        mu = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - mu).pow(2)
        var = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n
        
        # Energy function (inverse of attention)
        y = x_minus_mu_square / (4 * (var + self.e_lambda)) + 0.5
        
        # Apply sigmoid to get attention weights
        return x * self.activation(y)


class SimAM_C3(nn.Module):
    """C3 module with SimAM attention integrated"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1) for _ in range(n)))
        self.simam = SimAM()
        
    def forward(self, x):
        y1 = self.simam(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))
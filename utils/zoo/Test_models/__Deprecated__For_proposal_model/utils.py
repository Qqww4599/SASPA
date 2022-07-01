import torch
import torchvision.models as models
from torch import nn
import pdb
import torch.nn.functional as F
import sys


class basicblock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(basicblock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class GlobalCnn(nn.Module):
    """input: b,c(in_plane),h,w"""
    def __init__(self, in_plane, out_plane, downsample=False, upsample=False, mid_dim=32):
        super(GlobalCnn, self).__init__()

        self.in_plane = in_plane
        self.out_plane = out_plane
        if downsample:
            self.stride = 2
        else:
            self.stride = 1
        self.encoder1 = basicblock(in_plane, mid_dim, kernel_size=5, padding=2)
        self.encoder2 = basicblock(mid_dim, mid_dim, kernel_size=5, padding=2)
        self.encoder3 = basicblock(mid_dim, mid_dim, kernel_size=5, padding=2)
        self.final_encoder = basicblock(mid_dim, self.out_plane, stride=self.stride, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.final_encoder(x)
        x = self.relu(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')
        return x

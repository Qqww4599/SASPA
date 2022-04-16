import torch
import torchvision.models as models
from torch import nn
import pdb
import torch.nn.functional as F
import sys

class global_cnn(nn.Module):
    '''input: b,c(in_plane),h,w'''
    def __init__(self, in_plane, out_plane, downsample=False):
        super(global_cnn, self).__init__()

        self.in_plane = in_plane
        self.out_plane = out_plane
        self.downsample = True
        self.encoder = self._build_block(in_plane, self.out_plane)
    def _build_block(self, in_dim, out_dim, mid_dim=32):
        layers = []
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1))
        if self.downsample:
            layers.append(nn.Conv2d(mid_dim, out_dim, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder(x)
        return x


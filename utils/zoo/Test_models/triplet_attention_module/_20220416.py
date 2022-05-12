import torch
import pdb
import math
import sys
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt
import time
import warnings

from triple_attention import TripletAttention
'''
Global CNN module

測試conv_basic_block、conv_attn_blocks功能。
在Proposal model中使用。


input: b,in_plane, h/2, w/2
'''

class conv_basic_block(nn.Module):
    def __init__(self, inplane, outplane, plane=64, kernel=1, stirde=1, padding=1, dilation=1, downsample=True):
        super(conv_basic_block, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(inplane, plane, 7,1,3, dilation=dilation, bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(plane, plane, 3,1,1, dilation=dilation, bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(plane, plane, 3,1,1, dilation=dilation, bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(plane, outplane, 3,1,1, dilation=dilation, bias=False),
            nn.BatchNorm2d(outplane),
            nn.ReLU()
        )
        if downsample:
            self.downsample = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x = self.conv0(x)
        x = x + self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        if hasattr(self, 'downsample'):
            x = self.downsample(x)
        return x

class pos_embedding(nn.Module):
    '''
    input: x -> b,c,h,w
    code resource from https://github.com/YtongXie/CoTr/blob/main/CoTr_package/CoTr/network_architecture/DeTrans/position_encoding.py


    :parameter
        sin_cos: use sine, cosine embedding as main postion embedding. detail in
                https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#the-intuition
    :return: x -> b,c,h,w
    '''
    def __init__(self, inplane, temperature=10000, norm=False, scale=False):
        super(pos_embedding, self).__init__()
        self.inplane = inplane
        self.temp = temperature
        self.norm = norm
        if scale is not None and norm is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2*math.pi
        self.scale = scale

    def forward(self, x):
        assert x.shape[1] == (self.inplane)
        b, c, h, w = x.shape
        mask = torch.zeros(b,c,h,w, dtype=torch.bool).to('cuda')
        assert mask is not None
        not_mask = ~mask
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        y_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.norm:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale

        dim_tx = torch.arange(h, dtype=torch.float32, device=x.device)
        dim_tx = self.temp ** (2 * (dim_tx//2) / h) # 2 means 2 dims(x,y)
        pos_x = x_embed / dim_tx.to('cuda')
        pos_x = torch.stack(
            (
                pos_x[:,:,:,0::2].sin(), # b,c,h,w
                pos_x[:,:,:,1::2].cos(),
             ),
            dim=4
        ).flatten(3) # b,c,h,w

        dim_ty = torch.arange(h, dtype=torch.float32, device=x.device)
        dim_ty = self.temp ** (2 * (dim_ty//2) / h)
        pos_y = y_embed / dim_ty.to('cuda')
        pos_y = torch.stack(
            (
                pos_y[:, :, :, 0::2].sin(),  # b,c,h,w
                pos_y[:, :, :, 1::2].cos(),
            ),
            dim=4
        ).flatten(3) # b,c,h,w

        pos = torch.add(pos_x, pos_y)

        return pos

class conv_attn_blocks(nn.Module):
    def __init__(self, inplane, outplane, n_conv:int=4, n_attn:int=4, conv_mid_dim=64, ATTN=True):
        super(conv_attn_blocks, self).__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.n_conv = n_conv
        self.n_attn = n_attn

        # ----- impl of conv -----
        self.conv_mid_dim = conv_mid_dim
        self.conv_f_layer = nn.Conv2d(self.conv_mid_dim, self.outplane, 3,1,1)
        self.conv_layers = self._conv_layer()


        # ----- impl of attn -----
        if ATTN is not None:
            attn_inplane = inplane if inplane else ValueError('this must be a attention dim input!')
            self.pos_embed = None
            self.projection = None
            self.ATTN = TripletAttention(no_spatial=True)

    def _attn_layer(self, attn_inplane, attn_outplane, head=4, n_attn=1, downsample=False):
        pos_embed = pos_embedding(attn_inplane)


        # raise NotImplementedError('{} 還沒寫完'.format(conv_attn_blocks._attn_layer.__name__))
        return x
    def _conv_layer(self):
        conv_layers = []
        conv_layers.append(conv_basic_block(self.inplane, self.conv_mid_dim, downsample=False))
        if self.n_conv > 0:
            for _ in range(self.n_conv):
                conv_layers.append(conv_basic_block(self.conv_mid_dim, self.conv_mid_dim, downsample=False))

        conv_layers.append(self.conv_f_layer)
        # conv_layers.append(nn.Conv2d(self.conv_mid_dim, self.outplane, 3,1,1))

        # raise NotImplementedError('{} 還沒寫完'.format(conv_attn_blocks._conv_layer.__name__))
        return nn.Sequential(*conv_layers)

    def _make_layers(self, x):
        pass
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_layers(x)
        if hasattr(self, 'ATTN'):
            x = self.ATTN(x)
            print('use attn')
        return x



class global_cnn(nn.Module):
    def __init__(self, c_in=32, c_out=16, dilation=1):
        '''
        Parameters:
            c_in: 輸入特徵維度
            c_out: 輸出特徵維度。128*s = 16


        '''
        super(global_cnn, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dilation = dilation
        if self.dilation > 1:
            raise NotImplementedError('dilation>1  is not allow in this model!')




        self.layer0 = self._make_layer()
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # about kaiming_normal_ introduction:
                # https://zhuanlan.zhihu.com/p/53712833
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    def _forward_impl(self, x):
        b, c, h, w = x.shape


    def forward(self, x):
        output = self._forward_impl(x)
        return output


if __name__ == '__main__':
    x = torch.arange(3*4*4).reshape(1,3,4,4).to(torch.float32)
    m_conv = conv_attn_blocks(3,3)
    m = pos_embedding(3, norm=True, scale=True)
    x = x.to('cpu')
    x = m_conv(x)
    print(x.shape)
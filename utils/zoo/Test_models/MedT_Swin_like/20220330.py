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
import triple_attention as ta
from torch.utils.checkpoint import checkpoint
import axiel_attention_blocks as aab
import Swin_transformer_unet_expand_decoder_sys as swin

class conv1x1(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1x1, self).__init__()
        self.Conv = nn.Conv2d(int(in_planes), int(out_planes), kernel_size=1, stride=stride, bias=False)
    def forward(self, x):
        x = self.Conv(x)
        return x

class medical_transformer(nn.Module):
    def __init__(self, block, block2, layers, num_classes, groups, width_per_group,
                 relpace_stride_with_dilation=None, norm_layer=None, s=0.125, img_size=256,
                 img_chan=3):
        '''
        Attributes:
            block: 使用的第一個注意力模塊(attention block)
            block2: 使用的第二個注意力模塊(attention block)
            layers: 每個模塊需要使用的層數
            num_classes: 最終輸出的向量包含多少類別
            groups: 注意力機制的HEAD數量
            width_per_group: 每個group的基礎寬度(維度上的寬度)
            relpace_stride_with_dilation: 用擴張卷積代替卷積的步數
            norm_layer: 使用標準化層
            s: scale,表示維度進行縮放的倍數
            img_size: 輸入影像的大小
            img_chan: 輸入影像的原始通道數
        '''
        super(medical_transformer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = int(64*s) # 表示經過scale的通道/維度數量，作為模塊輸入通道數，隨著模塊設定改變。
        self.dilation = 1 # 預設1，表示卷積使用不使用擴張卷積
        if relpace_stride_with_dilation is None:
            relpace_stride_with_dilation = [False, False, False] # 後續將會使用到
        if len(relpace_stride_with_dilation) != 3:
            raise ValueError("擴張卷積預設None、或是使用包含3個元素的tuple，"
                             "這邊得到{}".format(relpace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # ----- 參數定義完成，接下來進行模型定義 -----
        # --- 全局特徵 注意力機制(Attention mechanism) ---
        self.conv1 = nn.Conv2d(img_chan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) # 先進行128,self.inplanes通道的bottleneck。此處有尺寸減半。
        self.bn1 = norm_layer(self.inplanes)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(128)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # -- 注意力層開始 --
        # 使用128*s作為輸出向量，但輸出時的self.inplane為256*s
        self.layer1 = self._make_layer(block, int(128*s), layers[0], kernel_size=(img_size//2))
        # 使用256*s作為輸出向量，但輸出時的self.inplane為512*s
        self.layer2 = self._make_layer(block, int(256*s), layers[1], stride=2, kernel_size=(img_size//2),
                                       dilate=relpace_stride_with_dilation[0])
        # Decoder定義
        self.decoder4 = nn.Conv2d(int(512*s), int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256*s), int(128*s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv2d(int(128*s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1) # 對於類別維度做softmax進行歸一化

        # ----- 區域特徵 注意力機制 -----
        self.conv1_p = nn.Conv2d(img_chan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False) # 先進行128,self.inplanes通道的bottleneck，此時self.inplanes應為512*s
        self.bn1_p = norm_layer(self.inplanes)
        self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_p = norm_layer(128)
        self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1)
        self.bn3_p = norm_layer(self.inplanes)
        self.relu_p = nn.ReLU(inplace=True)
        # -- 區域注意力層開始 --
        img_size_p = img_size // 4 # 每個patch大小為 (H/4, W/4)
        self.layer1_p = self._make_layer(block2, int(128*s), layers[0], kernel_size=img_size_p) # 第一層輸出尺寸不變
        self.layer2_p = self._make_layer(block2, int(256*s), layers[1], stride=2, kernel_size=img_size_p,
                                         dilate=relpace_stride_with_dilation[0]) # 尺寸減半，通道數倍增
        self.layer3_p = self._make_layer(block2, int(512*s), layers[2], stride=2, kernel_size=img_size_p,
                                         dilate=relpace_stride_with_dilation[1]) # 尺寸減半，通道數倍增
        self.layer4_p = self._make_layer(block2, int(1024*s), layers[3], stride=2, kernel_size=img_size_p,
                                         dilate=relpace_stride_with_dilation[2]) # 尺寸減半，通道數倍增
        # Decoder定義
        self.decoder1_p = nn.Conv2d(int(1024*2*s), int(1024*2*s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(int(1024*2*s), int(1024*s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(1024*s), int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(512*s) , int(256*s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(256*s) , int(128*s) , kernel_size=3, stride=1, padding=1)

        self.decoder_f = nn.Conv2d(int(128*s) , int(128*s) , kernel_size=3, stride=1, padding=1) # 最後再疊Conv學習
        self.adjust_p = nn.Conv2d(int(128*s), num_classes, kernel_size=1, stride=1, padding=1)
        self.soft_p = nn.Softmax(dim=1)

        # 測試區域，Swin加入模型測試的可能性。
        self.swin_block = swin.SwinTransformerBlock(dim=64, input_resolution=(64,64), num_heads=4, window_size=64)

    def _BCHW_to_BNC(self, x, inplanes, size ,reverse=False):
        '''
        測試用區域，將B,C,H,W轉換成B,N,C，便於nn.Linear使用
        Parameters:
            x: 輸入特徵，預設為(256,256,self.inplanes)
            reverse: 是否進行反轉換。預設為False(傳換成B,N,C)。如果reverse==True，轉換成B,C,H,W
        '''

        if not reverse:
            # assert inplanes == 8, f'self.inplanes must be 8, now {self.inplanes}'
            assert x.shape == (1, inplanes, size, size), '輸入尺寸應為(1,{},{},{}), 但現在輸入形狀為{}'.format(inplanes,
                                                                                                   size,
                                                                                                   size,
                                                                                                   x.shape,
                                                                                                   )
            B,C,H,W = x.shape
            x = x.resize(B,C,H*W).permute(0,2,1)
            return x
        else:
            H,W = size,size
            assert x.shape == (1,size*size,inplanes), '輸入尺寸應為(1,{},{}), 但現在輸入形狀為{}'.format(size*size,
                                                                                                   inplanes,
                                                                                                   x.shape,
                                                                                                   )
            # 為何self.inplanes在forward裡面調用會是256而不是8?256從哪裡來? 因為在init進行axialatten_block就會將self.inplane放大了
            B,N,C = x.shape
            x = x.resize(B,H,W,C).permute(0,3,1,2)
            return x

    def _forward_impl(self, x):
        xin = x
        _, _, H, W = xin.shape


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # --- 全局注意力層開始 ---
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # 1, 64, 64, 64

        # - 測試項目: 用Swin_block當作外加的module
        x2 = self._BCHW_to_BNC(x2, inplanes=int(512*0.125), size=64)
        x2 = self.swin_block(x2)
        x2 = self._BCHW_to_BNC(x2, inplanes=int(512*0.125), size=64 ,reverse=True)

        x = F.interpolate(self.decoder4(x2), scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.add(x, x1)
        x = F.interpolate(self.decoder5(x), scale_factor=2, mode='bilinear', align_corners=True)
        # end of full image training

        # --- 區域注意力訓練開始 ---
        x_loc = x.clone() # x_loc表示x的local特徵
        x_loc = self._local_attention(16, xin, x_loc)

        # ----- 結合全局特徵與區域特徵 -----
        x = torch.add(x, x_loc)
        x = F.relu(self.decoder_f(x)) # 最終進行一次conv學習
        x = self.adjust(x)
        return x


    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        '''
        Attributes:
            block: 使用的注意力模塊
            planes: 注意力模塊輸出的通道/維度
            blocks: 注意力模塊堆疊數量
            kernel_size: 卷積核大小
            stride: 移動步長
            dilate: 是否使用擴張卷積
        '''
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.stride *= stride # 步長*擴張數，此處擴張表示步長的擴增
            stride = 1 # 原始步長不變
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 如果步長不為1，且 輸入通道數 與 注意力模塊 接受不同時(planes*模塊擴增數)，需改變維度。
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion),
            )
        layers = [] # 注意力模塊堆疊層，所有堆疊後的模塊都會聚集在這邊

        # 首先，先加入一層注意力模塊，預設第一層不會尺寸縮小，
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion # 第一層注意力結束後，更改輸入通道數，每次以2倍上升。
        # 值得注意的是，直接修改內部屬性，作為下次通道輸入使用

        if stride != 1:
            kernel_size = kernel_size // 2 # 如果步長不為1，則更改捲積核大小

        # 第一層結束後，要疊更多層就需要透過此處迴圈
        for _ in range(1, blocks):
            # 此處沒有 stride, downsample 使用模塊預設值
            layers.append(block(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=previous_dilation,
                          norm_layer=norm_layer, kernel_size=kernel_size))
        return nn.Sequential(*layers)
    def _local_attention(self, patches, xin, x_loc):
        '''學習patch注意力，包含注意力層前的埢積部分
        Parameter:
            xin: B,3,H,W，是輸入的原影像
            x_loc: 是global輸出的影像，但只是作為定位用。
        '''
        _, _ ,H, W = xin.shape
        H_len, W_len = int(patches ** 0.5), int(patches ** 0.5)
        for i in range(0, H_len):
            for j in range(0, H_len):
                h, w = H // H_len, W // W_len  # assume H,W = 256
                x_p = xin[:, :, h * i:h * (i + 1), w * j:w * (j + 1)]
                # begin patch wise
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)
                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)
                x_p = self.conv3_p(x_p) # B,64,H/8,H/8
                x_p = self.bn3_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)

                # x = self.maxpool(x)
                x1_p = self.layer1_p(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                x4_p = self.layer4_p(x3_p)

                x_p = F.relu(F.interpolate(checkpoint(self.decoder1_p, x4_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(F.interpolate(checkpoint(self.decoder2_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(F.interpolate(checkpoint(self.decoder3_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(checkpoint(self.decoder4_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(F.interpolate(checkpoint(self.decoder5_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                x_loc[:, :, h * i:h * (i + 1), w * j:w * (j + 1)] = x_p
        return x_loc
    def forward(self, x):
        output = self._forward_impl(x)
        return output

def medt():
    '重新編排+標註後的medt'
    layers = [1, 2, 4, 1]
    model = medical_transformer(aab.AxialBlock_conv_dynamic,
                                aab.AxialBlock_wopos,
                                layers,
                                num_classes=2,
                                groups=8,
                                width_per_group=64,)
    return model


def main(x):
    # ----use model---
    model = medt().cuda()

    out = model(x.cuda())
    out = out[:,1,:,:].squeeze(0).detach().cpu().numpy()

    fig, axs = plt.subplots(1,2)
    [axi.set_axis_off() for axi in axs.ravel()] # turn of all axis
    axs[0].imshow(x.permute(0,2,3,1).squeeze(0).numpy())
    axs[1].imshow(out)

    plt.show()


if __name__ == '__main__':
    test = torch.div(torch.arange(0,256*256*3).resize(1,3,256,256), 256*256*3)
    main(test)


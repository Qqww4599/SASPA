import torch
import pdb
import math
import sys
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from axialattn import AxialAttention
from torchvision import models

class resnet_branch(nn.Module):
    '''Get ResNet models output'''
    def __init__(self, model_name='resnet101', mid_dim=64):
        super().__init__()
        # ---提取特徵的resnet model---
        net = models.resnet18(pretrained=True)
        self.feature_extract_model = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=1, bias=False),
            net.conv1,
            net.bn1,
            net.layer1,
            net.layer2,  # (B,128,64,64)
        )
        # self.loss_1 = nn.Sequential(nn.Conv2d(256, mid_dim, kernel_size=7, stride=1, padding=3),
        #                             nn.Conv2d(mid_dim, 2, kernel_size=7, stride=1, padding=3)) # to loss_1
        self.loss_1 = nn.Sequential(nn.Conv2d(256, mid_dim, kernel_size=7, stride=1, padding=3))  # to loss_1
        self.F2_extract = net.layer3  # (B,256,32,32)
        self.F1_meet_F2 = nn.Conv2d(1024, 2048, kernel_size=1)  # 調整通道數，F1調整到256，讓F1和F2可以相加
        self.F1_adjust_for_loss_3 = nn.Conv2d(256, mid_dim, kernel_size=1)
        self.model = models.resnet101(pretrained=True) if model_name in {'resnet101'} else ValueError('should put ResNet Name')

    def forward(self, x):
        '''x = (B,3,H,W)'''
        # ---ResNet feature extraction branch---

        # F1 = (B,256,H//2,W//2)
        # F2 = (B,256,H//2,W//2)

        x = self.model.conv1(x)
        x = self.model.bn1(x)

        l1 = self.model.layer1(x)
        l2 = self.model.layer2(l1)

        l3 = self.model.layer3(l2)        # torch.Size([4, 1024, 32, 32])
        F1 = F.relu(F.interpolate(self.F1_meet_F2(l3), scale_factor=(2, 2), mode='bilinear'))  # (B,2048,H//2,W//2)

        l4 = self.model.layer4(l3)        # torch.Size([4, 2048, 16, 16])
        F2 = F.relu(F.interpolate(l4 , scale_factor=(4, 4), mode='bilinear'))  # (B,2048,H//2,W//2)

        loss_1 = self.loss_1((F1 + F2))  # (B,2048,H//2,W//2)

        return F1, F2, loss_1

class attn_origin_branch(nn.Module):
    def __init__(self, in_dim, out_dim, num_patch, attn_depth=2, mid_dim=64):
        '''
        使用三層金字塔結構(global, Original, Tiny)
        input:
            dim_in: 表示輸入feature的通道數量
            dim_out: 輸出feature通道數量
            attn_depth:表示block深度
            head: attention head數量
        '''
        super().__init__()
        self.dims_dilation = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.conv0 = nn.Conv2d(in_dim, mid_dim, kernel_size=1)

        self.ori_conv1_p = nn.Conv2d(mid_dim, 128, kernel_size=3, stride=1, padding=1)
        self.o_bn1 = nn.BatchNorm2d
        self.ori_conv2_p = nn.Conv2d(128, mid_dim, kernel_size=3, stride=1, padding=1)
        self.o_bn2 = nn.BatchNorm2d
        self.attn_local_ori_1 = self._build_attn_block(dim_in=mid_dim, attn_depth=2)  # (B, C, H, W)
        self.ori_down1 = nn.Conv2d(mid_dim, mid_dim * 2, kernel_size=3, padding=1, stride=2)
        self.attn_local_ori_2 = self._build_attn_block(dim_in=mid_dim * 2, attn_depth=2)  # (B, C*2, H//2, W//2)
        self.ori_down2 = nn.Conv2d(mid_dim * 2, mid_dim * 4, kernel_size=3, padding=1, stride=2)
        self.attn_local_ori_3 = self._build_attn_block(dim_in=mid_dim * 4, attn_depth=2)  # (B, C*4, H//4, W//4)
        self.ori_down3 = nn.Conv2d(mid_dim * 4, mid_dim * 8, kernel_size=3, padding=1, stride=2)
        self.attn_local_ori_4 = self._build_attn_block(dim_in=mid_dim * 8, attn_depth=2)  # (B, C*8 ,H//8, W//8)
    def _build_attn_block(self, dim_in, attn_depth=2, heads=8):
        layers = []
        for i in range(attn_depth):
            layers.append(AxialAttention(dim=dim_in, heads=heads, dim_index=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        # ---local_ori_branch---
        # x_loc_ori = x.clone() # (B,3,H,W)
        x = self.dims_dilation(x) # (B,mid_dim=64,H,W)
        x_loc_ori = self.conv0(x)  # (B,mid_dim=64,H,W)
        _, _, H, W = x_loc_ori.shape
        for i in range(0, 4):
            for j in range(0, 4):
                h, w = H // 4, W // 4
                x_p = x_loc_ori[:, :, h * i:h * (i + 1), w * j:w * (j + 1)]
                # ---Transformer Encoder---
                x_ori_l1 = self.attn_local_ori_1(x_p)  # (B,C,p_H,p_W)
                x_ori_l2 = self.ori_down1(x_ori_l1)
                x_ori_l2 = self.attn_local_ori_2(x_ori_l2)
                x_ori_l3 = self.ori_down2(x_ori_l2)  # if size=128, this patch will be 8x8
                x_ori_l3 = self.attn_local_ori_3(x_ori_l3) + x_ori_l3  # (B, C*4, p_H//4, p_W//4)
                # x_ori_l4 = self.ori_down3(x_ori_l3)
                # x_ori_l4 = self.attn_local_ori_4(x_ori_l4) + x_ori_l4 # residual, # (B, C*8, p_H//8, p_W//8)

                # ---Decoder---
                # x_ori_l3 = torch.add(x_ori_l3, F.relu(F.interpolate(checkpoint(self.ori_local_decoder3, x_ori_l4), scale_factor=(2, 2), mode='bilinear'))) # (B,C*4,p_H//4,p_W//4)
                # x_ori_l3 = torch.add(x_ori_l4, x_ori_l3) # (B,C*4,p_H//4,p_W//4)
                x_ori_l2 = torch.add(x_ori_l2, F.relu(
                    F.interpolate(checkpoint(self.ori_local_decoder2, x_ori_l3), scale_factor=(2, 2),
                                  mode='bilinear')))  # (B,C*2,p_H//2,p_W//2)
                x_ori_l1 = torch.add(x_ori_l1, F.relu(
                    F.interpolate(checkpoint(self.ori_local_decoder1, x_ori_l2), scale_factor=(2, 2),
                                  mode='bilinear')))  # (B,C,p_H,p_W)
                x_loc_ori[:, :, h * i:h * (i + 1), w * j:w * (j + 1)] = x_ori_l1  # x_local_ori.shape = (B, C, H, W)
                del x_ori_l1, x_ori_l2, x_ori_l3

        # try let all branch output dim = 64 = mid_dim
        # end_of_ori = self._fit_dims(x_loc_ori, 64, 2) # (B,2,H,W)
        end_of_ori = x_loc_ori  # (B,C,H,W) # try it!!
        return end_of_ori

class attn_global_branch(nn.Module):
    def __init__(self, in_dim, out_dim, num_patch, attn_depth=2, mid_dim=64):
        super().__init__()
        self.switch_dim_global_branch = nn.Conv2d(in_dim, mid_dim, kernel_size=7, stride=1, padding=3,
                                                  bias=False)  # 將stride改為1，大小才不會降
        self.atten_global_l1 = self._build_attn_block(dim_in=mid_dim, attn_depth=2)  # (B,C,H,W)
        # self.global_avgpool_l1 = nn.AvgPool2d(kernel_size=2, stride=1)
        self.downsample_global = nn.Conv2d(mid_dim, mid_dim * 2, kernel_size=3, padding=1, stride=2)
        self.atten_global_l2 = self._build_attn_block(dim_in=mid_dim * 2, attn_depth=2)  # (B,C*2,H//2,W//2)
        self.global_decoder2 = nn.Conv2d(mid_dim * 2, mid_dim, kernel_size=3, stride=1, padding=1)
        self.loss_2 = nn.Conv2d(mid_dim, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.switch_dim_global_branch(x)
        x = self.switch_dim_global_branch(x)  # (B,mid_dim=64,H,W)
        x_l1 = x.clone()  # (B,mid_dim=64,H,W)
        x_l1 = self.atten_global_l1(x_l1)  # (B,C,H,W)
        x_l1 = self.downsample_global(x_l1)  # (B,C*2,H//2,W//2)
        x_l2 = self.atten_global_l2(x_l1) + x_l1  # residual, # (B,C*2,H//2,W//2)

        x += F.relu(
            F.interpolate(checkpoint(self.global_decoder2, x_l2), scale_factor=(2, 2), mode='bilinear'))  # (B,C,H,W)
        # loss_2 = self.loss_2(x_g) # loss_2, (B,2,H,W)
        loss_2 = x  # loss_2, (B,C,H,W) # try it!!
        return x

class tiny(nn.Module):
    def __init__(self, in_dim, out_dim, num_patch, attn_depth=2, mid_dim=64):
        super().__init__()
        # ------attn_local_tiny表示size // 2後切成patch的attention-----
        self.dims_dilation = nn.Conv2d(in_dim, mid_dim, kernel_size=2)
        self.conv0 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1) # 多一層捲積層

        self.tiny_conv1_p = nn.Conv2d(mid_dim, 128, kernel_size=3, stride=1, padding=1)
        self.t_bn1 = nn.BatchNorm2d
        self.tiny_conv2_p = nn.Conv2d(128, mid_dim, kernel_size=3, stride=1, padding=1)
        self.t_bn2 = nn.BatchNorm2d
        self.attn_local_tiny_1 = self._build_attn_block(dim_in=mid_dim, attn_depth=2)  # (B, C, H, W)
        self.tiny_down1 = nn.Conv2d(mid_dim, mid_dim * 2, kernel_size=3, padding=1, stride=2)
        self.attn_local_tiny_2 = self._build_attn_block(dim_in=mid_dim * 2, attn_depth=2)  # (B, C*2, H//2, W//2)
        self.tiny_down2 = nn.Conv2d(mid_dim * 2, mid_dim * 4, kernel_size=3, padding=1, stride=2)
        self.attn_local_tiny_3 = self._build_attn_block(dim_in=mid_dim * 4, attn_depth=2)  # (B, C*4, H//4, W//4)
        self.tiny_down3 = nn.Conv2d(mid_dim * 4, mid_dim * 8, kernel_size=3, padding=1, stride=2)
        self.attn_local_tiny_4 = self._build_attn_block(dim_in=mid_dim * 8, attn_depth=2)  # (B, C*8 ,H//8, W//8)

        self.tiny_local_decoder3 = nn.Conv2d(mid_dim * 8, mid_dim * 4, kernel_size=3, stride=1,
                                             padding=1)  # (B, C*4 ,H//8, W//8)
        self.tiny_local_decoder2 = nn.Conv2d(mid_dim * 4, mid_dim * 2, kernel_size=3, stride=1,
                                             padding=1)  # (B, C*2 ,H//4, W//4)
        self.tiny_local_decoder1 = nn.Conv2d(mid_dim * 2, mid_dim, kernel_size=3, stride=1,
                                             padding=1)  # (B, C ,H//2, W//2)
        self.tiny_local_for_loss_3 = nn.Conv2d(mid_dim, 2, kernel_size=3, stride=1,
                                               padding=1)  # (B, C -> 2 ,H//2, W//2)
    def _build_attn_block(self, dim_in, attn_depth=1, heads=8):
        '''
        input:
            dim_in: 表示輸入feature的通道數量
            attn_depth:表示block深度
            head: attention head數量
        '''
        layers = []
        for i in range(attn_depth):
            layers.append(AxialAttention(dim=dim_in, heads=heads, dim_index=1))
        return nn.Sequential(*layers)

    def forward(self, x, F1):
        # ---local_tiny_branch---, for tiny branch use
        x = self.dims_dilation(x) # (B, C, H//2,W//2)
        x_local_tiny = self.conv0(x)  # size to 1/2, (B,C,H//2,W//2)
        x_loc_t = x_local_tiny
        x_loc_t = self.switch_dim_global_branch(x_loc_t)  # (B,mid_dim=64,H,W)
        _, _, H, W = x_loc_t.shape
        for i in range(0, 4):
            for j in range(0, 4):
                h, w = H // 4, W // 4
                x_p = x_loc_t[:, :, h * i:h * (i + 1), w * j:w * (j + 1)]

                # ---Transformer Encoder---
                x_tiny_l1 = self.attn_local_tiny_1(x_p)  # (B,C,H,W)
                x_tiny_l2 = self.tiny_down1(x_tiny_l1)  # if size=128, this patch will be 8x8
                x_tiny_l2 = self.attn_local_tiny_2(x_tiny_l2) + x_tiny_l2
                # x_tiny_l3 = self.tiny_down2(x_tiny_l2)
                # x_tiny_l3 = self.attn_local_tiny_3(x_tiny_l3)
                # x_tiny_l4 = self.tiny_down3(x_tiny_l3)
                # x_tiny_l4 = self.attn_local_tiny_4(x_tiny_l4) + x_tiny_l4  # residual, # (B,C*8,H//8,W//8)

                # ---Decoder---
                # x_tiny_l3 = torch.add(x_tiny_l3, F.relu(F.interpolate(checkpoint(self.tiny_local_decoder3, x_tiny_l4), scale_factor=(2, 2),
                #                                  mode='bilinear')))  # (B,C*4,H//4,W//4)
                # x_tiny_l2 = torch.add(x_tiny_l2, F.relu(F.interpolate(checkpoint(self.tiny_local_decoder2, x_tiny_l3), scale_factor=(2, 2),
                #                                  mode='bilinear')))  # (B,C*2,H//2,W//2)
                x_tiny_l1 = torch.add(x_tiny_l1, F.relu(
                    F.interpolate(checkpoint(self.tiny_local_decoder1, x_tiny_l2), scale_factor=(2, 2),
                                  mode='bilinear')))  # (B,C,H,W)
                x_loc_t[:, :, h * i:h * (i + 1), w * j:w * (j + 1)] = x_tiny_l1  # (B, mid_dim=64, H//2, W//2)
                del x_tiny_l1, x_tiny_l2
        x_loc_t = F.relu(
            F.interpolate(x_loc_t, scale_factor=(2, 2), mode='bilinear'))  # (B, mid_dim=64, H//2, W//2) -> (B,C,H,W)

        # F1: (B,256,H//2,W//2) -> (B,C,H//2,W//2) -> (B,C,H,W)
        F1 = F.relu(F.interpolate(self.F1_adjust_for_loss_3(F1), scale_factor=(2, 2), mode='bilinear'))
        # loss_3 = self.tiny_local_for_loss_3(x_loc_t + F1) # (B,C,H,W) -> (B,2,H,W)
        end_of_tiny = torch.add(x_loc_t, F1)  # (B,C,H,W) # try it!!

        return end_of_tiny

# --------------------main model structure---------------------
class advance_seg_transformer(nn.Module):
    mid_dim = 64
    def __init__(self, input_dim=3, img_size=128, out_dim=1, head=8, mid_dim=mid_dim, resnet_mlp=False):
        '''
        :param input_dim: 輸入影像維度
        :param img_size: 輸入影像尺寸
        :param out_dim: 輸出影像通道數
        :param head: attention head
        :param mid_dim: 中間通道數，用於影像轉換到global_branch的輸出通道數，表示特徵大小，預設為可學習
        '''
        super(advance_seg_transformer, self).__init__()
        self.out_dim = out_dim
        # ---提取特徵的resnet model---
        self.linear = nn.Sequential(nn.Linear(512, 1024),  nn.Linear(1024, 512))
        net = models.resnet18(pretrained=True)
        self.feature_extract_model = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=1, bias=False),
            net.conv1,
            net.bn1,
            net.layer1,
            net.layer2, # (B,128,64,64)
        )
        # self.loss_1 = nn.Sequential(nn.Conv2d(256, mid_dim, kernel_size=7, stride=1, padding=3),
        #                             nn.Conv2d(mid_dim, 2, kernel_size=7, stride=1, padding=3)) # to loss_1
        self.loss_1 = nn.Sequential(nn.Conv2d(256, mid_dim, kernel_size=7, stride=1, padding=3)) # to loss_1
        self.F2_extract = net.layer3 # (B,256,32,32)
        self.F1_meet_F2 = nn.Conv2d(128, 256, kernel_size=1) # 調整通道數，F1調整到256，讓F1和F2可以相加
        self.F1_adjust_for_loss_3 = nn.Conv2d(256, mid_dim, kernel_size=1)
        # ---transformer in branches---
        self.relu = nn.ReLU(inplace=True)

        # ---attention block---
        self.switch_dim_global_branch = nn.Conv2d(3,mid_dim, kernel_size=7, stride=1, padding=3,
                                 bias=False) # 將stride改為1，大小才不會降
        self.atten_global_l1 = self._build_attn_block(dim_in=mid_dim,attn_depth=2) # (B,C,H,W)
        # self.global_avgpool_l1 = nn.AvgPool2d(kernel_size=2, stride=1)
        self.downsample_global = nn.Conv2d(mid_dim, mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.atten_global_l2 = self._build_attn_block(dim_in=mid_dim*2,attn_depth=2) # (B,C*2,H//2,W//2)
        self.global_decoder2 = nn.Conv2d(mid_dim*2, mid_dim, kernel_size=3, stride=1, padding=1)
        self.loss_2 = nn.Conv2d(mid_dim, 2, kernel_size=3, stride=1, padding=1)

        # ------attn_local_ori表示原始size切成patch的attention------
        self.ori_conv1_p = nn.Conv2d(mid_dim, 128, kernel_size=3, stride=1, padding=1)
        self.o_bn1 = nn.BatchNorm2d
        self.ori_conv2_p = nn.Conv2d(128, mid_dim, kernel_size=3, stride=1, padding=1)
        self.o_bn2 = nn.BatchNorm2d
        self.attn_local_ori_1 = self._build_attn_block(dim_in=mid_dim,attn_depth=2) # (B, C, H, W)
        self.ori_down1 = nn.Conv2d(mid_dim,mid_dim*2, kernel_size=3, padding=1, stride=2)
        self.attn_local_ori_2 = self._build_attn_block(dim_in=mid_dim*2,attn_depth=2) # (B, C*2, H//2, W//2)
        self.ori_down2 = nn.Conv2d(mid_dim*2,mid_dim*4, kernel_size=3, padding=1, stride=2)
        self.attn_local_ori_3 = self._build_attn_block(dim_in=mid_dim*4,attn_depth=2) # (B, C*4, H//4, W//4)
        self.ori_down3 = nn.Conv2d(mid_dim*4,mid_dim*8, kernel_size=3, padding=1, stride=2)
        self.attn_local_ori_4 = self._build_attn_block(dim_in=mid_dim*8,attn_depth=2) # (B, C*8 ,H//8, W//8)

        self.ori_local_decoder3 = nn.Conv2d(mid_dim*8, mid_dim*4, kernel_size=3, stride=1, padding=1) # (B, C*4 ,H//8, W//8)
        self.ori_local_decoder2 = nn.Conv2d(mid_dim*4, mid_dim*2, kernel_size=3, stride=1, padding=1) # (B, C*2 ,H//4, W//4)
        self.ori_local_decoder1 = nn.Conv2d(mid_dim*2, mid_dim, kernel_size=3, stride=1, padding=1) # (B, C ,H//2, W//2)

        #------attn_local_tiny表示size // 2後切成patch的attention-----
        self.img_to_tiny = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)

        self.tiny_conv1_p = nn.Conv2d(mid_dim, 128, kernel_size=3, stride=1, padding=1)
        self.t_bn1 = nn.BatchNorm2d
        self.tiny_conv2_p = nn.Conv2d(128, mid_dim, kernel_size=3, stride=1, padding=1)
        self.t_bn2 = nn.BatchNorm2d
        self.attn_local_tiny_1 = self._build_attn_block(dim_in=mid_dim, attn_depth=2)  # (B, C, H, W)
        self.tiny_down1 = nn.Conv2d(mid_dim, mid_dim * 2, kernel_size=3, padding=1, stride=2)
        self.attn_local_tiny_2 = self._build_attn_block(dim_in=mid_dim * 2, attn_depth=2)  # (B, C*2, H//2, W//2)
        self.tiny_down2 = nn.Conv2d(mid_dim * 2, mid_dim * 4, kernel_size=3, padding=1, stride=2)
        self.attn_local_tiny_3 = self._build_attn_block(dim_in=mid_dim * 4, attn_depth=2)  # (B, C*4, H//4, W//4)
        self.tiny_down3 = nn.Conv2d(mid_dim * 4, mid_dim * 8, kernel_size=3, padding=1, stride=2)
        self.attn_local_tiny_4 = self._build_attn_block(dim_in=mid_dim * 8, attn_depth=2)  # (B, C*8 ,H//8, W//8)

        self.tiny_local_decoder3 = nn.Conv2d(mid_dim * 8, mid_dim * 4, kernel_size=3, stride=1,
                                            padding=1)  # (B, C*4 ,H//8, W//8)
        self.tiny_local_decoder2 = nn.Conv2d(mid_dim * 4, mid_dim * 2, kernel_size=3, stride=1,
                                            padding=1)  # (B, C*2 ,H//4, W//4)
        self.tiny_local_decoder1 = nn.Conv2d(mid_dim * 2, mid_dim, kernel_size=3, stride=1,
                                            padding=1)  # (B, C ,H//2, W//2)
        self.tiny_local_for_loss_3 = nn.Conv2d(mid_dim, 2,kernel_size=3, stride=1, padding=1)# (B, C -> 2 ,H//2, W//2)

        # ---other setting---
        self.resnet_mlp = resnet_mlp

        # ---Final output---
        # self.adjust = nn.Sequential(nn.Conv2d(2,mid_dim,kernel_size=1),
        #                             nn.Sigmoid(),
        #                             nn.ReLU(),
        #                             nn.Conv2d(mid_dim,2,kernel_size=1))
        self.adjust = nn.Sequential(nn.Conv2d(64, mid_dim, kernel_size=1),
                                    nn.Sigmoid(),
                                    nn.ReLU(),
                                    nn.Conv2d(mid_dim, out_dim, kernel_size=1))

    def _build_attn_block(self, dim_in, attn_depth=1, heads=8):
        '''
        input:
            dim_in: 表示輸入feature的通道數量
            attn_depth:表示block深度
            head: attention head數量
        '''
        layers = []
        for i in range(attn_depth):
            layers.append(AxialAttention(dim=dim_in, heads=heads, dim_index=1))
        return nn.Sequential(*layers)
    def _fit_dims(self, x, in_dim, out_dim):
        '''使用1x1 conv匹配feature'''
        x = x.to('cuda') # 確保輸入都是cuda.tensor
        m = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        if torch.cuda.is_available():
            m.to('cuda')
        return m(x).to('cuda')

    def forward(self,x):
        'input shape: (B,3,H,W), 預設輸入: (4, 3, 128, 128)'
        # ---ResNet feature extraction branch---
        F1 = self.feature_extract_model(x)# 預設(B,128,H//2,W//2), (B,128,64,64)
        F2 = self.F2_extract(F1) # 預設(B,256,H//4,W//4), (B,256,16,16)
        # pdb.set_trace()
        F1 = F.relu(F.interpolate(self.F1_meet_F2(F1), scale_factor=(2, 2), mode='bilinear'))  # (B,256,H//2,W//2)
        F2 = F.relu(F.interpolate(F2, scale_factor=(4, 4), mode='bilinear'))  # (B,256,H//2,W//2)
        if self.resnet_mlp:
            B,_,H,W = F1.shape
            # 使用MLP映射高維度再回來
            F1, F2 = F1.flatten(2), F2.flatten(2)
            F1, F2 = self.linear(F1), self.linear(F2) # (B,256,H,W)
            F1, F2 = F1.resize(B, 512, H, W)
        loss_1 = self.loss_1((F1 + F2)) # (B,C=mid_dim,H//2,W//2) # try it!!
        # F1 = self._fit_dims(F1, 256, 128)

        # ---global branch---
        x_g = self.switch_dim_global_branch(x) # (B,mid_dim=64,H,W)
        x_l1 = x_g.clone() # (B,mid_dim=64,H,W)
        x_l1 = self.atten_global_l1(x_l1) # (B,C,H,W)
        x_l1 = self.downsample_global(x_l1) # (B,C*2,H//2,W//2)
        x_l2 = self.atten_global_l2(x_l1) + x_l1 # residual, # (B,C*2,H//2,W//2)

        x_g += F.relu(F.interpolate(checkpoint(self.global_decoder2, x_l2), scale_factor=(2, 2), mode='bilinear')) # (B,C,H,W)
        # loss_2 = self.loss_2(x_g) # loss_2, (B,2,H,W)
        loss_2 = x_g # loss_2, (B,C,H,W) # try it!!

        # return_features['loss_2'] = loss_2 # loss_2, (B,3,H,W)
        del x_l1,x_l2

        # ---local_ori_branch---
        x_loc_ori = x.clone()
        x_loc_ori = self.switch_dim_global_branch(x_loc_ori)  # (B,mid_dim=64,H,W)
        _, _, H, W = x_loc_ori.shape
        for i in range(0, 4):
            for j in range(0, 4):
                h, w = H // 4, W // 4
                x_p = x_loc_ori[:, :, h * i:h * (i + 1), w * j:w * (j + 1)]
                # ---Transformer Encoder---
                x_ori_l1 = self.attn_local_ori_1(x_p) # (B,C,p_H,p_W)
                x_ori_l2 = self.ori_down1(x_ori_l1)
                x_ori_l2 = self.attn_local_ori_2(x_ori_l2)
                x_ori_l3 = self.ori_down2(x_ori_l2) # if size=128, this patch will be 8x8
                x_ori_l3 = self.attn_local_ori_3(x_ori_l3) + x_ori_l3 # (B, C*4, p_H//4, p_W//4)
                # x_ori_l4 = self.ori_down3(x_ori_l3)
                # x_ori_l4 = self.attn_local_ori_4(x_ori_l4) + x_ori_l4 # residual, # (B, C*8, p_H//8, p_W//8)

                # ---Decoder---
                # x_ori_l3 = torch.add(x_ori_l3, F.relu(F.interpolate(checkpoint(self.ori_local_decoder3, x_ori_l4), scale_factor=(2, 2), mode='bilinear'))) # (B,C*4,p_H//4,p_W//4)
                # x_ori_l3 = torch.add(x_ori_l4, x_ori_l3) # (B,C*4,p_H//4,p_W//4)
                x_ori_l2 = torch.add(x_ori_l2, F.relu(F.interpolate(checkpoint(self.ori_local_decoder2, x_ori_l3), scale_factor=(2, 2), mode='bilinear'))) # (B,C*2,p_H//2,p_W//2)
                x_ori_l1 = torch.add(x_ori_l1, F.relu(F.interpolate(checkpoint(self.ori_local_decoder1, x_ori_l2), scale_factor=(2, 2), mode='bilinear'))) # (B,C,p_H,p_W)
                x_loc_ori[:, :, h * i:h * (i + 1), w * j:w * (j + 1)] = x_ori_l1 # x_local_ori.shape = (B, C, H, W)
                del x_ori_l1, x_ori_l2, x_ori_l3

        # loss_1 = torch.add(x_loc_ori, F1)
        # try let all branch output dim = 64 = mid_dim
        # end_of_ori = self._fit_dims(x_loc_ori, 64, 2) # (B,2,H,W)
        end_of_ori = x_loc_ori # (B,C,H,W) # try it!!

        # ---local_tiny_branch---, for tiny branch use
        x_local_tiny = self.img_to_tiny(x) # size to 1/2, (B,3,H//2,W//2)
        x_loc_t = x_local_tiny.clone()
        x_loc_t = self.switch_dim_global_branch(x_loc_t)  # (B,mid_dim=64,H,W)
        _, _, H, W = x_loc_t.shape
        for i in range(0, 4):
            for j in range(0, 4):
                h, w = H // 4, W // 4
                x_p = x_loc_t[:, :, h * i:h * (i + 1), w * j:w * (j + 1)]

                # ---Transformer Encoder---
                x_tiny_l1 = self.attn_local_tiny_1(x_p)  # (B,C,H,W)
                x_tiny_l2 = self.tiny_down1(x_tiny_l1) # if size=128, this patch will be 8x8
                x_tiny_l2 = self.attn_local_tiny_2(x_tiny_l2) + x_tiny_l2
                # x_tiny_l3 = self.tiny_down2(x_tiny_l2)
                # x_tiny_l3 = self.attn_local_tiny_3(x_tiny_l3)
                # x_tiny_l4 = self.tiny_down3(x_tiny_l3)
                # x_tiny_l4 = self.attn_local_tiny_4(x_tiny_l4) + x_tiny_l4  # residual, # (B,C*8,H//8,W//8)

                # ---Decoder---
                # x_tiny_l3 = torch.add(x_tiny_l3, F.relu(F.interpolate(checkpoint(self.tiny_local_decoder3, x_tiny_l4), scale_factor=(2, 2),
                #                                  mode='bilinear')))  # (B,C*4,H//4,W//4)
                # x_tiny_l2 = torch.add(x_tiny_l2, F.relu(F.interpolate(checkpoint(self.tiny_local_decoder2, x_tiny_l3), scale_factor=(2, 2),
                #                                  mode='bilinear')))  # (B,C*2,H//2,W//2)
                x_tiny_l1 = torch.add(x_tiny_l1, F.relu(F.interpolate(checkpoint(self.tiny_local_decoder1, x_tiny_l2), scale_factor=(2, 2),
                                                 mode='bilinear')) ) # (B,C,H,W)
                x_loc_t[:, :, h * i:h * (i + 1), w * j:w * (j + 1)] = x_tiny_l1  # (B, mid_dim=64, H//2, W//2)
                del x_tiny_l1, x_tiny_l2
        x_loc_t = F.relu(F.interpolate(x_loc_t, scale_factor=(2, 2), mode='bilinear')) # (B, mid_dim=64, H//2, W//2) -> (B,C,H,W)


        # F1: (B,256,H//2,W//2) -> (B,C,H//2,W//2) -> (B,C,H,W)
        F1 = F.relu(F.interpolate(self.F1_adjust_for_loss_3(F1), scale_factor=(2, 2), mode='bilinear'))
        # loss_3 = self.tiny_local_for_loss_3(x_loc_t + F1) # (B,C,H,W) -> (B,2,H,W)
        loss_3 = torch.add(x_loc_t, F1) # (B,C,H,W) # try it!!
        # return_features['loss_3'] = loss_3 # (B, 2, H, W)

        # ---merge local tiny and local ori---
        # end_of_local = torch.add(loss_3, end_of_ori)  # (B, 2, H, W)
        end_of_local = torch.add(loss_3, end_of_ori)  # (B, C, H, W) # try it!!
        del end_of_ori
        # ori_tiny_merge = torch.add(x_loc_t, x_loc_ori) #(B,C,H,W)

        # ---merge global_attention global_resnet---
        # F1: (B,2,H//2,W//2) -> (B,2,H,W)
        # loss_1 = F.relu(F.interpolate(loss_1, scale_factor=(2, 2), mode='bilinear')) # 調整維度
        loss_1 = F.relu(F.interpolate(loss_1, scale_factor=(2, 2), mode='bilinear')) # 調整維度 # try it!!

        # loss2: (2, 2, H, W), loss1: (2, 2, H, W)
        end_of_global = torch.add(loss_2, loss_1) # (2, C, H, W)

        # ---merge local and global---
        final_out = self.adjust(end_of_local + end_of_global)

        # ---feature map dim switch---
        loss_1 = self._fit_dims(loss_1, self.mid_dim, out_dim=self.out_dim)
        loss_2 = self._fit_dims(loss_2, self.mid_dim, out_dim=self.out_dim)
        loss_3 = self._fit_dims(loss_3, self.mid_dim, out_dim=self.out_dim)

        return final_out, loss_1, loss_2, loss_3
        # return final_out # test for tarining

def adv_model_def(args):
    model = advance_seg_transformer(input_dim=3, img_size=args.imgsize, out_dim=args.imgchan, head=8, mid_dim=64, resnet_mlp=False)
    return model


# -------------------unit test--------------------
if __name__ == '__main__':

    adv_model = advance_seg_transformer().to('cuda')
    pred = torch.randn(2, 3, 128, 128).to('cuda')
    mask = torch.randint(0, 2, (2, 1, 128, 128)).float()
    a = adv_model(pred)
    for o in a:
        print(o.shape)

from .blocks import TransFPNModule, BasicBlock
from timm.models.registry import register_model
from torch import nn

# Testing on NEW TransFPN architecture!!
from .TransFPN_Other_Architecture import TransFPNModuleOtherArchitecture
from .Deeplabv3_src_Modified import Deeplabv3Modified

import sys
import torch
import torch.nn.functional as F
import time
import logging


# Test
def build_conv_layer(inplanes, planes, n=1, stride=1, feature_size=128,
                     trans_fpn_module: nn.Module = TransFPNModule):
    """
    :param inplanes: 輸入通道
    :param planes: 輸出通道
    :param n: 設定該層之層數。第一層為標準卷積層(from ResNet34)，n-1以後TransFPNModule
    :param stride: 步長。只用在第一層
    :param feature_size: 輸入特徵尺寸
    :param trans_fpn_module: 使用的TransFPNModule是甚麼

    :return: nn.Sequential
    """
    logging.debug(f"Now transFPN module is {trans_fpn_module.__class__}")
    layer = nn.ModuleList()
    layer.append(BasicBlock(inplanes, planes, stride=stride))
    if n == 1:
        return nn.Sequential(*layer)
    if n > 1:
        for _ in range(n - 1):
            layer.append(trans_fpn_module(planes, planes, feature_size=feature_size))
        return nn.Sequential(*layer)


class Unet(nn.Module):
    def __init__(self, imgsize, imgchan, classes, layers_num: list = None,
                 skip_conn_layer: list = None,
                 print_computing_time=None,
                 ):
        """
        :param imgsize: TODO: Not Use!
        :param imgchan: 輸入通道(default: 3)
        :param classes: 輸出類別數(default: 1)
        :param layers_num: List。各尺度大小的層數設定
        :param skip_conn_layer: List。Unet底層(尺度最小的)層數設定
        :param print_computing_time: 顯示運算時間(for debug)
        """
        super().__init__()
        if layers_num is None:
            layers_num = [1, 2, 3, 4, 1, 1, 1, 1]
            skip_conn_layer = [4, 5]
        C1, C2, C3, C4, C5 = 64, 128, 256, 512, 256
        self.layer1 = nn.Sequential(build_conv_layer(imgchan, C1, n=layers_num[0], feature_size=128),
                                    )
        self.layer2 = nn.Sequential(build_conv_layer(C1, C2, n=layers_num[0], feature_size=128),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    build_conv_layer(C2, C2, n=layers_num[1], feature_size=64),
                                    )
        self.layer3 = nn.Sequential(build_conv_layer(C2, C3, n=layers_num[2], feature_size=64),
                                    nn.MaxPool2d(stride=2, kernel_size=2))
        self.layer4 = nn.Sequential(build_conv_layer(C3, C4, n=layers_num[3], feature_size=32),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        # layer4 橫向CNN堆疊+skip connection
        self.layer4_skip_connections = nn.Sequential(build_conv_layer(C4, C4, n=skip_conn_layer[0], feature_size=16,),
                                                     build_conv_layer(C4, C4, n=skip_conn_layer[1], feature_size=16,))

        self.d_layer4 = nn.Sequential(build_conv_layer(C4, C3, n=layers_num[4], feature_size=16,))
        self.d_layer3 = nn.Sequential(build_conv_layer(C3, C2, n=layers_num[5], feature_size=32,))
        self.d_layer2 = nn.Sequential(build_conv_layer(C2, C1, n=layers_num[6], feature_size=64,))
        self.d_layer1 = nn.Sequential(build_conv_layer(C1, classes, n=layers_num[7], feature_size=128,))

        self.print_computing_time = print_computing_time

    def _forward_implement(self, x):
        start = time.time()
        L1 = self.layer1(x)
        t_l1 = time.time()
        print(t_l1 - start) if self.print_computing_time is not None else None
        L2 = self.layer2(L1)
        t_l2 = time.time()
        print(t_l2 - t_l1) if self.print_computing_time is not None else None
        L3 = self.layer3(L2)
        t_l3 = time.time()
        print(t_l3 - t_l2) if self.print_computing_time is not None else None
        L4 = self.layer4(L3)
        t_l4 = time.time()
        print(t_l4 - t_l3) if self.print_computing_time is not None else None
        L4 = torch.add(self.layer4_skip_connections(L4), L4)
        t_l4_skip_conn = time.time()
        print(t_l4_skip_conn - t_l4) if self.print_computing_time is not None else None

        L3 = L3 + F.interpolate(self.d_layer4(L4), scale_factor=2, mode='bilinear', align_corners=True)
        L2 = L2 + F.interpolate(self.d_layer3(L3), scale_factor=2, mode='bilinear', align_corners=True)
        L2 = self.d_layer2(L2)
        L1 = L1 + F.interpolate(L2, scale_factor=2, mode='bilinear', align_corners=True)
        L1 = self.d_layer1(L1)
        end = time.time()
        print(end - start) if self.print_computing_time is not None else None

        return L1

    def forward(self, x):
        return self._forward_implement(x)


class Unet_ver2_reduce_layer(nn.Module):
    def __init__(self, imgsize, imgchan, classes,
                 layers_num: list = None,
                 skip_conn_layer: list = None,
                 print_computing_time=None,
                 **kwargs):
        """
        :param imgsize: TODO: Not Use!
        :param imgchan: 輸入通道(default: 3)
        :param classes: 輸出類別數(default: 1)
        :param layers_num: 各尺度大小的層數設定
        :param skip_conn_layer: Unet底層(尺度最小的)層數設定
        :param print_computing_time: 顯示運算時間(for debug)
        :param kwargs: 其他設定 TODO: Not Use!
        """
        super().__init__()
        if layers_num is None:
            layers_num = [1, 3, 4, 4, 3, 1]
            skip_conn_layer = [3, 4]
        C1, C2, C3, C4, C5 = 64, 128, 256, 512, 256
        self.layer1 = nn.Sequential(build_conv_layer(imgchan, C1, n=layers_num[0], feature_size=128),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        self.layer2 = nn.Sequential(build_conv_layer(C1, C2, n=layers_num[1], feature_size=64),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        self.layer3 = nn.Sequential(build_conv_layer(C2, C3, n=layers_num[2], feature_size=32),
                                    nn.MaxPool2d(stride=2, kernel_size=2))
        # layer3 橫向CNN堆疊+skip connection
        self.layer3_skip_connections = nn.Sequential(build_conv_layer(C3, C4, n=skip_conn_layer[0], feature_size=16),
                                                     build_conv_layer(C4, C3, n=skip_conn_layer[1], feature_size=16))

        self.d_layer3 = nn.Sequential(build_conv_layer(C3, C2, n=layers_num[3], feature_size=16))
        self.d_layer2 = nn.Sequential(build_conv_layer(C2, C1, n=layers_num[4], feature_size=32))
        self.d_layer1 = nn.Sequential(build_conv_layer(C1, classes, n=layers_num[5], feature_size=64))  # 這層應該設為1不走FPN

        self.print_computing_time = print_computing_time

    def _forward_implement(self, x):
        start = time.time()
        L1 = self.layer1(x)
        t_l1 = time.time()
        print(t_l1 - start) if self.print_computing_time is not None else None
        L2 = self.layer2(L1)
        t_l2 = time.time()
        print(t_l2 - t_l1) if self.print_computing_time is not None else None
        L3 = self.layer3(L2)
        t_l3 = time.time()
        print(t_l3 - t_l2) if self.print_computing_time is not None else None
        L3 = torch.add(self.layer3_skip_connections(L3), L3)
        t_l3_skip_conn = time.time()
        print(t_l3_skip_conn - t_l3) if self.print_computing_time is not None else None

        L2 = L2 + F.interpolate(self.d_layer3(L3), scale_factor=2, mode='bilinear', align_corners=True)  # 128, 32, 32
        d_l2 = time.time()
        print(d_l2 - t_l3_skip_conn) if self.print_computing_time is not None else None
        L1 = L1 + F.interpolate(self.d_layer2(L2), scale_factor=2, mode='bilinear', align_corners=True)  # 64, 64, 64
        d_l1 = time.time()
        print(d_l1 - d_l2) if self.print_computing_time is not None else None
        L1 = F.interpolate(self.d_layer1(L1), scale_factor=2, mode='bilinear', align_corners=True)  # 3, 128, 128
        end = time.time()
        print(end - d_l1) if self.print_computing_time is not None else None
        print(end - start) if self.print_computing_time is not None else None

        return L1

    def forward(self, x):
        return self._forward_implement(x)


class UnetReduceLayerMobilevitv2Module(nn.Module):
    """使用Mobilevitv2Module的Reduce-layer-Unet-model"""
    def __init__(self, imgsize, imgchan, classes,
                 layers_num: list = None,
                 skip_conn_layer: list = None,
                 print_computing_time=None,
                 trans_fpn_module=TransFPNModuleOtherArchitecture):
        """
        :param imgsize: TODO: Not Use!
        :param imgchan: 輸入通道(default: 3)
        :param classes: 輸出類別數(default: 1)
        :param layers_num: 各尺度大小的層數設定
        :param skip_conn_layer: Unet底層(尺度最小的)層數設定
        :param print_computing_time: 顯示運算時間(for debug)
        :param trans_fpn_module: 使用的模組
        """
        super().__init__()
        if layers_num is None:
            layers_num = [1, 3, 4, 4, 3, 1]
            skip_conn_layer = [3, 4]
        C1, C2, C3, C4, C5 = 64, 128, 256, 512, 256
        # imgchan = 64
        tfm = trans_fpn_module
        self.layer1 = nn.Sequential(build_conv_layer(imgchan, C1, n=layers_num[0], feature_size=128, trans_fpn_module=tfm),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        self.layer2 = nn.Sequential(build_conv_layer(C1, C2, n=layers_num[1], feature_size=64, trans_fpn_module=tfm),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        self.layer3 = nn.Sequential(build_conv_layer(C2, C3, n=layers_num[2], feature_size=32, trans_fpn_module=tfm),
                                    nn.MaxPool2d(stride=2, kernel_size=2))
        # layer3 橫向CNN堆疊+skip connection
        self.layer3_skip_connections = nn.Sequential(build_conv_layer(C3, C3, n=skip_conn_layer[0], feature_size=16, trans_fpn_module=tfm),
                                                     build_conv_layer(C3, C3, n=skip_conn_layer[1], feature_size=16, trans_fpn_module=tfm))

        self.d_layer3 = nn.Sequential(build_conv_layer(C3, C2, n=layers_num[3], feature_size=16, trans_fpn_module=tfm))
        self.d_layer2 = nn.Sequential(build_conv_layer(C2, C1, n=layers_num[4], feature_size=32, trans_fpn_module=tfm))
        self.d_layer1 = nn.Sequential(build_conv_layer(C1, classes, n=layers_num[5], feature_size=64, trans_fpn_module=tfm))

        self.print_computing_time = print_computing_time

    def _forward_implement(self, x):
        start = time.time()
        L1 = self.layer1(x)
        t_l1 = time.time()
        print(t_l1 - start) if self.print_computing_time is not None else None
        L2 = self.layer2(L1)
        t_l2 = time.time()
        print(t_l2 - t_l1) if self.print_computing_time is not None else None
        L3 = self.layer3(L2)
        t_l3 = time.time()
        print(t_l3 - t_l2) if self.print_computing_time is not None else None
        L3 = torch.add(self.layer3_skip_connections(L3), L3)
        t_l3_skip_conn = time.time()
        print(t_l3_skip_conn - t_l3) if self.print_computing_time is not None else None

        L2 = L2 + F.interpolate(self.d_layer3(L3), scale_factor=2, mode='bilinear', align_corners=True)  # 128, 32, 32
        d_l2 = time.time()
        print(d_l2 - t_l3_skip_conn) if self.print_computing_time is not None else None
        L1 = L1 + F.interpolate(self.d_layer2(L2), scale_factor=2, mode='bilinear', align_corners=True)  # 64, 64, 64
        d_l1 = time.time()
        print(d_l1 - d_l2) if self.print_computing_time is not None else None
        L1 = F.interpolate(self.d_layer1(L1), scale_factor=2, mode='bilinear', align_corners=True)  # 3, 128, 128
        end = time.time()
        print(end - d_l1) if self.print_computing_time is not None else None
        print(end - start) if self.print_computing_time is not None else None
        return L1

    def forward(self, x):
        return self._forward_implement(x)


def _reset_parameter(self):
    for m in self.modules():
        # print(m)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


@register_model
def TransFPN_Module_Unet_VANNILA(pretrained=False, **kwargs):
    # 2022/5/16測試之模型
    model = Unet(128,
                 3,
                 1,
                 layers_num=[1, 2, 3, 4, 1, 1, 1, 1],
                 skip_conn_layer=[5, 4],  # [4, 5]
                 print_computing_time=None
                 )
    model.apply(_reset_parameter)
    return model


@register_model
def TransFPN_Module_Unet_M2(pretrained=False, **kwargs):
    # 2022/5/18測試之模型。ResAttnModule_reduce_layer_M2
    model = Unet_ver2_reduce_layer(128,
                                   3,
                                   1,
                                   layers_num=[1, 3, 4, 4, 3, 1],
                                   skip_conn_layer=[5, 4],
                                   print_computing_time=None
                                   )
    model.apply(_reset_parameter)
    return model


@register_model
def TransFPN_Module_Unet_S(pretrained=False, **kwargs):
    model = Unet_ver2_reduce_layer(128,
                                   3,
                                   1,
                                   layers_num=[1, 2, 3, 3, 2, 1],
                                   skip_conn_layer=[2, 3],
                                   print_computing_time=None
                                   )
    model.apply(_reset_parameter)
    return model


@register_model
def TransFPN_Module_Unet_M(pretrained=False, **kwargs):
    model = Unet_ver2_reduce_layer(128,
                                   3,
                                   1,
                                   layers_num=[1, 3, 4, 4, 3, 1],
                                   skip_conn_layer=[4, 5],
                                   print_computing_time=None,
                                   **kwargs
                                   )
    model.apply(_reset_parameter)
    return model


@register_model
def TransFPN_Module_Unet_L(pretrained=False, **kwargs):
    model = Unet(128,
                 3,
                 1,
                 layers_num=[2, 2, 3, 4, 4, 3, 2, 1],
                 skip_conn_layer=[3, 3],  # [4, 5]
                 print_computing_time=None
                 )
    model.apply(_reset_parameter)
    return model


@register_model
def TransFPN_Module_Unet_TTTTEEEESSSSTTTT(pretrained=False, **kwargs):
    # 2022/5/24、2022/5/25、2022/5/26測試之模型。
    # 2022/5/24。skip_conn_layer=[1, 1]。每一個block/layer都是1層(也就是不包含transFPN模組)
    # 2022/5/25。skip_conn_layer=[2, 2]
    # 2022/5/26。skip_conn_layer=[3, 3]

    model = Unet(128,
                 3,
                 1,
                 layers_num=[1, 1, 1, 1, 1, 1, 1, 1],
                 skip_conn_layer=[3, 3],
                 print_computing_time=None
                 )
    model.apply(_reset_parameter)
    return model


# <TEST> TransFPN_Module_Unet_reduce_layer (only transFPN module)
@register_model
def TransFPN_Module_Unet_XS(pretrained=False, **kwargs):
    model = Unet_ver2_reduce_layer(128,
                                   3,
                                   1,
                                   layers_num=[1, 1, 1, 1, 1, 1],
                                   skip_conn_layer=[1, 1],
                                   print_computing_time=None
                                   )
    model.apply(_reset_parameter)
    return model


# <TEST> TransFPN_Module_Unet_reduce_layer (Mobilevitv2Module)
@register_model
def Mobilevitv2Module_Unet(pretrained=False, **kwargs):
    model = UnetReduceLayerMobilevitv2Module(128,
                                             3,
                                             1,
                                             layers_num=[1, 1, 1, 1, 1, 1],
                                             skip_conn_layer=[2, 2],
                                             print_computing_time=None
                                             )
    model.apply(_reset_parameter)
    return model

# 更改的DeeplabV3+架構。DeeplabV3+之head部分用attention替換
@register_model
def Deeplabv3_Modified(pretrained=False, **kwargs):
    model = Deeplabv3Modified(in_channels=3, classes=1).cuda()
    model.apply(_reset_parameter)
    return model


if __name__ == '__main__':
    import argparse
    import timm

    parser = argparse.ArgumentParser(description='model run TEST')
    parser.add_argument('--imgsize', default=128)
    parser.add_argument('--imgchan', default=3)
    parser.add_argument('--classes', default=1)
    args = parser.parse_args()

    test_model = timm.create_model('Mobilevitv2Module_Unet')
    # model = only_transFPN_module_Unet(args)
    # for n,m in model.named_modules():
    #     print(n, '\t', m)
    x_test = torch.randn(8, 3, 128, 128)
    out = test_model(x_test)
    total_params = sum(p.numel() for p in test_model.parameters())
    print('{} parameter：{:8f}M'.format(test_model.__class__.__name__, total_params / 1000000))  # 確認模型參數數量

    if type(out) == list:
        for f in out:
            print(f.shape)
            # ResNet34 encoder output
            # torch.Size([1, 3, 128, 128])
            # torch.Size([1, 64, 64, 64])
            # torch.Size([1, 64, 32, 32])
            # torch.Size([1, 128, 16, 16])
            # torch.Size([1, 256, 16, 16])
            # torch.Size([1, 512, 16, 16])
    else:
        print(out.shape, end='\n\n\n\n\n\n')

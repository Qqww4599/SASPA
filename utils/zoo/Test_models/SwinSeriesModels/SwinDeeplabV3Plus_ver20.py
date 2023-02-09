import pdb
import sys
from typing import (Optional, Tuple)

import torch
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders import deeplabv3
from segmentation_models_pytorch.decoders.deeplabv3.decoder import ASPPConv, ASPPSeparableConv, ASPPPooling
from segmentation_models_pytorch.encoders import get_encoder
from monai.networks.nets.swin_unetr import BasicLayer

from torch import nn
from torch.nn import functional as F

__all__ = ['swindeeplabv3plus_ver20']


class swindeeplabv3plus_ver20(SegmentationModel):
    """DeepLabV3-based Swin model ver2.0, from smp.decoder.deeplabv3 """

    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_channels: int = 256,
                 encoder_output_stride: int = 16,  # 用8也可以run
                 in_channels: int = 3,
                 classes: int = 1,
                 upsampling: int = 4,
                 swinblock: bool = True
                 ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        self.decoder = Swin_deeplabv3plusdecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=(12, 24, 36),
            output_stride=encoder_output_stride,
            swinblock=swinblock
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=None,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.classification_head = None


class Swin_deeplabv3plusdecoder(nn.Sequential):
    """
    Swin-deeplabv3 Modefied ver1.1
    modified deeplabv3+ with Swin block

    ver1.2: ASPP.swin_feature_fusion刪除Basiclayer後的conv

    Ideas:
    使用的encoder特徵圖不只特定二層(default: encoder_output[-1] for ASPP, encoder_output[-4] for block1)
    """
    def __init__(
            self,
            encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
            swinblock: bool=False
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True, swinblock=swinblock),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        # block REMAKE
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-3:])  # (128, 16, 16), (256, 8, 8), (512, 8, 8)
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])  # 64, 32, 32
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features


class ASPP(nn.Module):
    """
    經過Swin修改後。
    1. rate1, rate2加入SwinBlock
    ver1.3: input=(128, 16, 16), (256, 8, 8), (512, 8, 8)
    """
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False, swinblock=False):
        super(ASPP, self).__init__()
        modules = []
        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        ASPPConvModule_rate1 = nn.Sequential(ASPPConvModule(128, out_channels, rate1),
                                             nn.AvgPool2d(kernel_size=2)
                                             )
        self.Swinblock = None
        if swinblock:
            SWIN_IN_CHANNEL = 128
            self.Swinblock = nn.Sequential(
                BasicLayer(dim=SWIN_IN_CHANNEL, depth=3, num_heads=8, window_size=(7, 7),drop_path=0.2, attn_drop=0.1),
                nn.Conv2d(SWIN_IN_CHANNEL, out_channels, kernel_size=3, padding=1, stride=1),
                nn.AvgPool2d(kernel_size=2)
            )
            modules.append(nn.Sequential(ASPPConvModule_rate1))
            modules.append(nn.Sequential(ASPPConvModule(256, out_channels, rate3)))
            modules.append(nn.Sequential(ASPPPooling(512, out_channels)))
        else:
            modules.append(ASPPConvModule_rate1)
            # modules.append(ASPPConvModule(in_channels, out_channels, rate2))
            modules.append(ASPPConvModule(256, out_channels, rate3))
            modules.append(ASPPPooling(512, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for i, (conv, feature) in enumerate(zip(self.convs, x)):
            if i == 0 and self.Swinblock is not None:  # Only multiply Swinblock in ASPPConvModule_rate1 branch
                res.append(self.Swinblock(feature) * conv(feature))
                continue
            res.append(conv(feature))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


def main():
    from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3Plus
    # parameter
    atrous_rates = (12, 24, 36)
    in_channels = 8,
    out_channels = 32,

    testdata = torch.randn(2, 3, 128, 128)
    # Deeplabv3Plus = DeepLabV3Plus()
    # print(Deeplabv3Plus(testdata).shape)

    model = swindeeplabv3plus_ver20(swinblock=True)
    print(model(testdata).shape)
    # total_params = sum(p.numel() for p in model.parameters())
    # print('{} parameter：{:8f}M'.format(model, total_params / 1000000))  # 確認模型參數數量
    # print(output.shape)


if __name__ == '__main__':
    main()
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel
from torch import nn
from .TransFPN_Other_Architecture import TransFPNModuleOtherArchitecture

import sys
import segmentation_models_pytorch as timm
import torch
import time


class Deeplabv3Modified(SegmentationModel):
    """
    更改的DeeplabV3+架構。DeeplabV3+之head部分用attention替換
    """
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights="imagenet",
                 encoder_output_stride: int = 16,
                 decoder_channels: int = 256,
                 decoder_atrous_rates: tuple = (12, 24, 36),
                 in_channels: int = 3,
                 classes: int = 1,
                 activation=None,
                 upsampling: int = 4,
                 aux_params=None,
                 ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        self.classification_head = None


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        # conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # activation = Activation(activation)

        # mid_dim: 設定最終分類層的內部通道數
        mid_dim = 64

        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        FPN = FpnLayers(in_channels, mid_dim)
        conv2d = nn.Conv2d(mid_dim, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = Activation(activation)

        super().__init__(FPN, upsampling, conv2d, activation)


class FpnLayers(nn.Module):
    def __init__(self, in_channels, layersOfFFPN=2, mid_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_dim, kernel_size=3, stride=1, padding=1)
        self.fpn_layers = nn.ModuleList()
        for l in range(layersOfFFPN):
            self.fpn_layers.append(TransFPNModuleOtherArchitecture(mid_dim, mid_dim))

    def forward(self, x):
        x = self.conv(x)
        for l in self.fpn_layers:
            x = l(x)
        return x


def foo():
    # 訓練模型batch_size至少要大於1。 至少4個dim(B,C,H,W)。
    testData = torch.randn(2, 3, 128, 128)
    deeplabEncoderModule = timm.DeepLabV3Plus().encoder
    deeplabDecoderModule = timm.DeepLabV3Plus().decoder
    outputs = deeplabEncoderModule(testData)
    for i, f in enumerate(outputs):
        print(f'No {i} feature is {f.shape}')
    outputs = deeplabDecoderModule(*outputs)
    print(outputs.shape)


if __name__ == '__main__':
    foo()
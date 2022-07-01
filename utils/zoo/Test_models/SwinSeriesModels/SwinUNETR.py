from monai.networks.nets import SwinUNETR
from timm.models.registry import register_model
from . import SwinDeeplabV3plus

from typing import (List, Sequence)
import segmentation_models_pytorch as smp
import timm
import monai


@register_model
def SwinUNETR_Original():
    model = SwinUNETR(img_size=(128, 128), in_channels=3, out_channels=1, feature_size=48, spatial_dims=2)
    return model


@register_model
def SwinUnet():
    model = smp.UnetPlusPlus()
    return model


@register_model
def Swin_deeplabv3():
    model = smp.DeepLabV3Plus(encoder_name='resnet34',)
    return model


def main():
    model = smp.create_model('deeplabv3')
    print(model)


if __name__ == '__main__':
    main()

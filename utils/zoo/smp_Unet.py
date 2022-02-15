import segmentation_models_pytorch as smp
import torch


def create_model():
    model2 = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2)
    model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2)
    return model

if __name__ == '__main__':
    model = create_model()
    TOTAL_PARAMETER = sum(p.numel() for p in model.parameters())
    print(TOTAL_PARAMETER)
    input_data = torch.randn(1,3,224,224)
    out = model(input_data)
    print('model output shape: ',out.shape)
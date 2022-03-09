import argparse
import sys
import torch
import timm
import pdb
import segmentation_models_pytorch as smp
import torch.optim.lr_scheduler as scheduler

__name__ = []

def Use_model(args):
    '''
    Model參數設置
    '''
    model_name = str(args.modelname)
    if model_name == 'VisionTransformer':
        from .zoo.vision_transformer import VisionTransformer
        model = VisionTransformer()  # 測試asyml的vision transformer code

    if model_name == 'medt':
        from .zoo.MedT.lib.models.axialnet import MedT
        model = MedT(args)

    if model_name == 'gated':
        from .zoo.MedT.lib.models.axialnet import gated
        model = gated(args)

    if model_name == 'U_net':
        from .zoo import U_net
        model = U_net.U_Net(img_ch=3, output_ch=3)

    if model_name == 'logo':
        from .zoo.MedT.lib.models.axialnet import logo
        model = logo(args)
    if model_name == 'timm':
        '暫時使用vit_base_patch16_384'
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.imgsize ** 2)
        # model.reset_classifier()
    if model_name == 'unet_resnet34':
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2)
    if model_name == 'unet++_resnet34':
        model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2)
    if model_name == 'medt_retrofit':
        from .zoo.medt_retrofit import medt_retrofit_model
        model = medt_retrofit_model(args)



    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print('{} parameter：{:8f}M'.format(model_name,total_params / 1000000))  # 確認模型參數數量
    return model
def use_opt(args, model):
    opt = args.optimizer
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

def use_scheduler(args, opt):
    scheduler_name = str(args.scheduler)
    if scheduler_name == 'ReduceLROnPlateau':
        return scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5, patience=10, verbose=True)
    if scheduler_name == 'CosineAnnealingLR':
        return scheduler.CosineAnnealingLR(optimizer=opt, T_max=10) # T_max是週期的1/2




# Use_model 單元測試
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelname', type=str, default='timm')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-is', '--imgsize', type=int, default=256, help='圖片大小')
    args = parser.parse_args()

    m = Use_model(args)
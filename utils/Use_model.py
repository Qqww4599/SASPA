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
    print(model_name)
    if model_name == 'medt':
        from .zoo.MedT.lib.models.axialnet import MedT
        model = MedT(args)
    elif model_name == 'gated':
        from .zoo.MedT.lib.models.axialnet import gated
        model = gated(args)
    elif model_name == 'U_net':
        from .zoo import U_net
        model = U_net.U_Net(img_ch=3, output_ch=3)
    elif model_name == 'logo':
        from .zoo.MedT.lib.models.axialnet import logo
        model = logo(args)
    elif model_name == 'timm':
        '暫時使用vit_base_patch16_384'
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=args.imgsize ** 2)
    elif model_name == 'unet_resnet34':
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif model_name == 'unet++_resnet18':
        model = smp.UnetPlusPlus(encoder_name='resnet18', encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif model_name == 'unet++_resnet34':
        model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif model_name == 'unet++_resnet50':
        model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif model_name == 'unet++_resnet101':
        model = smp.UnetPlusPlus(encoder_name='resnet101', encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif model_name == 'medt_retrofit':
        from .zoo.medt_retrofit import medt_retrofit_model_use
        model = medt_retrofit_model_use(args)
    elif model_name == 'pranet':
        from .zoo.reverse_attn_Unet import Reverse_attn_unet
        print('this is pranet')
        model = Reverse_attn_unet()
    # -----------------2022/3/16, 2022/3/17測試使用之model--------------
    elif model_name == 'multi_loss':
        from ..utils.zoo.Test_models.multi_loss_return_medt.multi_loss_return_medt import medt_retrofit_model_use
        model = medt_retrofit_model_use(args)
    elif model_name == 'without_attn':
        from .zoo.Test_models.medt_global_wo_attn.medt_global_wo_attn import medt_retrofit_model_use
        model = medt_retrofit_model_use(args)
    # -------------------------測試用-------------------------
    elif model_name == '20220422 test1':
        from .zoo.medt_conv_map import medt_retrofit_model_use
        model = medt_retrofit_model_use(args)
    elif model_name == '20220423 test1':
        from .zoo._20220423test1 import medt_retrofit_model_use
        model = medt_retrofit_model_use(args)

    # --------------------測試新model使用------------------------
    elif model_name == 'TEST':
        # from model.utils.zoo.Test_models.global_branch_ver1_1.model import medt
        from .zoo.Test_models.global_branch_ver1_1.model import medt
        model = medt(args)
    else:
        raise ValueError(f'Should enter exist model name!! Now put model name is {model_name}!!!')


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

def use_loss_fn(args):
    # 選擇使用的loss function。
    # 經過測試可以使用的(在MedT環境中)：weight_cross_entropy, dice_coef_loss,IoU,FocalLoss
    # validation階段調用
    sys.path.append('.')
    from . import loss_fn
    if args.loss_fn == 'wce':
        # wce 可以接受channel=1的output
        loss_fn_name = 'wce'
        loss = loss_fn.weight_cross_entropy
    elif args.loss_fn == 'bce':
        # wce 可以接受channel=1的output
        loss_fn_name = 'bce'
        loss = loss_fn.binary_cross_entropy
    elif args.loss_fn == 'dice_coef_loss':
        loss_fn_name = 'dice_coef_loss'
        loss = loss_fn.dice_coef_loss
    elif args.loss_fn == 'IoU':
        loss_fn_name = 'IoU'
        loss = loss_fn.IoU
    elif args.loss_fn == 'FocalLoss':  # 這個criterion是用torch.nn.module建立，需要當作layer看待
        loss_fn_name = 'FocalLoss'
        loss = loss_fn.FocalLoss()
    elif args.loss_fn == 'lll':
        loss_fn_name = 'lll'
        loss = loss_fn.LogNLLLoss()
    elif args.loss_fn == 'diceloss':
        loss_fn_name = 'diceloss'
        loss = loss_fn.dice_loss
    elif args.loss_fn == 'clsiou':
        loss_fn_name = 'clsiou'
        loss = loss_fn.classwise_iou
    else:
        raise ValueError(f'Please choose a loss function, input is {args.loss_fn}')
    # print('----- loss_fn_name: ',loss_fn_name, '-----')
    return loss


# Use_model 單元測試
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelname', type=str, default='timm')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-is', '--imgsize', type=int, default=256, help='圖片大小')
    args = parser.parse_args()

    m = Use_model(args)
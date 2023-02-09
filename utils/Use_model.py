import torch
import segmentation_models_pytorch as smp
import torch.optim.lr_scheduler as scheduler
import argparse
import sys
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo')
from . import loss_fn_adv

# __name__ = []


def use_opt(args, model):
    opt = args.optimizer
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def use_model(args):
    """Model參數設置"""

    model_name = str(args.modelname)

    if model_name == 'medt':
        from .zoo.MedT.lib.models.axialnet import MedT
        model = MedT(args)
    # -------------------------過去model測試(已封存)-------------------------
    # elif model_name == '20220422 test1':
    #     from .zoo.MedT_Global_ResNet0422 import medt_retrofit_model_use
    #     model = medt_retrofit_model_use(args)
    # elif model_name == '20220423 test1':
    #     from .zoo.MedT_Global_ResNet0423 import medt_retrofit_model_use
    #     model = medt_retrofit_model_use(args)
    # elif 'TransCycle' in model_name:
    #     from .zoo.Test_models.TransCycle_model import get_model
    #     model = get_model(args)
    # elif model_name == 'gated':
    #     from .zoo.MedT.lib.models.axialnet import gated
    #     model = gated(args)
    # elif model_name == 'axialtransunet':
    #     from .zoo.MedT.lib.models.axialnet import axialunet
    #     model = axialunet(args)
    # elif model_name == 'logo':
    #     from .zoo.MedT.lib.models.axialnet import logo
    #     model = logo(args)
    # -------------------------過去model測試(已封存)-------------------------
    elif model_name == 'unet_resnet34':
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=args.classes)
    elif model_name == 'unet++_resnet18':
        model = smp.UnetPlusPlus(encoder_name='resnet18',
                                 encoder_weights='imagenet',
                                 in_channels=3,
                                 classes=args.classes)
    elif model_name == 'unet++_resnet34':
        model = smp.UnetPlusPlus(encoder_name='resnet34',
                                 encoder_weights='imagenet',
                                 in_channels=3,
                                 classes=args.classes)
    elif model_name == 'unet++_resnet50':
        model = smp.UnetPlusPlus(encoder_name='resnet50',
                                 encoder_weights='imagenet',
                                 in_channels=3,
                                 classes=args.classes)
    elif model_name == 'unet++_resnet101':
        model = smp.UnetPlusPlus(encoder_name='resnet101',
                                 encoder_weights='imagenet',
                                 in_channels=3,
                                 classes=args.classes)
    elif model_name == 'medt_retrofit':
        from .zoo.___Deprecated___Medt_retrofit import medt_retrofit_model_use
        model = medt_retrofit_model_use(args)
    elif model_name == 'pranet':
        from utils.zoo.Test_models.Reverse_AttentionUNet.reverse_attn_Unet import Reverse_attn_unet
        model = Reverse_attn_unet()
    elif model_name == 'segformer':
        from utils.zoo.COMPARE_MODEL.scformer.models import build
        model = build(model_name='segformer', class_num=args.classes)
    elif model_name == 'FPN_resnet34':
        model = smp.FPN(encoder_name='resnet34', encoder_weights=None, classes=args.classes)
    elif model_name == 'FPN_resnet18':
        model = smp.FPN(encoder_name='resnet18', encoder_weights=None, classes=args.classes)
    elif model_name == 'FPN_resnet50':
        model = smp.FPN(encoder_name='resnet50', encoder_weights=None, classes=args.classes)
    elif 'ResAttnModule' in model_name or 'TransFPN_Module' in model_name:
        from .zoo.Test_models.TransFPNSeries import get_model
        model = get_model(args)
    elif model_name == 'SwinUNETR_from_monai':
        from monai.networks.nets import SwinUNETR
        # model = ViT(in_channels=3, img_size=(128, 128), num_classes=1, hidden_size=384,
        #             mlp_dim=1024, classification=False, patch_size=16, spatial_dims=2)
        model = SwinUNETR(img_size=(128, 128), in_channels=3, out_channels=1, feature_size=48, spatial_dims=2)
    elif model_name == 'deeplabv3+':
        from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3Plus
        model = DeepLabV3Plus()
    elif model_name == 'SwinUnet':
        from utils.zoo.COMPARE_MODEL.Swin_Unet.networks.swin_transformer_unet_skip_expand_decoder_sys import \
            SwinTransformerSys
        model = SwinTransformerSys(img_size=128, patch_size=16, window_size=12, num_classes=1)
    elif model_name == 'SwinDeeplabv3Plus':
        from utils.zoo.Test_models.SwinSeriesModels.SwinDeeplabV3Plus_ver13 import swindeeplabv3plus_ver13
        model = swindeeplabv3plus_ver13()
    # --------------------測試新model使用------------------------
    elif '<test>' in model_name:
        """
        :keyword <test>是測試項目的特別標記。正式項目訓練需要拔掉
        
        測試項目: swindeeplabv3plus_ver20
        """
        from utils.zoo.Test_models.SwinSeriesModels.SwinDeeplabV3Plus_ver20 import swindeeplabv3plus_ver20
        model = swindeeplabv3plus_ver20(swinblock=True)
    else:
        raise ValueError(f'Should enter exist model name!! Now put model name is {model_name}!!!')
    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print('{} parameter：{:8f}M'.format(model_name, total_params / 1000000))  # 確認模型參數數量
    # macs, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # sys.exit()
    return model


def use_scheduler(args, opt):
    scheduler_name = str(args.scheduler)
    if scheduler_name == 'ReduceLROnPlateau':
        return scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5, patience=10, verbose=False)
    if scheduler_name == 'CosineAnnealingLR':
        return scheduler.CosineAnnealingLR(optimizer=opt, T_max=10)  # T_max是週期的1/2


def use_loss_fn(args):
    # 選擇使用的loss function。
    # 經過測試可以使用的(在MedT環境中)：weight_cross_entropy, dice_coef_loss,IoU,FocalLoss
    # validation階段調用
    sys.path.append('.')
    from . import loss_fn
    if args.loss_fn == 'wce':
        # wce 可以接受channel=1的output
        loss = loss_fn.weight_cross_entropy
    elif args.loss_fn == 'bce':
        # wce 可以接受channel=1的output
        loss = loss_fn.binary_cross_entropy
    elif args.loss_fn == 'ce' or args.loss_fn == 'CE':
        # wce 可以接受channel=1的output
        loss = torch.nn.functional.cross_entropy
    elif args.loss_fn == 'dice_coef_loss':
        loss = loss_fn.dice_coef_loss
    elif args.loss_fn == 'IoU':
        loss = loss_fn.IoU
    elif args.loss_fn == 'FocalLoss':  # 這個criterion是用torch.nn.module建立，需要當作layer看待
        loss = loss_fn.FocalLoss()
    elif args.loss_fn == 'lll' or args.loss_fn == 'LogNLLLoss':
        loss = loss_fn.LogNLLLoss()
    elif args.loss_fn == 'diceloss':
        loss = loss_fn.dice_loss
    elif args.loss_fn == 'clsiou':
        loss = loss_fn.classwise_iou
    elif args.loss_fn == 'NLLLoss':
        loss = loss_fn.NLLLoss
    elif args.loss_fn == 'BCEDicePenalizeBorderLoss':
        loss = loss_fn_adv.BCEDicePenalizeBorderLoss()
    elif args.loss_fn == 'BCEDiceFocalLoss':
        loss = loss_fn_adv.BCEDiceFocalLoss(focal_param=0.5)
    else:
        raise ValueError(f'Please choose a loss function, input is {args.loss_fn}')
    # print('----- loss_fn_name: ',loss_fn_name, '-----')
    return loss


# Use_model 單元測試
if __name__ == '__main__':
    testdata = torch.randn(2,3,128,128).to('cuda:0')

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-modelname', type=str, default='segformer')
        parser.add_argument('-device', type=str, default='cuda')
        parser.add_argument('-is', '--imgsize', type=int, default=128, help='圖片大小')
        parser.add_argument('-ic', '--imgchan', type=int, default=3, help='圖片通道')
        parser.add_argument('--classes', type=int, default=1, help='')
        parser.add_argument('--training_details', type=str, default='', help='訓練項目細節。training details')
        args = parser.parse_args()
        return args
    m = use_model(parse_args())
    print(m(testdata).shape)

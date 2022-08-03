import timm
import sys
import logging

from MainResearch.utils.zoo.Test_models.TransFPNSeries.model import *

sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\ResNet_other_architechture\model.py')
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\ResNet_other_architechture')


Modelname = ['ResAttnModule Unet No double axial attntion', 'ResAttnModule Unet_ver2_reduce_layer']

"""
這個模組就用來測試Unet架構下的TransFPN_Module。

主要架構包含:
 4 layers的 Unet
 3 layers的 Unet_ver2_reduce_layer

"""


def get_model(args):
    # if args.modelname not in Modelname:
    #     raise ValueError('Should use exist ResAttnModule series model!!')
    print_computing_time = None
    if args.modelname == 'TransFPN_Module_Unet_VANNILA':
        print("模型基礎結構，名稱：TransFPN_Module_Unet_VANNILA"
              "這個model是忘記加入雙向axial attention(只有單向)")
        model = timm.create_model(args.modelname)
    elif args.modelname == 'ResAttnModule_reduce_layer_M2' or args.modelname == 'TransFPN_Module_Unet_M2':
        print("2022/5/18測試之模型，結構類似ResAttnModule_reduce_layer_M，Skip-connection層數前後調換。"
              "ResAttnModule_reduce_layer_M2")
        model = timm.create_model(args.modelname)
    elif args.modelname == 'ResAttnModule_reduce_layer_S' or args.modelname == 'TransFPN_Module_Unet_S':
        # print("ResAttnModule_reduce_layer_S")
        model = timm.create_model(args.modelname)
    elif args.modelname == 'ResAttnModule_reduce_layer_M' or args.modelname == 'TransFPN_Module_Unet_M':
        # print("ResAttnModule_reduce_layer_M")
        model = timm.create_model(args.modelname)
    elif args.modelname == 'TransFPN_Module_Unet_L':
        # print("ResAttnModule_reduce_layer_L")
        model = timm.create_model(args.modelname)
    elif args.modelname == 'TransFPN_Module_Unet_TTTTEEEESSSSTTTT':
        model = timm.create_model(args.modelname)
    elif args.modelname in ['TransFPN_Module_Unet_reduce_layer (only transFPN module)', 'only_TransFPN_Module_Unet',
                            "TransFPN_Module_Unet_XS"]:
        model = timm.create_model('TransFPN_Module_Unet_XS')
    elif args.modelname == 'Mobilevitv2Module_Unet':
        model = timm.create_model(args.modelname)

    else:
        raise NotImplementedError('Not Implemented!!')

    return model


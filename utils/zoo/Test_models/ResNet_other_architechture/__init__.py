import sys
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\ResNet_other_architechture\model.py')
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\ResNet_other_architechture')
from model import *
import argparse

Modelname = ['ResAttnModule Unet No double axial attntion', 'ResAttnModule Unet_ver2_reduce_layer']

def get_model(args):
    # if args.modelname not in Modelname:
    #     raise ValueError('Should use exist ResAttnModule series model!!')
    print_computing_time = None
    if args.modelname == 'TransFPN_Module_Unet_VANNILA':
        print("模型基礎結構，名稱：TransFPN_Module_Unet_VANNILA"
              "這個model是忘記加入雙向axial attention(只有單向)")
        model = TransFPN_Module_Unet_VANNILA(args)
    elif args.modelname == 'ResAttnModule_reduce_layer_M2' or args.modelname == 'TransFPN_Module_Unet_M2':
        print("2022/5/18測試之模型，結構類似ResAttnModule_reduce_layer_M，Skip-connection層數前後調換。"
              "ResAttnModule_reduce_layer_M2")
        model = TransFPN_Module_Unet_M2(args)
    elif args.modelname == 'ResAttnModule_reduce_layer_S' or args.modelname == 'TransFPN_Module_Unet_S':
        # print("ResAttnModule_reduce_layer_S")
        model = TransFPN_Module_Unet_S(args)
    elif args.modelname == 'ResAttnModule_reduce_layer_M' or args.modelname == 'TransFPN_Module_Unet_M':
        # print("ResAttnModule_reduce_layer_M")
        model = TransFPN_Module_Unet_M(args)
    elif args.modelname == 'TransFPN_Module_Unet_L':
        # print("ResAttnModule_reduce_layer_M")
        model = TransFPN_Module_Unet_L(args)
    else:
        raise NotImplementedError('Not Implemented!!')

    return model


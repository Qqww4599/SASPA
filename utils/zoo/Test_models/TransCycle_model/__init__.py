import sys
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\TransCycle_model\axial_attention_module')
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\TransCycle_model')
import TransCycle_model_30
import TransCycle_model_20
import TransCycle_model_10

import argparse

Modelname = ['transcycle30 Unet', 'TransCycle3.0', 'TransCycle2.0','TransCycle1.0',]

def get_model(args):
    if args.modelname not in Modelname:
        raise ValueError('Should use exist TransCycle series model!!')
    elif args.modelname == 'transcycle30 Unet':
        model = TransCycle_model_30.Unet(args.imgsize, args.imgchan, args.classes)
        model.apply(TransCycle_model_30._reset_parameter)
    elif args.modelname == 'TransCycle3.0':
        model = TransCycle_model_30.TransCycle(args.imgsize, args.imgchan, args.classes)
        model.apply(TransCycle_model_30._reset_parameter)
    elif args.modelname == 'TransCycle2.0':
        model = TransCycle_model_20.TransCycle(args.imgsize, args.imgchan, args.classes)
        model.apply(TransCycle_model_30._reset_parameter)
    elif args.modelname == 'TransCycle1.0':
        model = TransCycle_model_10.TransCycle(args.imgsize, args.imgchan, args.classes)
        model.apply(TransCycle_model_30._reset_parameter)
    else:
        raise NotImplementedError('Not Implemented!!')

    return model


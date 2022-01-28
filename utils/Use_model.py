import argparse
import sys

def Use_model(args):
    # 確定使用model種類
    model_name = str(args.modelname)
    if model_name == 'VisionTransformer':
        from zoo.vision_transformer import VisionTransformer
        model = VisionTransformer()  # 測試asyml的vision transformer code

    if model_name == 'MedT':
        from zoo.MedT.lib.models.axialnet import MedT
        model = MedT(args)

    if model_name == 'gated':
        from .zoo.MedT.lib.models.axialnet import gated
        model = gated(args)

    if model_name == 'U_net':
        from zoo import U_net
        model = U_net.U_Net(img_ch=3, output_ch=3)

    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print('MedT parameter：{:8f}M'.format(total_params / 1000000))  # 確認模型參數數量
    return model

# Use_model 單元測試
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelname', type=str, default='U_net')
    parser.add_argument('-device', type=str, default='cuda')
    args = parser.parse_args()

    Use_model(args)
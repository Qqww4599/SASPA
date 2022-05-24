import sys
import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F
from blocks import ResAttnModule, BasicBlock
import time

# __name__ = ['ResAttnModule_Unet_model', 'ResAttnModule_reduce_layer_S', 'ResAttnModule_reduce_layer_M']

def build_model(model_name, in_channels=3, depth=5, stride=8):
    if model_name == 'resnet34':
        model = smp.encoders.get_encoder(model_name, in_channels=in_channels, depth=depth, output_stride=stride)
    elif model_name == 'se_resnet50':
        model = smp.encoders.get_encoder(model_name, in_channels=in_channels, depth=depth, output_stride=stride)
    else:
        raise ValueError('Input model name!!')
    return model
class Rebuild_ResNet_M(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module_list = nn.ModuleList()
        self.module_list.add_module('New_Input_layer', nn.Sequential(model.conv1,
                                                                model.bn1,
                                                                model.relu,
                                                                model.maxpool,))
        self.module_list.add_module('New_Layer1', model.layer1)
        self.module_list.add_module('New_Layer2', model.layer2)
        self.module_list.add_module('New_Layer3', model.layer3)
        self.module_list.add_module('New_Layer4', model.layer4)
    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x

# Test
def bulid_Conv_layer(inplanes, planes, n=1, stride=1, feature_size=128):
    layer = nn.ModuleList()
    layer.append(BasicBlock(inplanes, planes, stride=stride))
    if n == 1:
        return nn.Sequential(*layer)
    if n > 1:
        for _ in range(n-1):
            layer.append(ResAttnModule(planes, planes, feature_size=feature_size))
        return nn.Sequential(*layer)
class Unet(nn.Module):
    def __init__(self, imgsize, imgchan, classes, layers_num:list=None, skip_conn_layer:list=None, print_computing_time=None,):
        super().__init__()
        if layers_num == None:
            layers_num = [1, 2, 3, 4, 1, 1, 1, 1]
            skip_conn_layer = [4, 5]
        C1, C2, C3, C4, C5 = 64, 128, 256, 512, 256
        self.layer1 = nn.Sequential(bulid_Conv_layer(imgchan, C1, n=layers_num[0], feature_size=128),
                                    )
        self.layer2 = nn.Sequential(bulid_Conv_layer(C1, C2, n=layers_num[0], feature_size=128),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    bulid_Conv_layer(C2, C2, n=layers_num[1], feature_size=64),
                                    )
        self.layer3 = nn.Sequential(bulid_Conv_layer(C2, C3, n=layers_num[2], feature_size=64),
                                    nn.MaxPool2d(stride=2, kernel_size=2))
        self.layer4 = nn.Sequential(bulid_Conv_layer(C3, C4, n=layers_num[3], feature_size=32),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        # layer4 橫向CNN堆疊+skip connection
        self.layer4_skip_connections = nn.Sequential(bulid_Conv_layer(C4, C4, n=skip_conn_layer[0], feature_size=16),
                                                     bulid_Conv_layer(C4, C4, n=skip_conn_layer[1], feature_size=16))

        self.d_layer4 = nn.Sequential(bulid_Conv_layer(C4, C3, n=layers_num[4], feature_size=16))
        self.d_layer3 = nn.Sequential(bulid_Conv_layer(C3, C2, n=layers_num[5], feature_size=32))
        self.d_layer2 = nn.Sequential(bulid_Conv_layer(C2, C1, n=layers_num[6], feature_size=64))
        self.d_layer1 = nn.Sequential(bulid_Conv_layer(C1, classes, n=layers_num[7], feature_size=128))

        self.print_computing_time = print_computing_time

    def _forward_implement(self, x):
        start = time.time()
        L1 = self.layer1(x)
        t_l1 = time.time()
        print(t_l1 - start) if self.print_computing_time is not None else None
        L2 = self.layer2(L1)
        t_l2 = time.time()
        print(t_l2 - t_l1) if self.print_computing_time is not None else None
        L3 = self.layer3(L2)
        t_l3 = time.time()
        print(t_l3 - t_l2) if self.print_computing_time is not None else None
        L4 = self.layer4(L3)
        t_l4 = time.time()
        print(t_l4 - t_l3) if self.print_computing_time is not None else None
        L4 = torch.add(self.layer4_skip_connections(L4), L4)
        t_l4_skip_conn = time.time()
        print(t_l4_skip_conn - t_l4) if self.print_computing_time is not None else None

        L3 = L3 + F.interpolate(self.d_layer4(L4), scale_factor=2, mode='bilinear', align_corners=True)
        L2 = L2 + F.interpolate(self.d_layer3(L3), scale_factor=2, mode='bilinear', align_corners=True)
        L2 = self.d_layer2(L2)
        L1 = L1 + F.interpolate(L2, scale_factor=2, mode='bilinear', align_corners=True)
        L1 = self.d_layer1(L1)
        end = time.time()
        print(end - start) if self.print_computing_time is not None else None

        return L1
    def forward(self, x):
        return self._forward_implement(x)

class Unet_ver2_reduce_layer(nn.Module):
    def __init__(self, imgsize, imgchan, classes, layers_num:list=None, skip_conn_layer:list=None, print_computing_time=None):
        super().__init__()
        if layers_num == None:
            layers_num = [1, 3, 4, 4, 3, 1]
            skip_conn_layer = [3, 4]
        C1, C2, C3, C4, C5 = 64, 128, 256, 512, 256
        self.layer1 = nn.Sequential(bulid_Conv_layer(imgchan, C1, n=layers_num[0], feature_size=128),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        self.layer2 = nn.Sequential(bulid_Conv_layer(C1, C2, n=layers_num[1], feature_size=64),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    )
        self.layer3 = nn.Sequential(bulid_Conv_layer(C2, C3, n=layers_num[2], feature_size=32),
                                    nn.MaxPool2d(stride=2, kernel_size=2))
        # layer3 橫向CNN堆疊+skip connection
        self.layer3_skip_connections = nn.Sequential(bulid_Conv_layer(C3, C3, n=skip_conn_layer[0], feature_size=16),
                                                     bulid_Conv_layer(C3, C3, n=skip_conn_layer[1], feature_size=16))

        self.d_layer3 = nn.Sequential(bulid_Conv_layer(C3, C2, n=layers_num[3], feature_size=16))
        self.d_layer2 = nn.Sequential(bulid_Conv_layer(C2, C1, n=layers_num[4], feature_size=32))
        self.d_layer1 = nn.Sequential(bulid_Conv_layer(C1, classes, n=layers_num[5], feature_size=64))

        self.print_computing_time = print_computing_time
    def _forward_implement(self, x):
        start = time.time()
        L1 = self.layer1(x)
        t_l1 = time.time()
        print(t_l1-start) if self.print_computing_time is not None else None
        L2 = self.layer2(L1)
        t_l2 = time.time()
        print(t_l2 - t_l1) if self.print_computing_time is not None else None
        L3 = self.layer3(L2)
        t_l3 = time.time()
        print(t_l3 - t_l2) if self.print_computing_time is not None else None
        L3 = torch.add(self.layer3_skip_connections(L3), L3)
        t_l3_skip_conn = time.time()
        print(t_l3_skip_conn - t_l3) if self.print_computing_time is not None else None


        L2 = L2 + F.interpolate(self.d_layer3(L3), scale_factor=2, mode='bilinear', align_corners=True) # 128, 32, 32
        d_l2 = time.time()
        print(d_l2 - t_l3_skip_conn) if self.print_computing_time is not None else None
        L1 = L1 + F.interpolate(self.d_layer2(L2), scale_factor=2, mode='bilinear', align_corners=True) # 64, 64, 64
        d_l1 = time.time()
        print(d_l1 - d_l2) if self.print_computing_time is not None else None
        L1 = F.interpolate(self.d_layer1(L1), scale_factor=2, mode='bilinear', align_corners=True) # 3, 128, 128
        end = time.time()
        print(end - d_l1) if self.print_computing_time is not None else None
        print(end-start) if self.print_computing_time is not None else None

        return L1
    def forward(self, x):
        return self._forward_implement(x)

def _reset_parameter(self):
    for m in self.modules():
        # print(m)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def TransFPN_Module_Unet_VANNILA(arg):
    # 2022/5/16測試之模型
    model = Unet(arg.imgsize,
                 arg.imgchan,
                 arg.classes,
                 layers_num=[1, 2, 3, 4, 1, 1, 1, 1],
                 skip_conn_layer=[5, 4], # [4, 5]
                 print_computing_time=None
                 )
    model.apply(_reset_parameter)
    return model
def TransFPN_Module_Unet_M2(arg):
    # 2022/5/18測試之模型。ResAttnModule_reduce_layer_M2
    model = Unet_ver2_reduce_layer(arg.imgsize,
                                   arg.imgchan,
                                   arg.classes,
                                   layers_num=[1, 3, 4, 4, 3, 1],
                                   skip_conn_layer=[5, 4],
                                   print_computing_time=None
                                   )
    model.apply(_reset_parameter)
    return model
def TransFPN_Module_Unet_S(arg):
    model = Unet_ver2_reduce_layer(arg.imgsize,
                                   arg.imgchan,
                                   arg.classes,
                                   layers_num=[1, 2, 3, 3, 2, 1],
                                   skip_conn_layer=[2, 3],
                                   print_computing_time=None
                                   )
    model.apply(_reset_parameter)
    return model
def TransFPN_Module_Unet_M(arg):
    model = Unet_ver2_reduce_layer(arg.imgsize,
                                   arg.imgchan,
                                   arg.classes,
                                   layers_num=[1, 3, 4, 4, 3, 1],
                                   skip_conn_layer=[4, 5],
                                   print_computing_time=None
                                   )
    model.apply(_reset_parameter)
    return model
def TransFPN_Module_Unet_L(arg):
    model = Unet(arg.imgsize,
                 arg.imgchan,
                 arg.classes,
                 layers_num=[2, 2, 3, 4, 4, 3, 2, 1],
                 skip_conn_layer=[3, 3],  # [4, 5]
                 print_computing_time=None
                 )
    model.apply(_reset_parameter)
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='model run TEST')
    parser.add_argument('--imgsize', default=128)
    parser.add_argument('--imgchan', default=3)
    parser.add_argument('--classes', default=1)
    args = parser.parse_args()

    model = TransFPN_Module_Unet_VANNILA(args)
    # for n,m in model.named_modules():
    #     print(n, '\t', m)
    x_test = torch.randn(8,3,128,128)
    out = model(x_test)
    total_params = sum(p.numel() for p in model.parameters())
    print('{} parameter：{:8f}M'.format(model.__class__.__name__, total_params / 1000000))  # 確認模型參數數量
    if type(out) == list:
        for f in out:
            print(f.shape)
            # ResNet34 encoder output
            # torch.Size([1, 3, 128, 128])
            # torch.Size([1, 64, 64, 64])
            # torch.Size([1, 64, 32, 32])
            # torch.Size([1, 128, 16, 16])
            # torch.Size([1, 256, 16, 16])
            # torch.Size([1, 512, 16, 16])
    else:
        print(out.shape, end='\n\n\n\n\n\n')


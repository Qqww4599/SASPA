import sys
sys.path.append(r'..\TransCycle_model\axial_attention_module')
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math
import cv2
import matplotlib.pyplot as plt
# import axial_attention as A
import axial_attention as A
from pdb import set_trace as S
import segmentation_models_pytorch as smp
import pdb

'''
TransCycle兩種不同構建方式
1. 內外圈特徵傳遞鏈，內圈前期傳給外圈，後期外圈傳給內圈
2. 類似Unet架構，使用卷積層相同

另外，自型架構Unet
'''


# ---------------------- 捲積層工具(Convolution tools) -------------------------

class Depth_sep_conv(nn.Module):
    '''深度分離卷積。減少內存訪問量與減少參數量。詳細分析可見https://zhuanlan.zhihu.com/p/411522457'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        group = in_channels
        depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=group)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
    def forward(self, x):
        return self.depthwise_separable_conv(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        conv3x3 = self.conv3x3
        # self.conv1 = conv3x3(inplanes, planes, stride) # -------------original
        self.conv1 = Depth_sep_conv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes) # -------------original
        self.conv2 = Depth_sep_conv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.identity_conv = nn.Conv2d(inplanes, planes, 1, 1, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if not identity.shape == out.shape:
            identity = self.identity_conv(identity)
        out += identity
        out = self.relu(out)

        return out

    @staticmethod
    def conv3x3(inplanes, planes, stride=1):
        return nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)

def bulid_Conv_layer(inplanes, planes, n=1, stride=1):
    layer = nn.ModuleList()
    layer.append(BasicBlock(inplanes, planes, stride=stride))
    if n == 1:
        return nn.Sequential(*layer)
    if n > 1:
        for _ in range(n-1):
            layer.append(BasicBlock(planes, planes, stride=stride))
        return nn.Sequential(*layer)
## -----------------測試--------------

## ----------------- 上 略 ----------------------



# TransCycle
class TransCycle(nn.Module):
    def __init__(self, imgsize, imgchan, classes, mid=64):
        super().__init__()
        # local cycle
        layers_nums = [imgchan, 32, 64, 32]
        layers = [4,4,4,4]
        self.Inner_layer1 = bulid_Conv_layer(layers_nums[0], layers_nums[1], n=layers[0])
        self.i1_1x1conv = nn.Conv2d(layers_nums[1],layers_nums[3],kernel_size=1)
        self.Inner_layer2 = bulid_Conv_layer(layers_nums[1], layers_nums[2], n=layers[1])
        self.i2_1x1conv = nn.Conv2d(layers_nums[2],layers_nums[1],kernel_size=1)
        self.Inner_layer3 = nn.Sequential(bulid_Conv_layer(layers_nums[2], layers_nums[1], n=layers[2]))
        self.Inner_layer4 = bulid_Conv_layer(layers_nums[1], layers_nums[3], n=layers[3])

        # global cycle
        self.Out_layer1 = bulid_Conv_layer(layers_nums[0], layers_nums[1], n=layers[0])
        self.o1_1x1conv = nn.Conv2d(layers_nums[1],layers_nums[3],kernel_size=1)
        self.Out_layer2 = bulid_Conv_layer(layers_nums[1], layers_nums[2], n=layers[1])
        self.o2_1x1conv = nn.Conv2d(layers_nums[2], layers_nums[1], kernel_size=1)
        self.Out_layer3 = nn.Sequential(bulid_Conv_layer(layers_nums[2], layers_nums[1], n=layers[2]))
        self.Out_layer4 = bulid_Conv_layer(layers_nums[1], layers_nums[3], n=layers[3])

        # intergate
        self.model_cnn = nn.Sequential(BasicBlock(layers_nums[3], layers_nums[1]),
                                       BasicBlock(layers_nums[1], layers_nums[1]),
                                       BasicBlock(layers_nums[1], layers_nums[1]),
                                       BasicBlock(layers_nums[1], classes))

        # --------- test module -------------
        # _forward3
        self.start = BasicBlock(3, 32)
        self.layers = nn.Sequential(*[BasicBlock(32, 32) for _ in range(24)])
        self.end = BasicBlock(32, 1)

    def _forward1(self, x):
        x_in, x_out = x.clone(), x.clone()

        o1 = self.Out_layer1(x_out)
        i1 = self.Inner_layer1(x_in)

        o2 = self.Out_layer2(o1)
        i2 = self.Inner_layer2(i1)

        o3 = self.Out_layer3(o2) + self.i2_1x1conv(i2)
        i3 = self.Inner_layer3(i2) + self.o2_1x1conv(o2)

        o4 = self.Out_layer4(o3) + self.i1_1x1conv(i1)
        i4 = self.Inner_layer4(i3) + self.o1_1x1conv(o1)

        x = torch.add(o4, i4)
        x = self.model_cnn(x)
        return x
    def _forward2(self, x):
        p1 = self.Inner_layer1(x) # 16,h,w
        p2 = F.max_pool2d(self.Inner_layer2(p1), 2) # 64,h/2,w/2
        p3 = F.max_pool2d(self.Inner_layer3(p2), 2) # 16,h/4,w/4
        p4 = F.max_pool2d(self.Inner_layer4(p3), 2) # 16,h/8,w/8

        c4 = self.Out_layer4(p4) # 16,h/8,w/8
        c3 = F.interpolate(c4, scale_factor=2, align_corners=True, mode='bilinear') + p3 # 16,h/4,w/4
        c3 = self.Out_layer2(c3) # 64,h/4,w/4
        c2 = F.interpolate(c3, scale_factor=2, align_corners=True, mode='bilinear') + p2  # 64,h/2,w/2
        c2 = F.interpolate(self.Out_layer3(c2), scale_factor=2, align_corners=True, mode='bilinear') + p1 # 16,h,w
        x = self.model_cnn(c2)
        return x

    def _forward3(self, x):
        x = self.start(x)
        x = self.layers(x)
        x = self.end(x)
        return x

    def forward(self, x):
        return self._forward3(x)

    def __repr__(self):
        repr_str = self.brancnn1.__repr__()
        repr_str = repr_str[:-1]
        return repr_str

# Unet
class Unet(nn.Module):
    def __init__(self, imgsize, imgchan, classes):
        super().__init__()
        C1, C2, C3, C4, C5 = 16, 32, 64, 128, 256
        self.layer1 = nn.Sequential(bulid_Conv_layer(imgchan, C1, n=1),
                                    )
        self.layer2 = nn.Sequential(bulid_Conv_layer(C1, C2, n=5),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    bulid_Conv_layer(C2, C2, n=5),
                                    )
        self.layer3 = nn.Sequential(bulid_Conv_layer(C2, C3, n=7),
                                    nn.MaxPool2d(stride=2, kernel_size=2))
        self.layer4 = nn.Sequential(bulid_Conv_layer(C3, C4, n=7),
                                    nn.MaxPool2d(stride=2, kernel_size=2),
                                    bulid_Conv_layer(C4, C4, n=9),
                                    bulid_Conv_layer(C4, C4, n=9),)

        self.d_layer4 = nn.Sequential(bulid_Conv_layer(C4, C3, n=9))
        self.d_layer3 = nn.Sequential(bulid_Conv_layer(C3, C2, n=7))
        self.d_layer2 = nn.Sequential(bulid_Conv_layer(C2, C1, n=5))
        self.d_layer1 = nn.Sequential(bulid_Conv_layer(C1, classes, n=5))
    def _forward_implement(self, x):
        L1 = self.layer1(x)
        L2 = self.layer2(L1)
        L3 = self.layer3(L2)
        L4 = self.layer4(L3)

        L3 = L3 + F.interpolate(self.d_layer4(L4), scale_factor=2, mode='bilinear', align_corners=True)
        L2 = L2 + F.interpolate(self.d_layer3(L3), scale_factor=2, mode='bilinear', align_corners=True)
        L2 = self.d_layer2(L2)
        L1 = L1 + F.interpolate(L2, scale_factor=2, mode='bilinear', align_corners=True)
        L1 = self.d_layer1(L1)

        return L1
    def forward(self, x):
        return self._forward_implement(x)

# Reset paramaters
def _reset_parameter(self):
    for m in self.modules():
        # print(m)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1)
            nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight.data)
            nn.init.zeros_(m.bias.data)

def model(arg):
    model = TransCycle(arg.imgsize, arg.imgchan, arg.classes)
    model.apply(_reset_parameter)
    return model
def Unet_model(arg):
    model = Unet(arg.imgsize, arg.imgchan, arg.classes)
    model.apply(_reset_parameter)
    return model

if __name__ == '__main__':
    import sys
    img = cv2.imread(r"D:/Programming/AI&ML/(Dataset)STU-Hospital/images/Test_Image_4.png")
    img = torch.tensor(img.transpose(2,0,1)).unsqueeze(0).to(torch.float32)
    # print(img.shape)

    x = torch.randn(1,3,9,9)
    # m = TransCycle(imgsize=128, imgchan=3, classes=1, )
    m = Unet(imgsize=128, imgchan=3, classes=1, )

    o = m(img)
    print(o.shape)
    total_params = sum(p.numel() for p in m.parameters())
    print('parameter：{:8f}M'.format(total_params / 1000000))  # 確認模型參數數量
    # m = TransCycle(imgsize=128, imgchan=3, classes=1)
    # o = m(img)

    o = rearrange(o, 'b c h w -> b h w c').squeeze(0)
    o = F.softmax(o,dim=1).detach().numpy()
    plt.imshow(o)
    plt.show()
    sys.exit()

    print(o.shape)

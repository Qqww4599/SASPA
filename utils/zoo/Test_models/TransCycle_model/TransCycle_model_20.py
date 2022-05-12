import sys
sys.path.append(r'D:\Programming\AI&ML\MainResearch\utils\zoo\Test_models\TransCycle_model\axial_attention_module')
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math
import cv2
import matplotlib.pyplot as plt
import axial_attention as A
from pdb import set_trace as S

class Depth_sep_conv(nn.Module):
    '''深度分離卷積。減少內存訪問量與減少參數量。詳細分析可見https://zhuanlan.zhihu.com/p/411522457'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        group = in_channels
        depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=group)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
    def forward(self, x):
        return self.depthwise_separable_conv(x)

## -----------------測試：基本卷積模塊--------------
class Basic_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # self.conv = Depth_sep_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()

        # BCB=基本卷積模塊
        self.BCB = nn.Sequential(
            Depth_sep_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.BCB(x)

class cnn_block(nn.Module):
    def __init__(self, dim_in, dim_out, depth, feature_shape=128, middim=128):
        super().__init__()
        self.dim = dim_in
        middim = middim
        self.depth = depth
        self.first_conv = Depth_sep_conv(dim_in, middim, 1)
        self.layers = nn.Identity()
        layer = lambda: nn.Sequential(Depth_sep_conv(middim, middim, 3,1,1),
                                           nn.LayerNorm([middim,feature_shape,feature_shape]),
                                           nn.LeakyReLU(inplace=True),)
        layers = nn.ModuleList([])
        if depth > 2:
            for _ in range(depth-2):
                layers.append(layer())
            self.layers = nn.Sequential(*layers)
        self.final_conv = Depth_sep_conv(middim, dim_out, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = x + self.layers(x)
        x = self.final_conv(x)
        return x

class attention_block(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TransCycle(nn.Module):
    def __init__(self, imgsize, imgchan, classes, mid=64):
        super().__init__()
        # local cycle
        layers_nums = [imgchan,16,64,8]
        self.Inner_layer1 = Basic_conv_block(layers_nums[0],layers_nums[1],kernel_size=3,stride=1,padding=1)
        self.Inner_layer2 = Basic_conv_block(layers_nums[1],layers_nums[2],kernel_size=3,stride=2,padding=1)
        self.Inner_layer3 = nn.Sequential(Basic_conv_block(layers_nums[2],layers_nums[1],kernel_size=3,stride=1,padding=1),
                                       nn.Upsample(scale_factor=2))
        self.Inner_layer4 = Basic_conv_block(layers_nums[1],layers_nums[3],kernel_size=3,stride=1,padding=1)

        # global cycle
        self.Out_layer1 = Basic_conv_block(layers_nums[0], layers_nums[1], kernel_size=3, stride=1, padding=1)
        self.Out_layer2 = Basic_conv_block(layers_nums[1], layers_nums[2], kernel_size=3, stride=2, padding=1)
        self.Out_layer3 = nn.Sequential(
            Basic_conv_block(layers_nums[2], layers_nums[1], kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2))
        self.Out_layer4 = Basic_conv_block(layers_nums[1], layers_nums[3], kernel_size=3, stride=1, padding=1)

        # intergate
        self.model_cnn = Basic_conv_block(layers_nums[3], classes, 3,1,1)

        # --------- test module -------------
        self.layer_attention1 = nn.Sequential(*[A.AxialAttention(dim=64, heads=8) for _ in range(4)])
        self.layer_attention2 = nn.Sequential(*[A.AxialAttention(dim=64, heads=8) for _ in range(4)])

    def forward(self, x):
        x_in, x_out = x.clone(), x.clone()

        x_out =self.Out_layer1(x_out)
        x_in = self.Inner_layer1(x_in)

        x_out = self.Out_layer2(x_out)
        x_out = self.layer_attention2(x_out)
        x_in = torch.add(self.layer_attention1(self.Inner_layer2(x_in),) ,x_out)

        x_out = self.Out_layer3(x_out)
        x_in = self.Inner_layer3(x_in)

        x_out = self.Out_layer4(x_out)
        x_in = self.Inner_layer4(x_in)

        x = torch.add(x_out, x_in)
        x = self.model_cnn(x)
        return x
    def __repr__(self):
        repr_str = self.brancnn1.__repr__()
        repr_str = repr_str[:-1]
        return repr_str

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

if __name__ == '__main__':
    import sys
    import numpy as np
    img_path = r"D:/Programming/AI&ML/(Dataset)STU-Hospital/images/Test_Image_4.png"
    imgchan = 1

    img = cv2.imread(img_path) if imgchan == 3 else np.expand_dims(cv2.imread(img_path, 0),axis=-1)
    img = torch.tensor(img.transpose((2,0,1))).unsqueeze(0).to(torch.float32)

    x = torch.randn(1,1,9,9)
    # m = AxialImageTransformer(dim=3, depth=2, heads=1)
    # print(m(x).shape)
    m = TransCycle(imgsize=128, imgchan=imgchan, classes=1)

    o = m(img)
    print(o.shape)

    o = rearrange(o, 'b c h w -> b h w c').squeeze(0)
    o = F.softmax(o,dim=1).detach().numpy()
    plt.imshow(o)
    plt.show()

    total_params = sum(p.numel() for p in m.parameters())
    print('parameter：{:8f}M'.format(total_params / 1000000))  # 確認模型參數數量
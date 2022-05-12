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
        # self.position_embedding = nn.Identity()
        self.brancnn1 = cnn_block(4*imgchan, 4*mid, depth=1, feature_shape=imgsize//2)
        self.brantrans1 = nn.Sequential(nn.Conv2d(4*3, 4*mid, 7,1,3),
                                        A.AxialImageTransformer(dim=4*mid, depth=1, heads=8))

        self.brancnn_mid = cnn_block(mid, mid, depth=1)

        self.brancnn2 = cnn_block(16*mid, 16*mid, depth=1, feature_shape=imgsize//4)
        # self.brantrans2 = nn.Sequential(A.AxialImageTransformer(dim=16*mid, depth=1, heads=2),
        #                                 ) # 參數量最多的就是你

        self.f_brancnn = cnn_block(mid, mid, depth=1)

        # global cycle
        self.glocnn1 = cnn_block(imgchan, mid, depth=1, feature_shape=imgsize)
        self.glocnn2 = cnn_block(mid, mid, depth=1)

        # intergate
        self.model_cnn = cnn_block(2*mid, classes, depth=1)

        # reset param

    def _local_cycle(self, x, t_global):
        # x = self.position_embedding(x)
        #
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w', p1=2, p2=2)
        x_trans = self.brantrans1(x)
        x_cnn = self.brancnn1(x)
        x = torch.add(x_trans, x_cnn)
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (p1 h) (p2 w)', p1=2, p2=2)
        x = torch.add(x, t_global[1])

        x_res = x.clone()
        x = self.brancnn_mid(x) + x_res

        # x = self.position_embedding(x)
        # x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w', p1=4, p2=4)
        # x_trans = self.brantrans2(x)
        # x_cnn = self.brancnn2(x)
        # x = self.brancnn2(x)
        # x = torch.add(x_trans, x_cnn)
        # x = rearrange(x, 'b (p1 p2 c) h w -> b c (p1 h) (p2 w)', p1=4, p2=4)
        x = torch.add(x, t_global[0])

        x = self.f_brancnn(x)

        return x

    def _global_cycle(self, x):
        f1 = self.glocnn1(x)
        f2 = self.glocnn2(f1)
        return f1, f2

    def forward(self, x):
        f_glo = self._global_cycle(x)
        x = self._local_cycle(x, f_glo)
        x = torch.cat((f_glo[0], x), dim=1)
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
    img = cv2.imread(r"D:/Programming/AI&ML/(Dataset)STU-Hospital/images/Test_Image_4.png")
    img = torch.tensor(img.transpose(2,0,1)).unsqueeze(0).to(torch.float32)

    x = torch.randn(1,3,9,9)
    # m = AxialImageTransformer(dim=3, depth=2, heads=1)
    # print(m(x).shape)

    m = TransCycle(imgsize=128, imgchan=3, classes=1)
    o = m(img)

    o = rearrange(o, 'b c h w -> b h w c').squeeze(0)
    o = F.softmax(o,dim=1).detach().numpy()
    plt.imshow(o)
    plt.show()

    print(o.shape)
    total_params = sum(p.numel() for p in m.parameters())
    print('parameter：{:8f}M'.format(total_params / 1000000))  # 確認模型參數數量
from torch import nn

import time
import torch
import torch.nn.functional as F
import logging

from MainResearch.utils.zoo.Test_models.TransFPNSeries.Attention import AxialAttentionDynamic


class DepthSepConv(nn.Module):
    """深度分離卷積。減少內存訪問量與減少參數量。詳細分析可見https://zhuanlan.zhihu.com/p/411522457"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        group = in_channels
        depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=group, bias=False)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)

    def forward(self, feature):
        return self.depthwise_separable_conv(feature)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        # self.conv1 = conv3x3(inplanes, planes, stride) # -------------original
        self.conv1 = DepthSepConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes) # -------------original
        self.conv2 = DepthSepConv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if self.stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=(1, 1),
                                                      stride=(stride, stride), bias=False),
                                            nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True)
                                            )
        self.identity_conv = nn.Conv2d(inplanes, planes, 1, 1, 0)

    def forward(self, feature):
        identity = feature

        out = self.conv1(feature)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.inplanes != self.planes:
            identity = self.downsample(feature)
        # if not identity.shape == out.shape:
        #     identity = self.identity_conv(identity)
        out += identity
        out = self.relu(out)

        return out

    @staticmethod
    def conv3x3(inplanes, planes, stride=1):
        return nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)


class MiniFPN(nn.Module):
    def __init__(self, inplanes, planes, n_group, stride=1):
        """
        :param inplanes: Input channel
        :param planes: output channel. TODO: Not Use!
        :param n_group: 輸入特徵沿著<通道>平分成n等分
        :param stride: 步距。TODO: Not Use!
        """
        inplanes = inplanes // n_group
        super(MiniFPN, self).__init__()
        self.N = n_group
        self.branch = nn.ModuleList()
        for L in range(n_group):
            if L == 0:
                self.branch.add_module('miniFPN layer1', nn.Identity())
            else:
                cur_blocks = nn.Sequential(*[self.convblock(inplanes, inplanes) for _ in range(L)],
                                           nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1),
                                           nn.BatchNorm2d(inplanes))
                self.branch.add_module(f'miniFPN layer{L + 1}', cur_blocks)

    def forward(self, feature):
        """x.shape: (B,C,H,W)"""
        All_Feature = []
        feature = torch.chunk(feature, chunks=self.N, dim=1)  # return list
        current_Time = time.time()
        # miniFPN.branch.Convolution
        for i, (c, m) in enumerate(zip(feature, self.branch)):
            All_Feature.append(m(c))
        logging.debug("miniFPN.branch() use {:4f} sec".format((time.time() - current_Time)))
        current_Time = time.time()

        # miniFPN.branch.Fusion
        for i, ch_feature in enumerate(All_Feature[:0:-1]):
            All_Feature[len(feature) - (i + 1)] = F.interpolate(ch_feature, scale_factor=2)
        logging.debug("miniFPN.branch.Fusion use {:4f} sec".format((time.time() - current_Time)))
        All_Feature = torch.cat(All_Feature, dim=1)

        return All_Feature

    @staticmethod
    def convblock(inplanes, planes):
        return nn.Sequential(DepthSepConv(inplanes, planes),
                             nn.BatchNorm2d(planes),
                             nn.ReLU(inplace=True), )


class AttentionPathway(nn.Module):

    """
    目前沒有加入ResNet的殘差結構，未來可以考慮加入。
    feature size: 輸入特徵大小。進行計算時大小會 // 2**N
    """

    def __init__(self, inplanes, planes, n_of_layer=2, stride=1, attention_blocks=1, feature_size=128):
        """
        :param inplanes: Input channel
        :param n_of_layer: Attention縮減層數。每經過一層縮小1/2
        :param stride: 步距。TODO: Not Use!
        :param attention_blocks: TODO: Not Use!
        :param feature_size: attention windows大小。默認為輸入特徵大小

        :return feature: 經過attention的feature。size等於feature input。
        """
        super(AttentionPathway, self).__init__()
        self.Blocks = nn.Sequential()
        kernel_size = feature_size // 2 ** n_of_layer
        for L in range(n_of_layer):
            cur_block = nn.Sequential(self.convblock(inplanes, planes),
                                      nn.AvgPool2d(kernel_size=2, stride=2))
            self.Blocks.add_module(f'Num {L + 1} Attn Conv block', cur_block)

        self.attention_blocks = nn.Sequential(nn.Conv2d(planes, planes, 3, 1, 1),
                                              nn.BatchNorm2d(planes),
                                              *[nn.Sequential(AxialAttentionDynamic(inplanes=planes, planes=planes,
                                                                                    kernel_size=kernel_size),
                                                              AxialAttentionDynamic(inplanes=planes, planes=planes,
                                                                                    kernel_size=kernel_size,
                                                                                    width=True)
                                                              )],
                                              # *[AxialAttentionDynamic(inplanes=planes, planes=planes,
                                              #                         kernel_size=kernel_size) for _ in
                                              #   range(attention_blocks)]
                                              )

    def forward(self, feature):
        # print('input: ',x.shape, Attention_Pathway.__name__)
        current_Time = time.time()
        feature = self.Blocks(feature)  # 1, 16, 32, 32
        logging.debug("Attention_Pathway.Blocks() use {:4f} sec".format((time.time() - current_Time)))
        current_Time = time.time()
        feature = self.attention_blocks(feature)
        logging.debug("Attention_Pathway.attention_blocks() use {:4f} sec".format((time.time() - current_Time)))
        feature = F.interpolate(feature, scale_factor=4)
        return feature

    @staticmethod
    def convblock(inplanes, planes):
        return nn.Sequential(DepthSepConv(inplanes, planes),
                             nn.BatchNorm2d(planes),
                             # nn.ReLU(inplace=True)
                             )


class TransFPNModule(nn.Module):
    """
    Input: B, C, H, W
    Output: nn.Identity(Input)。與輸入影像相同大小

    特色：
    1. 通道縮減(reduction=4)
    2. 微型卷積特徵金字塔(FPN)，通道分離，學習特徵，通過微特徵金字塔整合
    3. 全局特徵卷積 + 注意力
    4. 注意力整合全局特徵，最終透過ResNet通道整合

    :parameter inplanes: 輸入通道
    :parameter planes: 輸出通道
    :parameter downsample: TODO: Not Use!
    :parameter n_fpn: 輸入特徵沿著<通道>平分成n等分，即為微型卷積特徵金字塔之通道分離數量(必須可以被整除)。
    :parameter n_attn: Attention縮減層數。每經過一層縮小1/2

    """

    def __init__(self, inplanes, planes, downsample=None, n_fpn=4, n_attn=2, feature_size=128, reduction=4):

        super().__init__()
        # Ci = inplanes // reduction # 原始版本。
        Ci = inplanes // reduction if inplanes > reduction else inplanes  # 如果輸入通道數量低於reduction數量會輸出0，此寫法已修正
        # Preprocessing
        self.preprocess = nn.Sequential()
        self.preprocess.add_module('Reduction layer',
                                   nn.Sequential(nn.Conv2d(inplanes, Ci, 1),
                                                 nn.BatchNorm2d(Ci)))
        # mini FPN pathway
        self.miniFPN = MiniFPN(Ci, Ci, n_group=n_fpn)
        # Attention pathway
        self.Attention_Pathway = AttentionPathway(inplanes=Ci, planes=Ci, n_of_layer=n_attn, feature_size=feature_size)

        # End_process
        self.End_process = nn.Sequential(BasicBlock(2 * Ci, Ci),
                                         nn.Conv2d(Ci, planes, kernel_size=1),
                                         nn.BatchNorm2d(planes))

        # Residual dimension adjust
        if inplanes != planes:
            self.adjust = nn.Conv2d(inplanes, planes, 1, 0, 0)

    def forward(self, module_input):

        """:parameter module_input: Module input,模組輸入特徵"""

        identity = module_input.clone()  # x.shape
        current_Time = time.time()
        module_input = self.preprocess(module_input)
        logging.debug("preprocess use {:4f} sec".format((time.time() - current_Time)))
        x_FPN = self.miniFPN(module_input)
        x_Trans = self.Attention_Pathway(module_input)
        module_input = torch.cat((x_FPN, x_Trans), dim=1)
        current_Time = time.time()
        module_input = self.End_process(module_input) + identity
        logging.debug("End_process use {:4f} sec".format((time.time() - current_Time)))
        if module_input.shape[1] != identity.shape[1]:
            self.adjust()
            raise NotImplementedError
        return module_input


if __name__ == '__main__':
    import pdb, sys

    logging.basicConfig(level=logging.DEBUG)
    B, C, H, W = 1, 64, 128, 128
    x = torch.randn(B, C, H, W)

    # module = BasicBlock(inplanes=64, planes=64)
    module = TransFPNModule(inplanes=64, planes=64, feature_size=128)
    print(module(x).shape)
    # total_params = sum(p.numel() for p in module.parameters())
    # print('{} parameter：{:8f}M'.format(module.__class__.__name__, total_params / 1000000))  # 確認模型參數數量

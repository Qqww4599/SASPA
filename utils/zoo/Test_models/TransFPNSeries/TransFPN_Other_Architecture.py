import logging
import time

import torch
from torch import nn
from torch.nn import functional as F

from MainResearch.utils.zoo.Test_models.TransFPNSeries.blocks import DepthSepConv, BasicBlock


class AttentionPathway(nn.Module):

    """
    目前沒有加入ResNet的殘差結構，未來可以考慮加入。
    feature size: 輸入特徵大小。進行計算時大小會 // 2**N
    """

    def __init__(self, inplanes, planes, n_of_layer=2, stride=1, attention_blocks=1, feature_size=128):
        super(AttentionPathway, self).__init__()
        self.Blocks = nn.Sequential()
        kernel_size = feature_size // 2 ** n_of_layer
        #
        for L in range(n_of_layer):
            cur_block = nn.Sequential(self.convblock(inplanes, planes),
                                      nn.AvgPool2d(kernel_size=2, stride=2))
            self.Blocks.add_module(f'Num {L+1} Attn Conv block', cur_block)

        # mobileViTv2_attention_module快很多
        self.attention_blocks = nn.Sequential(nn.Conv2d(planes, planes, 3, 1, 1),
                                              nn.BatchNorm2d(planes),
                                              Mobilevitv2AttentionModule(planes, planes, kernel_size,
                                                                         output_feature_size=True),
                                              )

    def forward(self, feature):
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


class Mobilevitv2AttentionModule(nn.Module):
    def __init__(self, in_feature, out_feature, size: int, patchsize=1, output_feature_size: bool = False):
        super(Mobilevitv2AttentionModule, self).__init__()
        # some parameters
        self.patchsize = patchsize
        self.Output_Feature_Size = output_feature_size
        self.mid_dim = out_feature

        # let input feature(B,C,H,W) to (B,N,C)
        assert size // patchsize, ValueError("remainder must be 0")
        self.feature_to_BNC = nn.Conv2d(in_feature, self.mid_dim, kernel_size=patchsize, stride=patchsize)
        self.K_feature_to_BNC = nn.Conv2d(in_feature, self.mid_dim, kernel_size=patchsize, stride=patchsize)
        self.V_feature_to_BNC = nn.Conv2d(in_feature, self.mid_dim, kernel_size=patchsize, stride=patchsize)

        # feature to scaler, context score
        self.to_context_score = nn.Sequential(nn.Linear(self.mid_dim, 1), nn.Softmax(dim=-1))

        # linear after mul_IK * V
        self.layers = nn.Linear(self.mid_dim, out_feature)

    def _forward_impl(self, feature) -> (torch.Tensor, tuple):
        B, C, H, W = feature.shape
        K = self.K_feature_to_BNC(feature).flatten(-2).permute(0, 2, 1)
        V = self.V_feature_to_BNC(feature).flatten(-2).permute(0, 2, 1)
        I = self.feature_to_BNC(feature).flatten(-2).permute(0, 2, 1)  # B,N,C
        context_score = self.to_context_score(I)  # B, N
        mul_IK = (context_score * K).sum(dim=0)
        feature = mul_IK * V
        return self.layers(feature), (B, self.mid_dim, H // self.patchsize, W // self.patchsize)

    def forward(self, x):
        x, shape = self._forward_impl(x)
        if self.Output_Feature_Size:
            return x.reshape(shape)
        return x, shape


class FeedForward(nn.Module):
    """TODO: ADD Mobilevitv2AttentionModule feedforward module"""
    def __init__(self):
        super().__init__()
        pass

    def _forward(self, feature, shape):
        return feature.reshape(shape)

    def forward(self, feature, shape):
        return self._forward(feature, shape)


# mobileViTv2_attention_module快很多
# ViTv2實作方法。
class MiniFPN(nn.Module):
    def __init__(self, inplanes, planes, n_group, stride=1,
                 input_size: int = 128,
                 test_switch: bool = False):
        """
        測試MiniFPN架構方法。
        TODO: 目前的通道數都是原本的輸入通道數，後面還需要再修正

        :param inplanes: Input channel
        :param planes: output channel. TODO: Not Use!
        :param n_group: 輸入特徵沿著<通道>平分成n等分
        :param stride: 步距。TODO: Not Use!
        :param input_size: 輸入影像大小

        :parameter test_switch: 測試開啟開關
        """
        inplanes = inplanes // n_group
        super(MiniFPN, self).__init__()
        self.N = n_group
        self.branch = nn.ModuleList()
        current_size: int = input_size
        for L in range(n_group):
            if L == 0:
                self.branch.add_module('miniFPN layer1', nn.Identity())
            else:
                cur_blocks = nn.Sequential(*[self.convblock(inplanes, inplanes) for _ in range(L)],
                                           nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
                                           if L >= 2 and test_switch else nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1),
                                           nn.BatchNorm2d(planes)
                                           if L >= 2 and test_switch else nn.BatchNorm2d(inplanes))
                # 設定test switch開啟，假設L>2以後加入attention layer
                if L >= 2 and test_switch:
                    cur_blocks.add_module(f'miniFPN Layer attention layer',
                                          Mobilevitv2AttentionModule(planes, inplanes, size=current_size // 2 ** L,
                                                                     output_feature_size=True),
                                          )
                self.branch.add_module(f'miniFPN layer{L + 1}', cur_blocks)

    def forward(self, feature):
        """
        前向傳播方法

        :param feature :(B,C,H,W)
        """
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
            All_Feature[self.N - (i + 1)] = F.interpolate(ch_feature, scale_factor=2)
        logging.debug("miniFPN.branch.Fusion use {:4f} sec".format((time.time() - current_Time)))
        All_Feature = torch.cat(All_Feature, dim=1)

        return All_Feature

    @staticmethod
    def convblock(inplanes, planes):
        return nn.Sequential(DepthSepConv(inplanes, planes),
                             nn.BatchNorm2d(planes),
                             nn.ReLU(inplace=True), )


class TransFPNModuleOtherArchitecture(nn.Module):
    """
    Input: B, C, H, W
    Output: nn.Identity(Input)

    測試不同TransFPN架構方法。

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
                                                 nn.BatchNorm2d(Ci),
                                                 )
                                   )
        # mini FPN pathway
        self.miniFPN = MiniFPN(Ci, Ci, n_group=n_fpn,
                               test_switch=True)
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
        identity = module_input.clone()  # x.shape
        current_time = time.time()
        module_input = self.preprocess(module_input)
        logging.debug("preprocess use {:4f} sec".format((time.time() - current_time)))

        x_fpn = self.miniFPN(module_input)
        x_trans = self.Attention_Pathway(module_input)
        module_input = torch.cat((x_fpn, x_trans), dim=1)

        current_time = time.time()
        module_input = self.End_process(module_input) + identity
        logging.debug("End_process use {:4f} sec".format((time.time() - current_time)))
        if module_input.shape[1] != identity.shape[1]:
            self.adjust()
            raise NotImplementedError
        return module_input


if __name__ == '__main__':
    import pdb
    import sys

    NewTransFPNModule = TransFPNModuleOtherArchitecture
    logging.basicConfig(level=logging.DEBUG)
    B, C, H, W = 1, 64, 128, 128
    x = torch.randn(B, C, H, W)

    # module = BasicBlock(inplanes=64, planes=64)
    module = NewTransFPNModule(inplanes=64, planes=64, feature_size=128)
    print(module(x).shape)
    # total_params = sum(p.numel() for p in module.parameters())
    # print('{} parameter：{:8f}M'.format(module.__class__.__name__, total_params / 1000000))  # 確認模型參數數量
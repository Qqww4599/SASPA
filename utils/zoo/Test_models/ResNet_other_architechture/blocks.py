import sys
import segmentation_models_pytorch as smp
import torch
from torch import nn
import math
import torch.nn.functional as F
import pdb
from einops import rearrange

# seblock = smp.encoders.senet.SEBottleneck(inplanes=3, planes=8, stride=1, groups=8, reduction=2)
# print(seblock)

class Depth_sep_conv(nn.Module):
    '''深度分離卷積。減少內存訪問量與減少參數量。詳細分析可見https://zhuanlan.zhihu.com/p/411522457'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        group = in_channels
        depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=group, bias=False)
        point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
    def forward(self, x):
        return self.depthwise_separable_conv(x)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        conv3x3 = self.conv3x3
        self.inplanes = inplanes
        self.planes = planes
        # self.conv1 = conv3x3(inplanes, planes, stride) # -------------original
        self.conv1 = Depth_sep_conv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes) # -------------original
        self.conv2 = Depth_sep_conv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if self.stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                                            nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                                            )
        self.identity_conv = nn.Conv2d(inplanes, planes, 1, 1, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.inplanes != self.planes:
            identity = self.downsample(x)
        # if not identity.shape == out.shape:
        #     identity = self.identity_conv(identity)
        out += identity
        out = self.relu(out)

        return out

    @staticmethod
    def conv3x3(inplanes, planes, stride=1):
        return nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
class qkv_transform(torch.nn.Conv1d):
    """Convolution 1d"""
class AxialAttention_dynamic(nn.Module):
    def __init__(self, inplanes, planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (inplanes % groups == 0) and (planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.groups = groups
        self.group_planes = planes // groups # groups = heads 512/8 = 64
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(inplanes, planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(planes * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        # print(x.shape)
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)
        # print('v.shape', v.shape)
        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        # print('q_embedding.shape', q_embedding.shape)
        # print('k_embedding.shape', k_embedding.shape)
        # print('v_embedding.shape', v_embedding.shape)
        # print('='*10,'q shape',q.shape, 'q_embedding shape',q_embedding.shape,'='*10)
        # print('self outplanes：',self.planes)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output) # 這一步讓H,W減半

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.inplanes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
class miniFPN(nn.Module):
    def __init__(self, inplanes, planes, N, stride=1):
        inplanes = inplanes // N
        super(miniFPN, self).__init__()
        self.N = N
        self.branch = nn.ModuleList()
        for l in range(N):
            if l == 0:
                self.branch.add_module('miniFPN layer1', nn.Identity())
            else:
                cur_blocks = nn.Sequential(*[self.Convblock(inplanes, inplanes) for _ in range(l)],
                                           nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1),
                                           nn.BatchNorm2d(inplanes))
                self.branch.add_module(f'miniFPN layer{l+1}', cur_blocks)
    def forward(self, x):
        '''x.shape: (B,C,H,W)'''
        All_Feature = []
        x = torch.chunk(x, chunks=self.N, dim=1) # return list
        for i,(c ,m) in enumerate(zip(x, self.branch)):
            All_Feature.append(m(c))
        for i, ch_feature in enumerate(All_Feature[:0:-1]):
            All_Feature[len(x)-(i+1)] = F.interpolate(ch_feature, scale_factor=2)
        All_Feature = torch.cat(All_Feature, dim=1)

        return All_Feature

    def Convblock(self, inplanes, planes):
        return nn.Sequential(Depth_sep_conv(inplanes, planes),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(inplace=True),)
class ResNet_Attention(nn.Module):
    '''
    目前沒有加入ResNet的殘差結構，未來可以考慮加入。
    feature size: 輸入特徵大小。進行計算時大小會 // 2**N
    '''
    def __init__(self, inplanes, planes, N=2, stride=1, attention_blocks=1, feature_size=128):
        super(ResNet_Attention, self).__init__()
        self.Blocks = nn.Sequential()
        kernel_size = feature_size // 2**N
        for l in range(N):
            cur_block = nn.Sequential(self.Convblock(inplanes, planes),
                                      nn.AvgPool2d(kernel_size=2, stride=2))
            self.Blocks.add_module(f'Num {l+1} Attn Conv block', cur_block)
        self.attention_blocks = nn.Sequential(nn.Conv2d(planes, planes, 3, 1, 1),
                                              nn.BatchNorm2d(planes),
                                              *[nn.Sequential(AxialAttention_dynamic(inplanes=planes, planes=planes, kernel_size=kernel_size),
                                                              AxialAttention_dynamic(inplanes=planes, planes=planes, kernel_size=kernel_size, width=True)
                                                              )]
                                              # *[AxialAttention_dynamic(inplanes=planes, planes=planes, kernel_size=kernel_size) for _ in range(attention_blocks)]
                                              )
    def Convblock(self, inplanes, planes):
        return nn.Sequential(Depth_sep_conv(inplanes, planes),
                    nn.BatchNorm2d(planes),
                    # nn.ReLU(inplace=True)
                    )
    def forward(self, x):
        x = self.Blocks(x)
        x = self.attention_blocks(x)
        x = F.interpolate(x, scale_factor=4)
        return x

class ResAttnModule(nn.Module):
    '''
    Input: B, C, H, W
    Output: nn.Identity(Input)

    特色：
    1. 通道縮減(reduction=4)
    2. 微型卷積特徵金字塔(FPN)，通道分離，學習特徵，通過微特徵金字塔整合
    3. 全局特徵卷積 + 注意力
    4. 注意力整合全局特徵，最終透過ResNet通道整合

    :parameter
    inplanes: 輸入通道
    planes: 輸出通道
    N: 微型卷積特徵金字塔之通道分離數量(必須可以被整除)
    '''
    def __init__(self, inplanes, planes, downsample=None, N_fpn=4, N_attn=2, feature_size=128, reduction=4):
        super().__init__()
        Ci = inplanes // reduction
        # Preprocessing
        self.preprocess = nn.Sequential()
        self.preprocess.add_module('Reduction layer',
                            nn.Sequential(nn.Conv2d(inplanes, Ci, 1),
                                          nn.BatchNorm2d(Ci)))
        # mini FPN
        self.miniFPN = miniFPN(Ci, Ci, N=4)
        # ResNet Attention
        self.ResNet_Attention = ResNet_Attention(inplanes=Ci, planes=Ci, N=2, feature_size=feature_size)

        # End_process
        self.End_process = nn.Sequential(BasicBlock(2*Ci, Ci),
                                         nn.Conv2d(Ci, planes, kernel_size=1),
                                         nn.BatchNorm2d(planes))

        # Residual dimension adjust
        if inplanes != planes:
            self.adjust = nn.Conv2d(inplanes, planes, 1, 0, 0)


        # raise NotImplementedError

    def forward(self, x):
        identity = x.clone() # x.shape
        x = self.preprocess(x)
        x_FPN = self.miniFPN(x)
        x_Trans = self.ResNet_Attention(x)
        x = torch.cat((x_FPN, x_Trans), dim=1)
        x = self.End_process(x) + identity
        if x.shape[1] != identity.shape[1]:
            self.adjust()
            raise NotImplementedError
        return x


if __name__ == '__main__':
    B, C, H, W = 8, 64, 128, 128
    x = torch.randn(B,C,H,W)
    # module = BasicBlock(inplanes=64, planes=64)
    module = ResAttnModule(inplanes=64, planes=32, feature_size=128)
    print(module(x).shape)
    total_params = sum(p.numel() for p in module.parameters())
    print('{} parameter：{:8f}M'.format(module.__class__.__name__, total_params / 1000000))  # 確認模型參數數量
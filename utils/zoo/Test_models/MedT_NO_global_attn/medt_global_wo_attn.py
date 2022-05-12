import torch
import pdb
import math
import sys
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
import segmentation_models_pytorch as smp
from torchvision import models

class basicblock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(basicblock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x
class global_cnn(nn.Module):
    '''input: b,c(in_plane),h,w'''
    def __init__(self, in_plane, out_plane, downsample=False, mid_dim=32):
        super(global_cnn, self).__init__()

        self.in_plane = in_plane
        self.out_plane = out_plane
        if downsample:
            self.stride = 2
        else:
            self.stride = 1
        self.encoder1 = basicblock(in_plane, mid_dim, kernel_size=5, padding=2)
        self.encoder2 = basicblock(mid_dim, mid_dim, kernel_size=5, padding=2)
        self.encoder3 = basicblock(mid_dim, mid_dim, kernel_size=5, padding=2)
        self.final_encoder = basicblock(mid_dim, self.out_plane, stride=self.stride, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.final_encoder(x)
        x = self.relu(x)
        return x
class qkv_transform(torch.nn.Conv1d):
    """Convolution 1d"""
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
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


        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)


        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups # groups = heads 512/8 = 64
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

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
        # print('self outplanes：',self.out_planes)
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

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output) # 這一步讓H,W減半

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
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

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W, 1, self.groups, H, H).sum(dim=1).contiguous()

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)

        sv = sv.reshape(N * W, self.out_planes * 1, H).contiguous()
        output = self.bn_output(sv).reshape(N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        # nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
class Axialattention_conv(AxialAttention_dynamic):
    '''
    input: b,c,h,w
    output: b,c,h,w

    :param
        in_planes: 傳入的通道數量(image channels)。
        planes: 欲輸出的通道數量。輸出維度會受到*expansion，ex:如果expansion=2，輸出維度則為planes*2
        stride: attention block中的步長。如果stirde=2則輸出H、W減半。降低大小操作在self.pooling()進行
        group: attention head數量。預設為1
        base_width: attention block輸入的通道擴張倍數。如果設置為128, 則attention時使用的通道數則為2*planes
        dilation: pass
        norm_layer: 使用的標準化層
        kernel_size: 位置編碼時的相對位置大小。
    :returns
        output: (B, planes*2, H // stride, W // stride)

    '''
    def __init__(self, in_plane, out_planes, groups=8, kernel_size=56 ,stride=1, width=False, *kwargs):
        super(Axialattention_conv, self).__init__(in_planes=in_plane, out_planes=in_plane)
        self.q_conv_proj = nn.Sequential(OrderedDict([('conv',nn.Conv2d(in_plane,in_plane // 2,kernel_size=3,stride=1, padding=1,bias=False, groups=in_plane // 2)),
                                                 ('bn', nn.BatchNorm2d(in_plane // 2)), # b,c,h,w
                                                 ])) # output shape: (b, in_plane // 2, h, w)
        self.k_conv_proj = nn.Sequential(OrderedDict([('conv',nn.Conv2d(in_plane,in_plane // 2,kernel_size=3,stride=1, padding=1, bias=False, groups=in_plane // 2)),
                                                 ('bn', nn.BatchNorm2d(in_plane // 2)), # b,c,h,w
                                                 ])) # output shape: (b, in_plane // 2, h, w)
        self.v_conv_proj = nn.Sequential(OrderedDict([('conv',nn.Conv2d(in_plane,in_plane,kernel_size=3,stride=1, padding=1, bias=False, groups=in_plane)),
                                                 ('bn', nn.BatchNorm2d(in_plane)), # b,c,h,w
                                                 ])) # output shape: (b, in_plane, h, w)
        self.group_planes = out_planes // groups # groups = heads 512/8 = 64
        self.kernel_size = kernel_size
        query_index = torch.arange(self.kernel_size).unsqueeze(0)
        key_index = torch.arange(self.kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + self.kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, self.kernel_size * 2 - 1), requires_grad=True)
        self.groups = groups
        self.bn_similarity = nn.BatchNorm2d(groups*3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)
        self.width = width
        self.pooling = nn.AvgPool2d(stride, stride=stride)
        self.stride = stride

    def forward(self, x):
        # x = N, C, H, W
        q = self.q_conv_proj(x) # conv 映射到q, 替代conv1D映射, q.shape = b, in_plane // 2, h, w
        k = self.k_conv_proj(x) # conv 映射到q, 替代conv1D映射, k.shape = b, in_plane // 2, h, w
        v = self.v_conv_proj(x) # conv 映射到q, 替代conv1D映射, v.shape = b, in_plane, h, w
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # B, W, C, H
        N, W, C, H = x.shape
        q = q.contiguous().view(N * W, self.groups, C // (2*self.groups), H)
        k = k.contiguous().view(N * W, self.groups, C // (2*self.groups), H)
        v = v.contiguous().view(N * W, self.groups, C // self.groups, H)

        # Calculate position embedding, embedding must to be:
        #   q_embedding.shape = in_plane // 2, h, w
        #   k_embedding.shape = in_plane // 2, h, w
        #   v_embedding.shape = in_plane, h, w
        all_embeddings = torch.index_select(self.relative, dim=1, index=self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        # q.shape = H*W , group, group_plane // 2, H
        # k.shape = H*W , group, group_plane // 2, H
        # v.shape = H*W , group, group_plane, H
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)
        stacked_output = torch.cat([sv, sve], dim=-1)
        stacked_output =  stacked_output.view(N * W, self.out_planes * 2, H) # 256, 512 *2, 256
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output
# end of attn definition
class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                          width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))  # 傳入axial_attention的維度
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                  width=True) # 這一步會讓H,W減半
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)
        # pdb.set_trace()
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # print('===finish once axielblock===')
        return out
class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # print(kernel_size)
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # pdb.set_trace()

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out = self.hight_block(out)
        out = self.width_block(out)

        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class AxialBlock_conv_dynamic(AxialBlock_dynamic):
    expansion = 2
    '''
    convolution映射的Axiel-attention block
    :param
        in_planes: 傳入的通道數量。
        planes: 欲輸出的通道數量。輸出維度會受到*expansion，ex:如果expansion=2，輸出維度則為planes*2
        stride: attention block中的步長。如果stirde=2則輸出H、W減半
        downsample: downsample函數。當attention block輸出通道與原本影像不同時會調用。
                包含一個conv1x1(self.in_planes, planes * block.expansion, stride)和norm_layer(planes * block.expansion)
        group: attention head數量。預設為1
        base_width: attention block輸入的通道擴張倍數。如果設置為128, 則attention時使用的通道數則為2*planes
        dilation: pass
        norm_layer: 使用的標準化層
        kernel_size: 
    :returns
        if input size is (B,C,H,W), 
        return (B, planes*2, H // stride, W // stride)
    '''
    def __init__(self,in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=2, norm_layer=None, kernel_size=56):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if dilation != 2:
        #     self.expansion = dilation
        width = int(planes * (base_width / 64.))  # 傳入axial_attention的維度
        self.conv_down = conv1x1(in_planes, width)
        self.height_block = Axialattention_conv(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = Axialattention_conv(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                  width=True)  # 這一步會讓H,W減半
        self.bn1 = norm_layer(width)
        self.conv_up = conv1x1(width, planes * self.expansion) # 輸出維度*2
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.height_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        # print('===finish once axielblock===')
        return out

class medt_retrofit_model(nn.Module):

    def __init__(self, block, block_2, layers, num_classes=1, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=256, imgchan=3, global_cnn=None):

        # ========!!!注意!!!如果要修改訓練影像大小，需要調整img_size的大小!!!========
        super(medt_retrofit_model, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 2),
                                       dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
        #                                dilate=replace_stride_with_dilation[2])

        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        # self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        # self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust1 = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=1, stride=1, padding=0)
        self.adjust2 = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=1, stride=1, padding=0)
        self.adjust3 = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)


        self.soft = nn.Softmax(dim=1)

        self.conv1_p = nn.Conv2d(imgchan, self.inplanes, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)

        self.relu_p = nn.ReLU(inplace=True)

        img_size_p = img_size // 4

        self.layer1_p = self._make_layer(block_2, int(128 * s), layers[0], kernel_size=(img_size_p // 2))
        self.layer2_p = self._make_layer(block_2, int(256 * s), layers[1], stride=2, kernel_size=(img_size_p // 2),
                                         dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(block_2, int(512 * s), layers[2], stride=2, kernel_size=(img_size_p // 4),
                                         dilate=replace_stride_with_dilation[1])
        self.layer4_p = self._make_layer(block_2, int(1024 * s), layers[3], stride=2, kernel_size=(img_size_p // 8),
                                         dilate=replace_stride_with_dilation[2])

        # Decoder
        self.decoder1_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

        self.decoderf = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust_p = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)
        self.global_cnn = global_cnn
        if self.global_cnn is not None:
            self.global_branch = use_global_branch(self.global_cnn, pretarin=False)
        self.init_conv = nn.Conv2d(16*3, 3, 1, bias=False) # 測試local部分加速運算


    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, groups=self.groups, downsample=downsample,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        xin = x.clone()  # B,C,H,W
        # print(xin.shape)
        B, _, H, W = xin.shape
        x = self.conv1(x) # stirde=2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # x = F.max_pool2d(x,2,2)
        x = self.relu(x) # 16, 8, 64, 64
        # x = self.maxpool(x)
        if self.global_cnn is not None:
            x = self.global_branch(x)
            assert x.shape == torch.Size([B, 16, 128, 128]), f'x.shape is {x.shape}, not (b,16,128,128)'
        else:
            x1 = checkpoint(self.layer1, x)
            # print('x1.shape',x1.shape) # 16, 32, 64, 64
            x2 = checkpoint(self.layer2, x1)
            # print('x2.shape',x2.shape) # 16, 64, 32, 32
            # x3 = self.layer3(x2)
            # # print(x3.shape)
            # x4 = self.layer4(x3)
            # # print(x4.shape)
            # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='bilinear'))
            # x = torch.add(x, x4)
            # x = F.relu(F.interpolate(self.decoder2(x4) , scale_factor=(2,2), mode ='bilinear'))
            # x = torch.add(x, x3)
            # x = F.relu(F.interpolate(self.decoder3(x3) , scale_factor=(2,2), mode ='bilinear'))
            # x = torch.add(x, x2)
            x = F.relu(F.interpolate(checkpoint(self.decoder4, x2), scale_factor=(2, 2), mode='bilinear',align_corners=True))
            x = torch.add(x, x1) # 8, 32, 64, 64
            x = F.relu(F.interpolate(checkpoint(self.decoder5, x), scale_factor=(2, 2), mode='bilinear',align_corners=True))
            # print(x.shape) 8, 16, 128, 128

        # end of full image training
        # y_out = torch.ones((1,2,128,128))
        x_loc = x.clone()
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='bilinear'))
        def patch_attention(patches, xin):
            # start
            H_len, W_len = int(patches**-0.5), int(patches**-0.5)
            for i in range(0, H_len):
                for j in range(0, H_len):
                    h, w = H // H_len, W // W_len # assume H,W = 256
                    x_p = xin[:, :, h * i:h * (i + 1), w * j:w * (j + 1)]
                    # begin patch wise
                    x_p = self.conv1_p(x_p)
                    x_p = self.bn1_p(x_p)
                    # x = F.max_pool2d(x,2,2)
                    x_p = self.relu(x_p)

                    x_p = self.conv2_p(x_p)
                    x_p = self.bn2_p(x_p)
                    # x = F.max_pool2d(x,2,2)
                    x_p = self.relu(x_p)
                    x_p = self.conv3_p(x_p)
                    x_p = self.bn3_p(x_p)
                    # x = F.max_pool2d(x,2,2)
                    x_p = self.relu(x_p)

                    # x = self.maxpool(x)
                    x1_p = self.layer1_p(x_p)
                    x2_p = self.layer2_p(x1_p)
                    x3_p = self.layer3_p(x2_p)
                    x4_p = self.layer4_p(x3_p)

                    x_p = F.relu(F.interpolate(checkpoint(self.decoder1_p, x4_p), scale_factor=(2, 2), mode='bilinear'))
                    x_p = torch.add(x_p, x4_p)
                    x_p = F.relu(F.interpolate(checkpoint(self.decoder2_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                    x_p = torch.add(x_p, x3_p)
                    x_p = F.relu(F.interpolate(checkpoint(self.decoder3_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                    x_p = torch.add(x_p, x2_p)
                    x_p = F.relu(F.interpolate(checkpoint(self.decoder4_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                    x_p = torch.add(x_p, x1_p)
                    x_p = F.relu(F.interpolate(checkpoint(self.decoder5_p, x_p), scale_factor=(2, 2), mode='bilinear'))
                    x_loc[:, :, h * i:h * (i + 1), w * j:w * (j + 1)] = x_p
            return x_loc
        def _local_attention_ver2(patches, xin, x_loc=None):
            '''
            Notice: 比較原版_local_attention減少兩層迴圈，將patches整合單一特徵中
                    增加一層init_conv(1x1 conv)調整維度。16*3 -> 3
                    速度比原版_local_attention快更多，平均0.025178154，相較原版0.371444146快14.75263601倍

            學習patch注意力，包含注意力層前的埢積部分
            Parameter:
                xin: B,3,H,W，是輸入的原影像
            '''
            from einops import rearrange

            _, _, H, W = xin.shape
            H_len, W_len = int(patches ** 0.5), int(patches ** 0.5)
            xin = rearrange(xin, 'b c (l h) (l2 w) -> b (l l2 c) h w', l=4, l2=4)
            x_p = self.init_conv(xin)
            # begin patch wise
            x_p = self.conv1_p(x_p)
            x_p = self.bn1_p(x_p)
            # x = F.max_pool2d(x,2,2)
            x_p = self.relu(x_p)
            x_p = self.conv2_p(x_p)
            x_p = self.bn2_p(x_p)
            # x = F.max_pool2d(x,2,2)
            x_p = self.relu(x_p)
            x_p = self.conv3_p(x_p)  # B,64,H/8,H/8
            x_p = self.bn3_p(x_p)
            # x = F.max_pool2d(x,2,2)
            x_p = self.relu(x_p)

            # x = self.maxpool(x)
            x1_p = self.layer1_p(x_p)
            x2_p = self.layer2_p(x1_p)
            x3_p = self.layer3_p(x2_p)
            x4_p = self.layer4_p(x3_p)

            x_p = F.relu(
                F.interpolate(self.decoder1_p(x4_p), scale_factor=(2, 2), mode='bilinear', align_corners=False))
            x_p = torch.add(x_p, x4_p)
            x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(2, 2), mode='bilinear', align_corners=False))
            x_p = torch.add(x_p, x3_p)
            x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(2, 2), mode='bilinear', align_corners=False))
            x_p = torch.add(x_p, x2_p)
            x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(2, 2), mode='bilinear', align_corners=False))
            x_p = torch.add(x_p, x1_p)
            x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(2, 2), mode='bilinear', align_corners=False))
            x_p = rearrange(x_p, 'b (l l2 c) h w -> b c (l h) (l2 w)', l=4, l2=4)

            return x_p

        x_loc = patch_attention(16, xin)
        # x_loc = _local_attention_ver2(16, xin)

        # if self.multiple_features:
        #     x_loc_2 = patch_attention(9, xin)
        #     x_loc += x_loc_2
        x = torch.add(x, x_loc)

        x = F.relu(self.decoderf(x)) # 兩次relu沒有效果，已去除
        x = self.adjust1(x)
        x = self.adjust2(x)
        x = self.adjust3(x)


        # 試著最後加入softmax
        # x = F.softmax(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def medt_retrofit_model_use(args, pretrained=False, **kwargs):
    model = medt_retrofit_model(AxialBlock_conv_dynamic,
                                AxialBlock_wopos,
                                [1, 2, 4, 1],
                                img_size=args.imgsize,
                                num_classes=args.classes,
                                s=0.125,
                                global_cnn=None,
                                **kwargs)
    return model

class use_global_branch(nn.Module):
    def __init__(self, module_name, pretarin=False, s=0.125):
        super(use_global_branch, self).__init__()
        self.modules_name = module_name
        pretarin = 'imagenet' if pretarin == True else None
        if module_name == 'self_def_cnn':
            self.global_encoder1 = global_cnn(8, out_plane=32)  # 8, 32
            self.global_encoder2 = global_cnn(32, out_plane=64, downsample=True)

            self.decoder4 = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
            self.decoder5 = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        if module_name == 'resnet18':
            self.model = smp.UnetPlusPlus(module_name, in_channels=8, classes=16, encoder_weights=pretarin)
            self.upsample = nn.Upsample(scale_factor=2)
        if module_name == 'resnet34':
            self.model = smp.UnetPlusPlus(module_name, in_channels=8, classes=16, encoder_weights=pretarin)
            self.upsample = nn.Upsample(scale_factor=2)
        if module_name == 'resnet52':
            self.model = smp.UnetPlusPlus(module_name, in_channels=8, classes=16, encoder_weights=pretarin)
            self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        'input: b,8,64,64  output: b,16,128,128'
        _, C,H,W = x.shape
        assert C == 8 and H == 64 and W == 64, 'input must be b,8,64,64'
        if self.modules_name == 'self_def_cnn':
            x1 = self.global_encoder1(x)
            x2 = self.global_encoder2(x1)

            x = F.relu(F.interpolate(checkpoint(self.decoder4, x2), scale_factor=(2, 2), mode='bilinear'))
            x = torch.add(x, x1) # 8, 32, 64, 64
            x = F.relu(F.interpolate(checkpoint(self.decoder5, x), scale_factor=(2, 2), mode='bilinear'))
            return x

        if self.modules_name == 'resnet18':
            x = self.model(x)
            x = self.upsample(x)
            return x

        if self.modules_name == 'resnet34':
            x = self.model(x)
            x = self.upsample(x)
            return x

        if self.modules_name == 'resnet52':
            x = self.model(x)
            x = self.upsample(x)
            return x




if __name__ == '__main__':
    imgsize = 128
    import argparse
    parser = argparse.ArgumentParser(description='No global attention Model')
    parser.add_argument('-is', '--imgsize', type=int, default=128, help='圖片大小')
    parser.add_argument('-ic', '--classes', type=int, default=1, help='訓練影像通道數')
    parser.add_argument('-b', '--batchsize', type=int, default=4, help='batchsize')
    parser.add_argument('-mn', '--modelname', default='medt_retrofit')
    parser.add_argument('--device', default='cuda', help='是否使用GPU訓練')
    args = parser.parse_args()

    # test whole model
    def model_test():
        # 只測試model
        test_input = torch.randn(8, 3, imgsize, imgsize).cuda()
        model = medt_retrofit_model_use(args).cuda()
        output = model(test_input)
        print('model_test_output:',output.shape)

    model_test()


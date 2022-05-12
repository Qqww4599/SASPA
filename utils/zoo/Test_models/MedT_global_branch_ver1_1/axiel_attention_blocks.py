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
    def __init__(self, in_plane, out_plane, downsample=False, upsample=False, mid_dim=32):
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
        self.upsample = upsample
    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.final_encoder(x)
        x = self.relu(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')
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
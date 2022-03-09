import torch
import pdb
import math
import sys
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict

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


class medt_retrofit(nn.Module):

    def __init__(self, block, block_2, layers, num_classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=256, imgchan=3, multiple_features=True):

        # ========!!!注意!!!如果要修改訓練影像大小，需要調整img_size的大小!!!========
        super(medt_retrofit, self).__init__()
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
        self.adjust = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
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
        self.multiple_features = multiple_features
        # Decoder
        self.decoder1_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

        self.decoderf = nn.Conv2d(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust_p = nn.Conv2d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

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
        _, _, H, W = xin.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # x = F.max_pool2d(x,2,2)
        x = self.relu(x)

        # x = self.maxpool(x)
        # pdb.set_trace()
        x1 = checkpoint(self.layer1, x)
        # print('x1.shape',x1.shape)
        x2 = checkpoint(self.layer2, x1)
        # print('x2.shape',x2.shape)
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
        x = F.relu(F.interpolate(checkpoint(self.decoder4, x2), scale_factor=(2, 2), mode='bilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(checkpoint(self.decoder5, x), scale_factor=(2, 2), mode='bilinear'))
        # print(x.shape)

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
        x_loc = patch_attention(16, xin)
        if self.multiple_features:
            x_loc_2 = patch_attention(9, xin)
            x_loc += x_loc_2
        x = torch.add(x, x_loc)
        # pdb.set_trace()
        # x = F.relu(self.decoderf(x))
        x = F.sigmoid(self.decoderf(x))

        x = self.adjust(F.relu(x))

        return x

    def forward(self, x):
        return self._forward_impl(x)

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

def medt_retrofit_model(args, pretrained=False, **kwargs):
    model = medt_retrofit(AxialBlock_conv_dynamic, AxialBlock_wopos, [1, 2, 4, 1], img_size=args.imgsize, num_classes=args.imgchan, s=0.125, multiple_features=True,**kwargs)
    return model

if __name__ == '__main__':
    imgsize = 256
    import argparse
    parser = argparse.ArgumentParser(description='Testers')
    parser.add_argument('-is', '--imgsize', type=int, default=256, help='圖片大小')
    parser.add_argument('-ic', '--imgchan', type=int, default=2, help='訓練影像通道數')
    parser.add_argument('-b', '--batchsize', type=int, default=4, help='batchsize')
    parser.add_argument('-mn', '--modelname', default='unet++_resnet34')
    parser.add_argument('--device', default='cuda', help='是否使用GPU訓練')
    args = parser.parse_args()
    # test for axialblock
    def axialblock_test():
        stride = 2
        in_plane = 32
        out_plane = 32
        img_size = 256
        kernal_size = 128
        block_expansion = 2
        # downsample為了改變輸入維度，輸出維度為block輸出的維度，同時stride也要改
        downsample = conv1x1(3, out_plane*2 * block_expansion, stride) # 更改block裡面redidual block的維度需要使用!!
        # stride表示輸出大縮減的倍數、輸入維度必須和輸入圖片相同，kernel_size大小必須和輸入影像大小一致
        # downsample必須使用，不然輸入維度會出錯
        # block輸出dims為plane的兩倍。
        block = AxialBlock_dynamic(in_plane, out_plane, stride, downsample=downsample, kernel_size=img_size)
        # conv_att = Axialattention_conv(in_plane, out_planes=out_plane, groups=8, kernel_size=img_size, stride=2, width=True) # axielblock_output.shape: torch.Size([1, in_planes, 256, 256])
        conv_block = AxialBlock_conv_dynamic(3, out_plane*2, stride=stride, kernel_size=img_size, downsample=downsample) # 如果不設定dilation，channels會兩倍(因為expansion)增加，長寬不會變。

        a = torch.randn(args.batchsize, 3, args.img_size, args.img_size)
        test4att = torch.randn(1, in_plane, img_size, img_size)
        # print('preatt:',test4att.shape)
        # conv_att_output = conv_att(a)
        # print('proatt:',conv_att_output.shape)
        o = conv_block(a)
        # block_o = block(a)
        # o = block(a)
        # 輸入影像：(1,3,256,256) ->  stride=1, plane=32 -> (1, 64, 256, 256)
        # 輸入影像：(1,3,256,256) ->  stride=2, plane=32 -> (1, 64, 128, 128)
        print('conv_axielblock_output.shape:',o.shape)

    # test whole model
    def model_test():
        # 只測試model
        test_input = torch.randn(4, 3, imgsize, imgsize).cuda()
        model = medt_retrofit_model(args).cuda()
        output = model(test_input)
        print('model_test_output:',output.shape)


    # print('origin_axielblock_output.shape:',axielblock_output[1])

    # 測試block
    # axialblock_test()
    # 測試model
    model_test()
    #
    # print(axielblock_output, sep='\n')


# ====================================================================
# from torch import nn
# import torch
# m = nn.Conv2d(3,16,kernel_size=7,padding=3,stride=2)
# x = torch.randn(256,1,64,256)
# y = torch.randn(256,1,64,256)
# o = torch.cat([x,y],dim=-1)
# print(o.shape)
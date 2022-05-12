import torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../..')
from MainResearch.utils.zoo.Test_models.Reverse_AttentionUNet.axialattn import AxialAttention

import xml.etree.ElementTree


def extract(elem, tag, drop_s):
    text = elem.find(tag).text
    if drop_s not in text: raise Exception(text)
    text = text.replace(drop_s, "")
    try:
        return int(text)
    except ValueError:
        return float(text)
def GPU_status():
    '''查看GPU使用量'''
    i = 0

    d = OrderedDict()
    d["time"] = time.time()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    if d["gpu_util"] < 15 and d["mem_used"] < 2816:
        msg = 'GPU status: Idle \n'
    else:
        msg = 'GPU status: Busy \n'

    now = time.strftime("%c")
    print('\n\nUpdated at %s\n\nGPU utilization: %s %%\nVRAM used: %s %%\n\n%s\n\n' % (
        now, d["gpu_util"], d["mem_used_per"], msg))

# ------- Model structure -------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        # self.upsample = nn.functional.upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(2*channel, 1, 1)

    def forward(self, x1, x2):
        x1_1 = x1
        x2_1 = self.conv_upsample1(
            F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        ) * x2

        x2_2 = torch.cat((x2_1, self.conv_upsample4(
            F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)
        )), 1)
        x2_2 = self.conv_concat2(x2_2)

        x = F.interpolate(self.conv4(x2_2),scale_factor=(2,2),mode='bilinear', align_corners=True)
        x = self.conv5(x)
        # 1,64,64
        return x

class net_in_net(nn.Module):
    def __init__(self):
        super(net_in_net, self).__init__()

    def forward(self,x):
        B, C, H, W = x.shape
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x4_mlp = x4.reshape(B, C, -1) # b,c,h,w
        x4 = self.mlp(x4_mlp) + x4
        x4 = x4.reshape(B, C, H, W)

        x3 = self.decoder(x4) + x3
        x2 = self.decoder(x3) + x2
        x1 = self.decoder(x2) + x1
        return x1



class Reverse_attn_unet(nn.Module):
    def __init__(self, channel=32):
        super(Reverse_attn_unet, self).__init__()
        # --- ResNet backbone ---
        self.resnet = models.resnet50(pretrained=True)
        # --- Reception field ---
        self.feature_branch1 = RFB_modified(512, channel)
        self.feature_branch2 = RFB_modified(1024, channel)
        self.feature_branch3 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        # --- reverse attention branch ---
        self.ra4_attn = AxialAttention(2048, heads=8, dim_index=1)
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_attn = AxialAttention(1024, heads=8, dim_index=1)
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_attn = AxialAttention(512, heads=8, dim_index=1)
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.apply(self.weight_init)

    @staticmethod
    # Initialize weight method 初始化方法
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # 4, 64, 32, 32

        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # 4, 256, 32, 32
        x2 = self.resnet.layer2(x1)  # 4, 512, 16, 16
        x3 = self.resnet.layer3(x2)  # bs, 1024, 8, 8
        x4 = self.resnet.layer4(x3)  # bs, 2048, 4, 4
        x2_rfb = self.feature_branch1(x2)  # channel -> 32
        x3_rfb = self.feature_branch2(x3)  # channel -> 32
        x4_rfb = self.feature_branch3(x4)  # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8,
                                      mode='bilinear', align_corners=True)  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(lateral_map_5, scale_factor=0.25, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1)
        x = self.ra4_attn(x)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32,
                                      mode='bilinear', align_corners=True)  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1)
        x = self.ra3_attn(x)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16,
                                      mode='bilinear', align_corners=True)  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1)
        x = self.ra2_attn(x)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8,
                                      mode='bilinear', align_corners=True)  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


if __name__ == '__main__':
    model = models.resnet50()
    x = torch.randn(3,3,128,128)
    a = x.clone()
    # x = model.conv1(x)
    # x = model.bn1(x)
    # x_layer1 = model.layer1(x) # torch.Size([4, 256, 64, 64])
    # x_layer2 = model.layer2(x_layer1) # torch.Size([4, 512, 32, 32])
    # x_layer3 = model.layer3(x_layer2) # torch.Size([4, 1024, 16, 16])
    # x_layer4 = model.layer4(x_layer3) # torch.Size([4, 2048, 8, 8])
    # print(x_layer1.shape, x_layer2.shape, x_layer3.shape, x_layer4.shape
    #       , sep='\n')
    ras = Reverse_attn_unet().to('cuda')
    out = ras(a.to('cuda'))

    from collections import OrderedDict
    import subprocess
    import sys
    import time
    import xml.etree.ElementTree

    print(out[0].shape)

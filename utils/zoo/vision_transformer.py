import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

'''
原始碼來源：https://github.com/asyml/vision-transformer-pytorch/blob/main/src/model.py
Pytorch version of Vision Transformer (ViT) with pretrained models. 
This is part of CASL (https://casl-project.github.io/) and ASYML project.
'''
class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out

class MlpBlock_2D(nn.Module):
    """
    MlpBlock_2D。
    用於2D image用的MLP
    Transformer Feed-Forward Block
    """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock_2D, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape # b, h * w, c

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3) # b,n,h,h_dim ->  b,h,n,h_dim
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3) # out.shape=(b,n,h,h_dim)

        out = self.out(out, dims=([2, 3], [0, 1]))
        # print(f'SelfAttention out：{out.shape}')
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # LinearGeneral(輸出2D影像) torch.Size = (B,C,H,W)
        h,w = 512,512
        self.to_2D = LinearGeneral()

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer

    最多只能還原成(n, c, gh, gw)的形式(因為被convolution過)，可以考慮使用deconvolution的方式還原輸出影像(n, c, h, w)
    """
    def __init__(self,
                 image_size=(256, 256),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=256,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size
        self.gh, self.gw = h // fh, w // fw
        num_patches = self.gh * self.gw

        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # deconvolution還原影像
        self.deconv = nn.ConvTranspose2d(in_channels=768,out_channels=3, kernel_size=patch_size, stride=patch_size,)
        mid_dim = 16
        self.MLP_linear1 = nn.Linear(3, mid_dim)
        self.MLP_linear2 = nn.Linear(mid_dim, 3)
        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # classfier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)     # (n, c, gh, gw)
        # print(f'embedding：{emb.shape}')
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        # cls_token = self.cls_token.repeat(b, 1, 1)
        # emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)
        # print(f'transformer out：{feat.shape}')
        # 假設本來輸入為(b,32,32,3)
        feat = rearrange(feat, 'b (h w) c->b h w c', h=self.gh, w=self.gw)
        # print(f'rearrange out：{feat.shape}')
        out = self.deconv(feat.permute(0, 3, 2, 1)) # b h w c -> b c h w
        out = out.permute(0, 2, 3, 1) # b c h w -> b h w c
        out = self.MLP_linear1(out)
        out = self.MLP_linear2(out)
        out = out.permute(0, 3, 2, 1) # b h w c -> b c h w

        return out
        # classifier
        # logits = self.classifier(feat[:, 0])
        # return logits

if __name__ == '__main__':
    model = VisionTransformer(image_size=(512,512),num_layers=2)
    x = torch.randn((1, 3, 512, 512))
    out = model(x)

    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))


    # from einops import rearrange
    # '''head = 8, in_dim=768, feat_dim=96'''
    # x = torch.randn(1,1024,768)
    # m = LinearGeneral((768,), (8, 96))
    # m2 = LinearGeneral((8, 96), (768,))
    # out = m(x, dims=([2], [0])) # torch.Size([1, 1024, 8, 96])
    # out2 = m2(out, dims=([2, 3], [0, 1]))
    # out2 = rearrange(out2,'b (h w) c->b h w c',h=32,w=32)
    # print(out2.shape) # torch.Size([1, 1024, 768])




import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.utils.checkpoint import checkpoint_sequential


def conv1x1(inplane, outplane, stride=1):
    '''1x1 convolution'''
    return torch.conv2d(inplane, outplane, kernel_size=1, stride=stride, bias=False)


class multiheadattntion(nn.Module):
    def __init__(self, embed_dim, num_head, dropout=0.5, bias=True, add_bias_kv=False, device='cuda:0'):
        super(multiheadattntion, self).__init__()
        self.multiheadattenrion = nn.MultiheadAttention(embed_dim, num_head, dropout, bias, add_bias_kv, device)
        self.flatten = nn.Flatten()


    def forward(self, query, key, value):
        return self.multiheadattenrion(query, key, value)

class convolution_token_embedding(nn.Module):
    '''
    input:
        token_map: (B, C, H, W)
    '''
    def __init__(self, in_channel, out_channnel):
        super(convolution_token_embedding, self).__init__()


        self.conv1 = nn.Conv2d(in_channel, out_channnel, stride=1, kernel_size=3, padding=1) #padding填充上下左右兩列
        self.flatten = nn.Flatten(2,3)
        self.LN = nn.LayerNorm([1])
        self.rearange = Rearrange('b c n -> b n c')



    def forward(self, token_map):
        x = self.conv1(token_map)
        x = self.flatten(x)
        x = self.rearange(x)
        x = self.LN(x)

        return x

class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = (7,7)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None


    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x

class Attention(nn.Module):
    '''
    attention module
    目前問題：使用參數量過大

    '''

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        del dots # 降低顯存占用
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out



'''-------------------------------------------'''
# from skimage import io
#
#
# path = r"D:\Programming\AI&ML\(Dataset)Gland Segmentation in Colon Histology Images Challenge\dataset\images\testA_1.bmp"
#
# y = torch.tensor(io.imread(path),dtype=torch.float32)
# y = torch.unsqueeze(y,0) # 增加batch維度
# y = torch.einsum('b h w c -> b c h w', y).to(torch.device('cuda'))
#
#
# # print(y.shape)
# # f = nn.Flatten(0)
# # x = f(y)
#
#
#
# # model = convolution_token_embedding(3, 5)
# m = ConvEmbed().to(torch.device('cuda'))
# y = m(y)
#
# # print(y.shape, sep='\t')
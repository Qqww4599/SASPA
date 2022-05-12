from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()*thetas.cos(), thetas.sin()*thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 3).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 3)
        :param input_flatten               (N, \sum_{l=0}^{L-1} D_l \cdot H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 3), [(D_0, H_0, W_0), (D_1, H_1, W_1), ..., (D_{L-1}, H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, D_0*H_0*W_0, D_0*H_0*W_0+D_1*H_1*W_1, D_0*H_0*W_0+D_1*H_1*W_1+D_2*H_2*W_2, ..., D_0*H_0*W_0+D_1*H_1*W_1+...+D_{L-1}*H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} D_l \cdot H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1] * input_spatial_shapes[:, 2]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 3)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 3:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 0], input_spatial_shapes[..., 2], input_spatial_shapes[..., 1]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = ms_deform_attn_core_pytorch_3D(value, input_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output

def ms_deform_attn_core_pytorch_3D(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([T_ * H_ * W_ for T_, H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    # sampling_grids = 3 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (T_, H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, T_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)[:,None,:,:,:]
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_.to(dtype=value_l_.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)[:,:,0]
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=[64, 64, 64], temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, d, h, w = x.shape
        mask = torch.zeros(bs, d, h, w, dtype=torch.bool).cuda()
        assert mask is not None
        not_mask = ~mask
        d_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            d_embed = (d_embed - 0.5) / (d_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (3 * (dim_tx // 3) / self.num_pos_feats[0])

        dim_ty = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (3 * (dim_ty // 3) / self.num_pos_feats[1])

        dim_td = torch.arange(self.num_pos_feats[2], dtype=torch.float32, device=x.device)
        dim_td = self.temperature ** (3 * (dim_td // 3) / self.num_pos_feats[2])

        pos_x = x_embed[:, :, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, :, None] / dim_ty
        pos_d = d_embed[:, :, :, :, None] / dim_td

        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 0::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 0::2].cos()), dim=5).flatten(4)
        pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 0::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_d, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        return pos


def build_position_encoding(mode, hidden_dim):
    N_steps = hidden_dim // 3
    if (hidden_dim % 3) != 0:
        N_steps = [N_steps, N_steps, N_steps + hidden_dim % 3]
    else:
        N_steps = [N_steps, N_steps, N_steps]

    if mode in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {mode}")

    return position_embedding
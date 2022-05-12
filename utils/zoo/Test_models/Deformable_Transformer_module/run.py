import numpy as np
import torch
from torch import nn

import ms_deform_attn as attn
import DeformableTrans

src = torch.randn(1,192,6,64,64).to('cuda')
mask = torch.randint(0, 2, (1, 6, 64, 64)).to('cuda').bool()
pos_m = attn.build_position_encoding('v2', hidden_dim=192)
pos = pos_m(src).to('cuda')

srcs, masks, pos_s = torch.zeros_like(src).unsqueeze(0), \
                     torch.zeros_like(mask).unsqueeze(0), \
                     torch.zeros_like(pos).unsqueeze(0)

for lvl, fea in enumerate(range(3)):
    if lvl > 0:
        srcs = torch.vstack((srcs, src.unsqueeze(0)))
        masks = torch.vstack((masks, mask.unsqueeze(0)))
        pos_s = torch.vstack((pos_s, pos.unsqueeze(0)))



m = DeformableTrans.DeformableTransformer(d_model=192,
                                          dim_feedforward=512,
                                          dropout=0.1,
                                          activation='gelu',
                                          num_feature_levels=3,
                                          nhead=4,
                                          num_encoder_layers=1,
                                          enc_n_points=4).to('cuda')
o = m(srcs=srcs, masks=masks, pos_embeds=pos_s)

'此配置可以跑了，要更動要標註'
print(o.shape) # 1, 16, 32, 32

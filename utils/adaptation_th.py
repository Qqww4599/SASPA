import torch
import cv2
from torchvision import transforms
import os
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import argparse
import warnings
import pdb


'''
影像處理系統
:二值化 用自適應二值化
:input b,1,h,w. torch.tensor
:output b,1,h,w. ndarray


更新紀錄: 
    utils ver 1.0
        新增自適應二值化方法 
'''


def THRESH_BINARY_for_pred(x, return_tensor=False):
    b,c,h,w = x.shape
    if x.ndim == 4:
        pred = x.squeeze(0) # 去掉batch維度
        x = pred.permute(1,2,0) # switch to H,W,C
    if torch.is_tensor(x):
        # print('Input shape：', x.shape)
        x = torch.sigmoid(x)
        x = x.numpy()
    x = x*255
    x = x.astype(np.uint8)

    # 用全局自適應(Otsu’s二值化)
    # ret, th = cv2.threshold(x[:,:,-1], th, 255, cv2.THRESH_OTSU)
    img = x[:, :, -1]
    img = cv2.medianBlur(img, 5)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 2)

    if th.ndim == 2:
        th = np.expand_dims(th, axis=2)
    if return_tensor:
        th = th.reshape((b,c,h,w))
        return torch.tensor(th)
    return th
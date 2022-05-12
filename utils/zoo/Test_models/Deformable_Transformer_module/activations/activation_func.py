import torch
from torch import nn
import torch.nn.functional as F
import math

'''
原始碼來自
A survey on recently proposed activation functions for Deep Learning
https://arxiv.org/abs/2204.02921
'''
def Swish(x, beta=0.05):
    out = x * torch.sigmoid(beta*x)
    return out

def Mish(x):
    '''
    中文注釋資料來源/原文網址: https://kknews.cc/zh-tw/code/63g4993.html
    以上無邊界(即正值可以達到任何高度)避免了由於封頂而導致的飽和。理論上對負值的輕微允許允許更好的梯度流，而不是像ReLU中那樣的硬零邊界。
    最後，可能也是最重要的，目前的想法是，平滑的激活函數允許更好的信息深入神經網絡，從而得到更好的準確性和泛化。

    '''
    out = x*(torch.tanh(F.softplus(x)))
    return out

def GCU(x):
    out = x * torch.cos(x)
    return out

# ------ 生物學啟發的震盪激活函數 -------
'''讓激活函數學習XOR函數'''
def SQU(x):
    'Shifted Quadratic Unit'
    out = x**2 + x
    return out

def NCU(x):
    'Non-Monotonic Cubic Unit'
    out = x - x**3
    return out

def DSU(x):
    'Decaying Sine Unit'
    out = math.pi/2 * (torch.sin(x-math.pi)/x - torch.sin(x+math.pi)/x)
    return out

def SSU(x):
    'Shifted Sinc Unit'
    out = math.pi * torch.sin(x-math.pi)/x
    return out


x = torch.randn(10,10)
print(SSU(x))
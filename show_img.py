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
import math
from utils import Other_utils as OU

'''
此模組主要功能是「輸出模型切割效果和GT之間的差異」
param:
    batch_size：傳入參數batch_size預設為1暫時不可更改，表示傳入1張影像(在training中代表從dataloader讀取的batchsize，現在應該沒有作用才對)
    model_name：表示模型所在位置，格式需要是.pth或.pt，必須為絕對路徑
    test_image_input：傳入模型的原始影像位置，必須為絕對路徑
    mask：與test_image_input對應的mask檔愛位置，必須為絕對路徑
'''
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def THRESH_BINARY_for_mask(x, th):
    '''影像二值化。輸入x為(H,W,C)的ndarray，th為閥值'''
    # print('THRESH_BINARY Input shape：',x.shape)
    if torch.is_tensor(x):
        # print('Input shape：', x.shape)
        x = x.numpy()
    # 如果最後維度不是channel(ex:C,H,W)，需要改成H,W,C。
    # 判定方式:如果channel不是3(RGB)或1(GRAY)
    if x.shape[-1] not in (1,3):
        x = np.transpose(x, [1,2,0])
        if x.shape[-1] == 1:
            return x
    # H,W,C 變成 H,W
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # print('Input shape：', x.shape) # Input shape： (512, 512)
    # print('Input：', x)
    ret, th = cv2.threshold(x*255, th, 255, cv2.THRESH_BINARY_INV)
    return th

def THRESH_BINARY_for_pred(x, th=85):
    if torch.is_tensor(x):
        # print('Input shape：', x.shape)
        x = torch.sigmoid(x)
        x = x.numpy()
    else:
        x = x
    x = cv2.normalize(x, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # x = x.astype(np.uint8)
    # img = x[:, :, -1]
    img = x[:, :]
    # ========== 用全局自適應(Otsu’s二值化) =============
    # ret, th = cv2.threshold(x[:,:,-1], th, 255, cv2.THRESH_OTSU)
    # img = cv2.medianBlur(img, 5)
    # th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 2)
    # ==============================================
    # 指定閥值二值化(th=0.333)
    th = (img > th).astype(np.float)
    # ==============================
    if th.ndim == 2:
        pass
        # th = np.expand_dims(th, axis=2)
    return th
def Save_image(*image,save_path,original_size, channel=2, resize=True):
    '''
    input：預期傳入圖片為3張(未來可能會推廣到更多張顯示)，處理前的圖片+處理後的圖片+GT,格式為torch.tensor
    input size to be (B,C,H,W)
    用plt.imshow顯示。
    '''
    warnings.filterwarnings("ignore", module="matplotlib\..*") # supress warning
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)  # supress warning

    matplotlib.use('Agg') #不要顯示圖片
    original_image, pred, mask, *_ = image
    original_size = (original_size[0].item(),(original_size[1].item()))  # (H,W),use item() change tensor to int

    if pred.ndim == 4:
        pred = pred.squeeze(0) # 去掉batch維度
    if mask.ndim == 4:
        mask = mask.squeeze(0) # 去掉batch維度
    if original_image.ndim == 4:
        original_image = original_image.squeeze(0) # 去掉batch維度
    pred, mask = pred.permute(1,2,0), THRESH_BINARY_for_mask(mask, 1) # switch to H,W,C
    # 自適應二值化
    bi_pred = THRESH_BINARY_for_pred(pred)

    original_image = original_image.permute(1,2,0)

    # ---- to torch tensor to numpy array ----
    original_image = original_image.numpy() if torch.is_tensor(pred) else original_image
    bi_pred = bi_pred.numpy() if torch.is_tensor(bi_pred) else bi_pred
    # bi_pred = bi_pred.transpose(2,0,1)
    # ---- resize to original size ----
    if resize:
        original_image = cv2.resize(original_image,original_size,interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred,original_size,interpolation=cv2.INTER_NEAREST)
        bi_pred = cv2.resize(bi_pred,original_size,interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        # print(pred.shape) # H,W,C
    if channel == 2:
        pred = pred[:,:,-1]
    # 用plt.imshow()顯示影像，用plt.imshow()傳入影像必須為C,H,W
    fig = plt.figure()

    fig.add_subplot(2,2,1)
    plt.xticks([]), plt.yticks([])  # 關閉座標刻度
    plt.axis('off')
    plt.title('original')  # 1*3的圖片 的 第1張
    plt.imshow(original_image)

    fig.add_subplot(2, 2, 3)
    plt.xticks([]), plt.yticks([])
    plt.axis('off')  # 關閉座標刻度
    plt.title('model pred')
    plt.imshow(pred)

    fig.add_subplot(2, 2, 2)
    plt.xticks([]), plt.yticks([])  # 關閉座標刻度
    plt.axis('off')
    plt.title('Ground Truth')  # 1*23的圖片 的 第3張
    plt.imshow(original_image, alpha=0.5)
    plt.imshow(mask, alpha=0.5)

    fig.add_subplot(2, 2, 4)
    plt.xticks([]), plt.yticks([])  # 關閉座標刻度
    plt.axis('off')
    plt.title('pred binary')  # 1*23的圖片 的 第3張
    plt.imshow(original_image, alpha=0.5)
    plt.imshow(bi_pred, alpha=0.5)

    plt.savefig(save_path)
    plt.clf()
    plt.close('all')

if __name__ == '__main__':
    # 輸入圖像位置
    parser = argparse.ArgumentParser(description='輸出模型切割效果和GT之間的差異')
    parser.add_argument('--model_path', required=False, default=r'./model_pth/num2_model.pt')
    # parser.add_argument('--epoch', default=3, type=int, help='需要跑的輪數')
    # parser.add_argument('--train_dataset', required=True, type=str, help='訓練資料集位置')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--modelname', default='Transformer_Test')
    # parser.add_argument('--device', default='cuda')
    parser.add_argument('--test_image_input', required=False, type=str, help='傳入模型的原始影像位置，必須為路徑')
    parser.add_argument('--mask', type=str, required=False, help='傳入mask原始影像位置，必須為路徑')
    # parser.add_argument('--save_freq',default=1,help='多少個epoch儲存一次checkpoint')
    # parser.add_argument('--save_state_dict', type=bool, default=True, help='是否只儲存權重，默認為權重')
    # parser.add_argument('--loss_fn',default='weight_cross_entropy', choices=['weight_cross_entropy','dice_coef_loss','IoU','FocalLoss'])
    # parser.add_argument('--wce_beta', type=float, default=1e-04, help='wce_loss的wce_beta值，如果wce_loss時需要設定')
    arg = parser.parse_args()

    model_path = arg.model_path

    dataset_path = r'./(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset'
    img_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_1.bmp'
    img_path2 = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_2.bmp'

    # 測試影像的絕對路徑
    img_path1_n = r'D:\Programming\AI&ML\(Dataset)Gland Segmentation in Colon Histology Images Challenge\dataset\images\testA_1.bmp'
    img_path2_n = r'D:\Programming\AI&ML\(Dataset)Gland Segmentation in Colon Histology Images Challenge\dataset\masks\testA_1.bmp'
    def read_image(path):
        '''------------只有單元測試會用到------------'''
        x = cv2.imread(path)
        # print(f'{filesname}.shape:{x.shape}')
        (B, G, R) = cv2.split(x)
        x = torch.tensor(cv2.merge([R, G, B]))
        return x
    pre_process, post_process = read_image(img_path1_n), read_image(img_path2_n)


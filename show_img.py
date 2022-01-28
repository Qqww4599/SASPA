import torch
import cv2
from torchvision import transforms
import os
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import argparse
from utils.zoo.vision_transformer import VisionTransformer
import warnings

'''
此模組主要功能是「輸出模型切割效果和GT之間的差異」
param:
    batch_size：傳入參數batch_size預設為1暫時不可更改，表示傳入1張影像(在training中代表從dataloader讀取的batchsize，現在應該沒有作用才對)
    model_name：表示模型所在位置，格式需要是.pth或.pt，必須為絕對路徑
    test_image_input：傳入模型的原始影像位置，必須為絕對路徑
    mask：與test_image_input對應的mask檔愛位置，必須為絕對路徑
'''



# dataset_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset'
# img_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_1.bmp'
# img_path2 = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_2.bmp'

def import_model(model_path):
    '''
    :param model_path, path_like，傳入模型位置:
    :return:
    '''

    model_path = model_path
    model = VisionTransformer()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def add_batch_dim(x):
    '''增加batch維度，但目前沒作用'''
    if x.ndim == 3:
        return x
    elif x.ndim == 4:
        x = x.unsqueeze(0)
        return x

def THRESH_BINARY(x, th):
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

def resize(image,originHW):
    '''
    :param image: input image(ex: image, pred, mask), size:(H,W,C)
    :param originHW: tuple, (H,W)
    :return: image
    '''
    import cv2
    image = cv2.resize(image,originHW,interpolation=cv2.INTER_NEAREST)
    return image

# 不能隨便resize圖片會跑掉
def read_image(path):
    x = cv2.imread(path)
    filesname = os.path.split(path)[-1]
    # print(f'{filesname}.shape:{x.shape}')
    (B,G,R) = cv2.split(x)
    x = torch.tensor(cv2.merge([R,G,B]))
    return x

def Show_image(*image,):
    '''
    input：預期傳入圖片為2張(未來可能會推廣到更多張顯示)，處理前的圖片+處理後的圖片
    格式為torch.tensor
    用plt.imshow顯示。
    '''
    image, mask, *_ = image
    # 影像二值化。image1表示原本影像，image2表示mask影像
    # image1, image2 = THRESH_BINARY(image1,1), THRESH_BINARY(image2, 1)
    # print(image2.shape)
    image, mask = image, THRESH_BINARY(mask, 1)
    # original = read_image(arg.test_image_input)

    # 用plt.imshow()顯示影像，用plt.imshow()傳入影像必須為C,H,W
    plt.subplot(1, 3, 1)
    plt.xticks([]), plt.yticks([])  # 關閉座標刻度
    plt.axis('off')
    plt.title('original')  # 1*3的圖片 的 第1張
    plt.imshow(image)

    plt.subplot(1, 3, 2)  # 1*3的圖片 的 第2張
    plt.xticks([]), plt.yticks([])
    plt.axis('off')  # 關閉座標刻度
    plt.title('original\n(will change to model output)')
    plt.imshow(mask)

    plt.subplot(1, 3, 3)
    plt.xticks([]), plt.yticks([])  # 關閉座標刻度
    plt.axis('off')
    plt.title('Ground Truth')  # 1*23的圖片 的 第3張
    plt.imshow(mask)

    plt.show()

def Save_image(*image,save_path,original_size):
    '''
    input：預期傳入圖片為3張(未來可能會推廣到更多張顯示)，處理前的圖片+處理後的圖片+GT,格式為torch.tensor
    input size to be (B,C,H,W)
    用plt.imshow顯示。
    '''
    warnings.filterwarnings("ignore", module="matplotlib\..*") # supress warning
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)  # supress warning

    matplotlib.use('Agg') #不要顯示圖片
    original_image,pred, mask, *_ = image
    original_size = (original_size[0].item(),(original_size[1].item()))  # (H,W),use item() change tensor to int
    # print(original_size)
    # 影像二值化。image1表示原本影像，image2表示mask影像
    # image1, image2 = THRESH_BINARY(image1,1), THRESH_BINARY(image2, 1)
    # print('original_image.shape',original_image.shape) # torch.Size([1, 3, 256, 256])
    # print('mask',mask.shape) # mask:torch.Size([1, 1, 256, 256])
    # print('pred',pred.shape) # pred:torch.Size([1, 3, 256, 256])
    # 有時plt吃CHW有時候吃HWC?????
    if pred.ndim == 4:
        pred = pred.squeeze(0) # 去掉batch維度
    if mask.ndim == 4:
        mask = mask.squeeze(0) # 去掉batch維度
    if original_image.ndim == 4:
        original_image = original_image.squeeze(0) # 去掉batch維度
    pred, mask = pred.permute(1,2,0), THRESH_BINARY(mask, 1) # switch to HWC
    original_image = original_image.permute(1,2,0)
    # print(pred) # H,W,C

    # resize to original size
    original_image = cv2.resize(original_image.numpy(),original_size,interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize(pred.numpy(),original_size,interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    # print(pred) # H,W,C
    # 用plt.imshow()顯示影像，用plt.imshow()傳入影像必須為C,H,W
    plt.subplot(1, 3, 1)
    plt.xticks([]), plt.yticks([])  # 關閉座標刻度
    plt.axis('off')
    plt.title('original')  # 1*3的圖片 的 第1張
    plt.imshow(original_image)

    plt.subplot(1, 3, 2)  # 1*3的圖片 的 第2張
    plt.xticks([]), plt.yticks([])
    plt.axis('off')  # 關閉座標刻度
    plt.title('model pred')
    plt.imshow(pred)

    plt.subplot(1, 3, 3)
    plt.xticks([]), plt.yticks([])  # 關閉座標刻度
    plt.axis('off')
    plt.title('Ground Truth')  # 1*23的圖片 的 第3張
    plt.imshow(mask)

    plt.savefig(save_path)

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

    test: str = input('輸入是否為外來影像(y/n)')
    if test == 'y':
        dataset_path = r'./(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset'
        img_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_1.bmp'
        img_path2 = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_2.bmp'

        # 測試影像的絕對路徑
        img_path1_n = r'D:\Programming\AI&ML\(Dataset)Gland Segmentation in Colon Histology Images Challenge\dataset\images\testA_1.bmp'
        img_path2_n = r'D:\Programming\AI&ML\(Dataset)Gland Segmentation in Colon Histology Images Challenge\dataset\masks\testA_1.bmp'
        # print(img_path,img_path2,sep='\n')
    else:
        sys.exit()

    pre_process, post_process = read_image(img_path).numpy(), read_image(img_path2).numpy()
    pre_process, post_process = read_image(img_path1_n), read_image(img_path2_n)
    Show_image(pre_process, post_process) # 模組化調度使用

    # test_mask,test_image_input = load_image_mask(arg.test_image_input, arg.mask)
    # Show_image(test_mask,test_image_input)

    #用plt.imshow()顯示影像
    # plt.subplot(1, 2, 1)  # 1*2的圖片 的 第1張
    # plt.xticks([]),plt.yticks([])
    # plt.axis('off')# 關閉座標刻度
    # plt.title('pre_processing')
    # plt.imshow(test_mask)
    #
    # plt.subplot(1, 2, 2)
    # plt.xticks([]), plt.yticks([]) # 關閉座標刻度
    # plt.axis('off')
    # plt.title('post_processing')  # 1*2的圖片 的 第2張
    # plt.imshow(test_image_input)
    #
    # plt.show()


import torch
from skimage import io
import matplotlib.pyplot as plt
import warnings
import matplotlib
from utils.Dataloader_breastUS import ImageToImage2D
from utils.Use_model import Use_model
import torchvision
import torchvision.transforms.transforms as T
from torch.utils.data import DataLoader
import configparser
import os
import pdb
import cv2
import numpy as np
from utils.loss_fn import *
from utils.Dataloader_breastUS import Image2D
import sys
import torch.nn.functional as F
import time

import argparse

# 設定傳入參數。測試相關的儲存位置、測試集位置、model位置都在config設定即可
config = {'mp':"./TotalResult_HAND/20220222/test1/best_model.pth",
          'mn':'medt' ,
          'tdp': "D:/Programming/AI&ML/(Dataset)breast Ultrasound lmage Dataset/archive/val_ds2/images",
          }
'''
主要測試model使用的檔案

進行測試時需要修改val_config內配置。包含模型名稱modelname、
model_path
modelname
test_dataset_path = D:\Programming\AI&ML\(Dataset)STU-Hospital

scale = 1 pred輸出是否要經過sigmoid scale
save_path = 輸出影像儲存路徑
classes = 1 模型輸出影像通道數
show_image = 是否顯示影像

已棄用: 
binarization_th = 111 進行二值化之閥值
binarization = 0 是否進行二值化
'''
def args_parser(config):
    parser = argparse.ArgumentParser()
    with open('./val_config.ini') as fp:
        # source code from https://blog.csdn.net/wozaiyizhideng/article/details/107821713
        cfg = configparser.ConfigParser()
        cfg.read_file(fp)
        section_list = cfg.sections() # 讀取段落, 此處段落包含：model_set
        # print(section_list)
        for section in section_list:
            options_list = cfg.options(section)
            items_list = cfg.items(section)
            # print(options_list, items_list, sep='\t')
        model_path = cfg.get('model_set', 'model_path') # 獲取config文件中model_set的model_path值
        modelname = cfg.get('model_set', 'modelname')
        test_dataset_path = cfg.get('model_set', 'test_dataset_path')
        binarization_th = cfg.getint('model_set', 'binarization_th')
        classes = cfg.getint('model_set', 'classes')
        binarization = cfg.getboolean('model_set', 'binarization') # 取得bool
        scale = cfg.getboolean('model_set', 'scale') # 取得bool
        save_path = cfg.get('model_set', 'save_path')
        show_image = cfg.getboolean('model_set', 'show_image')


    parser.add_argument('-mp','--model_path', type=str, default=model_path)
    parser.add_argument('-mn','--modelname', default=modelname, type=str)
    parser.add_argument('--batchsize', default=1, type=int)
    parser.add_argument('-tdp','--test_dataset_path', type=str, default=test_dataset_path, help='測試資料集位置')
    parser.add_argument('--load_state_dict', type=bool, help='是否只載入權重，默認載入權重')
    parser.add_argument('--mode', type=bool, help='是否只載入權重，默認載入權重')
    parser.add_argument('--save_path', type=str, default=save_path, help='圖片儲存位置')
    parser.add_argument('-is', '--imgsize', type=int, default=128, help='圖片大小')
    parser.add_argument('--imgchan', type=int, default=3, help='')
    parser.add_argument('-ic', '--classes', type=int, default=classes, help='model輸出影像通道數(grayscale)')
    parser.add_argument('--device', default='cuda', help='是否使用GPU訓練')
    parser.add_argument('--save_result', default=True, type=bool, help='是否save影像')
    # mask傳入設定
    parser.add_argument('--ds_mask', type=bool, default=True, help='資料集是否含有mask')
    parser.add_argument('--ds_mask_gray', type=bool, default=True, help='mask是否輸出灰階')
    # pred輸出設定
    parser.add_argument('--scale', type=bool, default=scale, help='pred輸出是否要經過sigmoid scale')
    parser.add_argument('--binarization', type=bool, default=binarization, help='')
    parser.add_argument('--binarization_th', type=int, default=binarization_th, help='')
    parser.add_argument('--adp_bi', type=int, default=binarization_th, help='自適應二值化')
    parser.add_argument('--deep_supervise', type=bool, default=False, help='使用深層監督')
    parser.add_argument('--show_image', type=bool, default=show_image, help='show_image')
    parser.add_argument('--save_pred_binary', type=bool, default=True, help='')


    args = parser.parse_args()

    return args
# model_name = args.modelname



def read_image_to_tensor(img_path):
    '現有版本中torchvision只能讀RGB三通道影像'
    img = torchvision.io.read_image(img_path)
    return img

class test_dataloader(DataLoader):
    '''
    建立test的dataloader
    載入測試圖片格式：
    -test_dataset
        -images
            -image1
            -image2
            ...
        -masks
            -mask1
            -mask2
            ...
        ...
    '''
    def __init__(self, valid_ds, args=None):
        super().__init__(valid_ds)
        img_ds = os.path.join(valid_ds,'images')
        if args.ds_mask:
            mask_ds = os.path.join(valid_ds, 'masks')
            self.masks_list = [os.path.join(mask_ds, file) for file in os.listdir(mask_ds)]
        self.images_list = [os.path.join(img_ds, file) for file in os.listdir(img_ds)]
        self.imgsize = args.imgsize
        self.device = args.device
        self.ds_mask = args.ds_mask
        self.ds_mask_gray = args.ds_mask_gray
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, index):
        image = cv2.imread(self.images_list[index])
        original_img_size = image.shape  # 原始image大小，H,W,C
        mask_out, original_mask_size = torch.zeros(original_img_size), original_img_size
        # resize完後改回原本model能接受的尺寸(B,C,H,W)
        image = cv2.resize(image, (self.imgsize,self.imgsize), interpolation=cv2.INTER_NEAREST)
        # 輸入影像值為[0-1]之間，不然輸出結果會異常
        image = np.transpose(image, (2, 0, 1)) / 255.
        # print('Test dataset size：', image.shape,sep='\n')
        # 回傳沒有經過資料增量的影像+原圖尺寸(tuple)
        img_out = torch.tensor(image).to(self.device).to(torch.float32)
        if self.ds_mask:
            if not self.ds_mask_gray:
                mask = cv2.imread(self.masks_list[index])
            mask = cv2.imread(self.masks_list[index], 0)
            original_mask_size = mask.shape  # 原始mask大小，H,W,C
            mask = cv2.resize(mask, (self.imgsize,self.imgsize), interpolation=cv2.INTER_NEAREST)
            mask = mask / 255
            mask_out = torch.tensor(mask).to(self.device).to(torch.int)
            mask_out = mask_out.unsqueeze(0) # to match metrics fn

        return img_out, mask_out, (original_img_size,original_mask_size)

def calculate(model_out, mask):
    '''注意輸入的影像數值是否為[0,255]'''
    iou = IoU(model_out, mask.cuda())
    f1_s = classwise_f1(model_out, mask.cuda(), testing=True)  # 這邊有用
    m_dice = DiceLoss()
    Dice = 1 - dice_loss(mask, model_out // 255, )
    # print('f1:', f1_s, 'IoU:', iou)
    return f1_s.cpu().detach().numpy().astype(float), iou.cpu().detach().numpy(), Dice.cpu().detach().numpy()

def adaptiveThreshold(img):
    img = img[:,:,0].numpy() * 255
    img = img.astype(np.uint8)
    assert type(img)==np.ndarray, f'{type(img)}'
    img = cv2.medianBlur(img,5)
    th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,255,2)

    return th1,th2

def main():

    sys.path.append(r'D:\Programming\AI&ML\Main_Research\train_ver2.py')
    from train_ver2 import eval

    args = args_parser(config)
    dataloader = test_dataloader(args.test_dataset_path, args=args)
    dataloader = DataLoader(dataloader)

    save = True # 顯示圖片或儲存圖片
    # 載入模型
    model = Use_model(args)
    model.load_state_dict(torch.load(args.model_path)) # 載入權重
    ds_mask = args.ds_mask
    binarization_th = args.binarization_th / 255.

    f1_final, iou_final, Dice_final = 0., 0., 0.

    infer_time = 0
    for i, (image ,mask, size) in enumerate(dataloader):
        assert type(image) == torch.Tensor, f'correct type is torch.Tensor, now is {type(image)}'
        assert image.shape == (args.batchsize, 3, args.imgsize, args.imgsize), f'correct:{image.shape}, should be (1,3,h,w)'# confirm input format
        graph_num = 2
        # print(args.scale, args.binarization, args.binarization_th)
        time_start = time.time()
        pred = model(image.cuda())
        infer_time += time.time() - time_start
        if args.scale:
            pred = sigmoid_scaling(pred) # 使用sigmoid歸一化
        if args.binarization:
            pred = (pred > binarization_th).float() # 影像二值化
        if ds_mask:
            # 先經過自適應二值化
            pred = pred.to('cpu').detach().squeeze(0).permute(1, 2, 0)  # h,w,2
            mean_th, gussan_th = adaptiveThreshold(pred)
            # gussan_th = 255 - gussan_th
            # pred[:,:,0] = 255 - pred[:,:,0]

            f1, iou, Dice = calculate(torch.tensor(gussan_th).reshape(1,1,args.imgsize, args.imgsize).cuda(), mask)
            f1_final += f1
            iou_final += iou
            Dice_final += Dice
            graph_num += 1
        if args.adp_bi:
            image = image.to('cpu').detach().squeeze(0).permute(1, 2, 0)  # h,w,3
            # mean_th, gussan_th = adaptiveThreshold(pred)
            plt.subplot(221)
            plt.xticks([]), plt.yticks([])  # 關閉座標刻度
            plt.axis('off')
            plt.title('GT')  # 1*3的圖片 的 第1張
            plt.imshow(mask.cpu().reshape(128,128))

            plt.subplot(222)
            plt.xticks([]), plt.yticks([])  # 關閉座標刻度
            plt.axis('off')
            plt.title('binary_blend')  # 1*3的圖片 的 第1張，影像融合
            plt.imshow(gussan_th, alpha=0.5)
            plt.imshow(image, alpha=0.5)
            if args.save_pred_binary:
                f_name = os.path.join(args.save_path, 'pred_file', f'{i} pred.png')
                cv2.imwrite(f_name, gussan_th)

            plt.subplot(223)
            plt.xticks([]), plt.yticks([])  # 關閉座標刻度
            plt.axis('off')
            plt.title('image')  # 1*3的圖片 的 第1張
            plt.imshow(image)

            plt.subplot(224)
            plt.xticks([]), plt.yticks([])  # 關閉座標刻度
            plt.axis('off')
            plt.title('pred')  # 1*3的圖片 的 第1張
            plt.imshow(pred[:,:,0])

            save_path = os.path.join(args.save_path, f'{i}')
            plt.savefig(save_path)
            if args.show_image:
                plt.show()
            continue

    # print('test_avg_f1:{:4f}'.format(f1_final/(i+1)),',', 'test_avg_iou:{:4f}'.format(iou_final/(i+1)))
    print('F1 score TEST data: ',f1_final/(i+1),',', 'mIoU TEST data:', iou_final/(i+1), f'Dice TEST data: {Dice_final/(i+1)}')
    print('avg inferance time:', infer_time/(i+1),)

if __name__ == '__main__':
    main()
import os, stat
import shutil
import torch
import cv2
import numpy as np
import sys

def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

def init_training_result_folder(path):
    '''初始化並創建資料夾'''
    files = os.listdir(path)
    for file in files:
        cur_path = os.path.join(path,file)
        if os.path.isdir(cur_path):
            shutil.rmtree(cur_path, onerror=remove_readonly)
        else:
            os.remove(cur_path)
    if not os.path.exists(f'{path}/log'):
        os.makedirs(f'{path}/log')
    if not os.path.exists(f'{path}/test_files'):
        os.makedirs(f'{path}/test_files')

def save_model_mode(model, model_name):
    '''
    儲存model權重
    '''
    # 儲存模型
    save_name = model_name
    torch.save(model.state_dict(), save_name)
    print(f"1 epoch finish!!! {model_name} are saved!", sep='\t')

def THRESH_BINARY_for_pred(x, return_tensor=False):
    '''
    validation時使用。predication image二值化。
    目前使用方法是:
    medianBlur + adaptiveThreshold

    :return Torch.tensor or ndarray
    '''

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

    # 用全局自適應GAUSSIAN
    # ret, th = cv2.threshold(x[:,:,-1], th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    for ch in range(c):
        img = x[:, :, ch]
        img = cv2.medianBlur(img, 5)
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 2)
        x[:, :, ch] = th
    if x.ndim == 2:
        x = np.expand_dims(x, axis=2)
    if return_tensor:
        x = x.transpose((2,0,1))
        x = np.expand_dims(x, axis=0)
        return torch.tensor(x)
    return x

def Double_check_training_setting():
    reject_choices = {'no','n',}
    direc_check = input('確認儲存路徑是否正確(Y/N)?').lower()
    if 'n' in direc_check:
        print('再次確認並重新開始測試')
        sys.exit()
    model_check = input('確認模型是否正確(Y/N)?').lower()
    if 'n' in model_check:
        print('再次確認並重新開始測試')
        sys.exit()
    loss_check = input('確認loss函數設定是否正確(Y/N)?').lower()
    if 'n' in loss_check:
        print('再次確認並重新開始測試')
        sys.exit()
    meta_check = input('確認config meta是否正確設定，確認save valid fig、imgchan、epoch是否正確設定(Y/N)?').lower()
    if 'n' in meta_check:
        print('再次確認並重新開始測試')
        sys.exit()
    return None


if __name__ == '__main__':
    Double_check_training_setting()
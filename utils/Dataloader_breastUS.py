import os
import numpy as np
import torch

from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision import transforms
import pdb

from typing import Callable
from typing import Container

import os
import cv2
import pandas as pd
import argparse


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, crop=(32, 32), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = F.to_tensor(image)
        # pdb.set_trace()

        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask


class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, Gray=False,
                 merge_train:bool=True, img_size=(256,256), get_catagory=None, only_positive=True) -> None:

        """
        path example:"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT\benign_new"
        catagory: 是否使用類別
        only_positive: 只使用有腫瘤的影像訓練
        """

        # 如果是merge_train直接輸入包含benign和malignant的路徑。EX：folder：normal_new,malignant_new,benign_new
        # 訓練資料包含benign,malignant,normal，normal的mask全部都是0(代表無病灶標記)。
        # 如果只要訓練有腫瘤部分，需要再另外把範圍縮小到只有benign,malignant
        self.img_size = img_size
        self.dataset_path = dataset_path
        self.gray = Gray
        self.get_catagory = get_catagory
        self.img_catagory = None
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        if merge_train:
            # Normal, Malignant, Benign全部一起訓練
            # benign_ls = os.listdir(os.path.join(self.input_path_benign))
            # malignant_ls = os.listdir(self.input_path_malignant)
            self.images_list, self.masks_list = [],[]
            for catagory in os.listdir(self.dataset_path):
                # 如果需要縮小範圍到只有benign,malignant，再for迴圈後加上if catagory in {benign_new,malignant_new}
                if catagory in {'benign_new', 'malignant_new'}:
                    self.img_catagory = 1
                    # continue
                else:
                    if only_positive:
                        continue
                    self.img_catagory = 0
                catagory_path = os.path.join(self.dataset_path, catagory)
                for folder in os.listdir(catagory_path):
                    if folder == 'images':
                        folder_path = os.path.join(catagory_path,folder)
                        for image in os.listdir(folder_path):
                            # 直接加入images路徑
                            self.images_list.append(os.path.join(folder_path,image))
                    else:

                        folder_path = os.path.join(catagory_path, folder)
                        for mask in os.listdir(folder_path):
                            # 直接加入masks路徑
                            self.masks_list.append(os.path.join(folder_path, mask))
            print(f'Training on {len(self.images_list)} images, categories is {self.get_catagory}')

        else:
            self.input_path = os.path.join(dataset_path, 'images')
            self.output_path = os.path.join(dataset_path, 'masks')
            self.images_list = [os.path.join(self.input_path,i) for i in os.listdir(self.input_path)]
            self.masks_list = [os.path.join(self.output_path,i) for i in os.listdir(self.output_path)]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        images_path, masks_path = self.images_list[idx], self.masks_list[idx]
        # read image (choose RGB 3 channel or Gray 1 channel)
        image = cv2.imread(images_path) if not self.gray else cv2.imread(images_path, 0)
        mask = cv2.imread(masks_path, 0)
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        if self.joint_transform:
            # 原來一直都有呼叫這個函式但是我不知道= =，目前沒被賦予功能。
            image, mask = self.joint_transform(image, mask)

        # image, mask = image.permute(1,2,0).numpy(), mask.permute(1,2,0).numpy()
        image, mask = image.permute(1,2,0).numpy(), mask.permute(1,2,0).numpy()

        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        # 注意：20220504以前訓練的影像都是左轉90度的，以後都改正面訓練
        image = np.expand_dims(image,axis=0) if image.ndim == 2 else np.transpose(image,(2,0,1))
        mask = np.expand_dims(mask,axis=0)

        mask[mask != 0] = 1
        image_filename = os.path.split(images_path)[-1]
        # 注意：此處回傳的image和mask格式為torch.tenser()，維度是(C,H,W)。
        # 如果用Dataloader讀取，回傳格式為torch.tenser()，維度是(B,C,H,W)。

        # print('BreastUS資料集輸出影像尺寸：', image.shape, mask.shape, sep='\n')
        if self.get_catagory is not None:
            return image, mask, self.img_catagory
        return image, mask

class Image2D(DataLoader):
    '''
    資料型態：
    dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...
    '''
    def __init__(self, dataset_path, img_size=(256,256), Gray=False):
        # ========記得調整影像大小!!!!!!!!!!!!!======
        self.img_size = img_size
        self.dataset_path = dataset_path
        self.images_path = os.path.join(self.dataset_path, 'images')
        self.masks_path = os.path.join(self.dataset_path, 'masks')
        self.gray = Gray
        self.images_list = os.listdir(self.images_path) # val影像檔案名稱列表
        self.masks_list = os.listdir(self.masks_path) # val GT影像檔案名稱列表
    def __len__(self):
        # return len(os.listdir(self.dataset_path))
        return len(os.listdir(self.images_path))

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_path,'images',self.images_list[index])
        mask_path = os.path.join(self.dataset_path,'masks',self.masks_list[index])
        image = cv2.imread(image_path) if not self.gray else cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path,0)

        image = cv2.resize(image,self.img_size,interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask,self.img_size, interpolation=cv2.INTER_NEAREST)
        # resize完後改回原本model能接受的尺寸(B,C,H,W)
        image = np.expand_dims(image,axis=0) if image.ndim == 2 else np.transpose(image/255.0,(2,0,1)) # 歸一化
        mask = np.expand_dims(mask,axis=0)
        image, mask = torch.tensor(image,dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        # print('BUS驗證資料集的尺寸：', image.shape, mask.shape, sep='\n')
        # 回傳沒有經過資料增量的影像+原圖尺寸(tuple)
        return image, mask


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)





if __name__ == '__main__':

    import os
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import sys
    #
    #
    dataset_path = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\val_dataset"
    dataset_path_train = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT"
    save_path = r'./model/Model_Result/'
    # train_tf = JointTransform2D(crop=(32,32), color_jitter_params=None)
    # dataset = DataLoader(ImageToImage2D(dataset_path_train,joint_transform=train_tf))
    dataset = DataLoader(ImageToImage2D(dataset_path_train, get_catagory=True, Gray=True))
    v_dataset = DataLoader(Image2D(dataset_path, img_size=(128,128), Gray=True))
    for i, (image, mask) in enumerate(v_dataset):
        if i == 1:
            break
        # print(image.shape, mask.shape, sep='\n')
        if image.ndim or mask.ndim == 4:
            image, mask = image.reshape(*image.shape), mask.reshape(*mask.shape)# 去掉batch維度
            print(image.shape, mask.shape, sep='\n')

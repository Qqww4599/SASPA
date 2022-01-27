import os
import numpy as np
import torch

from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision import transforms

from typing import Callable
import os
import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict
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

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False,
                 merge_train:bool=True, img_size=(256,256)) -> None:
        '''
        path example:"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT\benign_new"

        '''
        # 如果是merge_train直接輸入包含benign和malignant的路徑。EX：folder：normal_new,malignant_new,benign_new
        # 訓練資料包含benign,malignant,normal，normal的mask全部都是0(代表無病灶標記)。
        # 如果只要訓練有腫瘤部分，需要再另外把範圍縮小到只有benign,malignant
        self.img_size = img_size
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        if merge_train:
            # benign_ls = os.listdir(os.path.join(self.input_path_benign))
            # malignant_ls = os.listdir(self.input_path_malignant)
            self.images_list, self.masks_list = [],[]
            for catagory in os.listdir(self.dataset_path):
                # 如果需要縮小範圍到只有benign,malignant，再for迴圈後加上if catagory in {benign_new,malignant_new}
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
            print(f'There are {len(self.images_list)} images')

        else:
            self.input_path = os.path.join(dataset_path, 'images')
            self.output_path = os.path.join(dataset_path, 'masks')
            self.images_list = [os.path.join(self.input_path,i) for i in os.listdir(self.input_path)]
            self.masks_list = [os.path.join(self.output_path,i) for i in os.listdir(self.output_path)]



    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        images_path, masks_path = self.images_list[idx], self.masks_list[idx]
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(images_path)
        # print(image.shape)
        # read mask image
        mask = cv2.imread(masks_path, 0)

        # print(mask,mask.shape,sep='\t')
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # print(image.shape)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # print(image)
        # print(mask)
        image, mask = image.permute(2,1,0).numpy(), mask.permute(2,1,0).numpy()
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print(mask.shape)
        # mask = np.transpose(mask,(2,0,1)) # mask輸出只有1 channel(沒有channel dim)，不需要換維度，未來須修正。
        image = np.transpose(image,(2,0,1))
        mask = np.expand_dims(mask,axis=0)
        # print(image.shape)
        # print(mask.shape)
        image_filename = os.path.split(images_path)[-1]
        # 注意：此處回傳的image和mask格式為torch.tenser()，維度是(H,W,1)。
        # 如果用Dataloader讀取，回傳格式為torch.tenser()，維度是(B,H,W,C)。

        # print('BreastUS資料集輸出影像尺寸：', image.shape, mask.shape, sep='\n')
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
    def __init__(self, dataset_path, img_size=(256,256),single_ch=False):
        # ========記得調整影像大小!!!!!!!!!!!!!======
        self.img_size = img_size
        self.dataset_path = dataset_path
        self.images_path = os.path.join(self.dataset_path, 'images')
        self.masks_path = os.path.join(self.dataset_path, 'masks')
        self.single_ch = single_ch
        self.images_list = os.listdir(self.images_path) # val影像檔案名稱列表
        self.masks_list = os.listdir(self.masks_path) # val GT影像檔案名稱列表
    def __len__(self):
        # return len(os.listdir(self.dataset_path))
        return len(os.listdir(self.images_path))

    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_path,'images',self.images_list[index])
        mask_path = os.path.join(self.dataset_path,'masks',self.masks_list[index])
        image, mask = cv2.imread(image_path), cv2.imread(mask_path,0)
        original_size = (len(image[:,1,1]),len(image[1,:,1])) # H,W,C 取 (H,W)tuple
        image = cv2.resize(image,self.img_size,interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask,self.img_size, interpolation=cv2.INTER_NEAREST)
        # resize完後改回原本model能接受的尺寸(B,C,H,W)
        image = np.transpose(image/255.0,(2,0,1)) # 歸一化
        mask = np.expand_dims(mask,axis=0)
        # print('BUS驗證資料集的尺寸：', image.shape, mask.shape, sep='\n')
        # 回傳沒有經過資料增量的影像+原圖尺寸(tuple)
        return image, mask, original_size


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value / normalize for key, value in self.results.items()}

# def THRESH_BINARY(x, th):
#     '''影像二值化。輸入x為(H,W,C)的ndarray，th為閥值'''
#     # print('THRESH_BINARY Input shape：',x.shape)
#     if torch.is_tensor(x):
#         # print('Input shape：', x.shape)
#         x = x.numpy()
#     # 如果最後維度不是channel(ex:C,H,W)，需要改成H,W,C。
#     # 判定方式:如果channel不是3(RGB)或1(GRAY)
#     if x.shape[-1] not in (1,3):
#         x = np.transpose(x, [1,2,0])
#         if x.shape[-1] == 1:
#             return x
#     # H,W,C 變成 H,W
#     x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
#     # print('Input shape：', x.shape) # Input shape： (512, 512)
#     # print('Input：', x)
#     ret, th = cv2.threshold(x*255, th, 255, cv2.THRESH_BINARY_INV)
#     return th
# def Save_image(*image,save_path):
#     '''
#     input：預期傳入圖片為2張(未來可能會推廣到更多張顯示)，處理前的圖片+處理後的圖片
#     格式為torch.tensor
#     用plt.imshow顯示。
#     '''
#     matplotlib.use('Agg') #不要顯示圖片
#     image1, image2, *_ = image
#     # 影像二值化。image1表示原本影像，image2表示mask影像
#     # image1, image2 = THRESH_BINARY(image1,1), THRESH_BINARY(image2, 1)
#     # print(image2.shape)
#     # 有時plt吃CHW有時候吃HWC?????
#     image1, image2 = image1.permute(1,2,0), THRESH_BINARY(image2, 1) # switch to HWC
#     # image1, image2 = image1, THRESH_BINARY(image2, 1)
#     # original = read_image(arg.test_image_input)
#
#     # 用plt.imshow()顯示影像，用plt.imshow()傳入影像必須為C,H,W
#     plt.subplot(1, 3, 1)
#     plt.xticks([]), plt.yticks([])  # 關閉座標刻度
#     plt.axis('off')
#     plt.title('original')  # 1*3的圖片 的 第1張
#     plt.imshow(image1)
#
#     plt.subplot(1, 3, 2)  # 1*3的圖片 的 第2張
#     plt.xticks([]), plt.yticks([])
#     plt.axis('off')  # 關閉座標刻度
#     plt.title('original\n(will change to model output)')
#     plt.imshow(image1)
#
#     plt.subplot(1, 3, 3)
#     plt.xticks([]), plt.yticks([])  # 關閉座標刻度
#     plt.axis('off')
#     plt.title('Ground Truth')  # 1*23的圖片 的 第3張
#     plt.imshow(image2)
#
#     plt.savefig(save_path)


# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# #
# #
# dataset_path = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\val_dataset"
# save_path = r'./model/Model_Result/'
# dataset = DataLoader(Image2D(dataset_path))
# for i, (image, mask, original_size) in enumerate(dataset):
#     if i == 2:
#         break
#     # print(image.shape, mask.shape, sep='\n')
#     if image.ndim or mask.ndim == 4:
#         image, mask = image.squeeze(0), mask.squeeze(0)# 去掉batch維度
#         print(image, mask, sep='\n')
# dataloader = Image2D(dataset_path)
# dataloader = DataLoader(dataloader)
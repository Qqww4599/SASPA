import os
import torch
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
from einops import rearrange
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np

import matplotlib.pyplot as plt

# dataset_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset'
# img_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_1.bmp'
# classes = os.listdir(dataset_path)
# for one_class in classes:
#     class_path = os.path.join(dataset_path, one_class)
#     for i, image in enumerate(os.listdir(class_path)):
#         if i%20 == 0:
#             print(f'第{i}張圖片路徑為{os.path.abspath(os.path.join(class_path, image))}')

class Make_Dataset(DataLoader):

    '''
    input: 影像路徑，分為"影像資料集"與"mask資料集"，影像資料集名稱必須為images。影像數量和masks數量必須相同

    output: 輸出格式為torch.tenser, 圖像部分：torch.Size=(c,h,w),[3,244,244]，
            mask部分：torch.Size=(c,h,w),[3,244,244]
            如果使用MedT，torch.Size=(c,h,w),[1,244,244]，配合training criterion
    '''

    def __init__(self, dataset_path, img_size=(512,512), joint_transform=None):
        self.img_size = img_size
        self.joint_transform = joint_transform
        self.dataset_pth = dataset_path
        self.img_data = None
        self.mask_data = None
        if len(os.listdir(self.dataset_pth)) > 1:
            for folder in os.listdir(self.dataset_pth):
                if folder == 'images':
                    x = tuple(os.listdir(os.path.join(self.dataset_pth, folder)))
                    self.img_data = [os.path.join(self.dataset_pth, folder, img) for img in x]
                else:
                    x = tuple(os.listdir(os.path.join(self.dataset_pth, folder)))
                    self.mask_data = [os.path.join(self.dataset_pth, folder, img) for img in x]

    def __getitem__(self, index):
        if os.path.exists(self.img_data[index]) is not True:
            return self.img_data[index]
        img_read, mask_read = cv2.imread(self.img_data[index]),cv2.imread(self.mask_data[index])
        # 用cv2修改resolution，可以選用插值法參考INTER_NEAREST,INTER_LANCZOS4,INTER_AREA,INTER_LINEAR,INTER_CUBIC
        img_read, mask_read = cv2.resize(img_read,self.img_size,interpolation=cv2.INTER_NEAREST), cv2.resize(mask_read,self.img_size, interpolation=cv2.INTER_NEAREST)
        mask_read = mask_read.transpose(1,0,2) # H,W,C
        # print(img_read.shape, mask_read.shape)

        if self.joint_transform:
            img_read, mask_read = self.joint_transform(img_read, mask_read)
        mask_read = mask_read.permute(2,0,1) # H,W,C
        # print(img_read.shape, mask_read.shape) # torch.Size([3, 256, 256]) torch.Size([3, 256, 256])
        transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
        img_read, mask_read = transform(img_read), transform(mask_read.to(torch.float))
        # img_read, mask_read = img_read, mask_read # h,w,c to c,h,w
        if self.joint_transform:
            # 代表使用MedT
            mask_read = mask_read.unsqueeze(0)
            _1x1conv = torch.nn.Conv2d(3,1,kernel_size=1)
            mask_read = _1x1conv(mask_read)
            mask_read = mask_read.squeeze(0) # 1,256,256
            # print(mask_read.shape)
        return img_read, mask_read

    def __len__(self):
        return min(len(self.img_data),len(self.mask_data))

    def create_dataset(dataset_path):
        pass


def Pad_resize(x):
    '''
    把影像resize到指定大小的transform物件。

    目前狀況：
        1.運作時維度部分會跑掉
        2.維度跟預期不同
    :return:
    '''
    def _return_Pad_size(paded_h, paded_w, x):
        '''return Pad_size'''
        h, w, c = x.shape

        pd_h, pd_w = h, w
        if paded_h > h:
            pd_h = (paded_h - h) // 2

        elif paded_w > w:
            pd_w = (paded_w - w) // 2

        print(f'pd_h:{pd_h}, pd_w:{pd_w}')
        return pd_w, pd_h

    pd_w, pd_h = _return_Pad_size(1024, 1024, x)
    trans = transforms.Compose([transforms.ToPILImage(),transforms.Pad([pd_w,pd_h]),transforms.ToTensor()])
    return trans

def No_resize():
    '''
    比較Pad_resize()的功能。

    未來不會使用到。
    '''
    trans = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    return trans

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

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from show_img import Show_image

    dataset_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset'
    img_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/images/testA_1.bmp'

    # 直接呼叫Make_Dataset
    # dataset_o = Make_Dataset(dataset_path)
    # i, m = dataset_o[0]
    # print(i.shape, m.shape)

    dataset = DataLoader(Make_Dataset(dataset_path))
    for i, (image, mask) in enumerate(dataset):
        if i == 2:
            break
        # print(image.shape, mask.shape, sep='\n')
        if image.ndim or mask.ndim == 4:
            image, mask = image.squeeze(0), mask.squeeze(0) # 去掉batch維度
        Show_image(image, mask)


#
# # 不能隨便resize圖片會跑掉
# x = torch.tensor(cv2.imread(img_path))
# plt.imshow(x)
# print(f'x.shape:{x.shape}')
# pd_w,pd_h = return_Pad_szie(1024,1024,x)
# trans = transforms.Compose([transforms.ToPILImage(),transforms.Pad([pd_w,pd_h]),transforms.ToTensor()])
#
# x = trans(x).numpy()
# print(x.shape)
# plt.imshow(x)
#
#
# x = torch.randn(1,2,3,4,5)
# print(x.permute(0,2,3,1,4).shape)
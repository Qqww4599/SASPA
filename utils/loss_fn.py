import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss # 用於LogNLLLoss

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

            target = target.view(target.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C
        target = target.reshape(-1,1)
        # print('target：',target.shape)
        logpt = F.log_softmax(input,dim=-1)
        # return logpt, logpt.shape,target, target.shape
        logpt = logpt.gather(1,target.type(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# import torch
#
# m = FocalLoss()
# x = torch.randn(1,3,256,256)
# y = torch.randn(1,3,256,256)
# i1,i2,t1,t2 = m(x,y)
# print(i1,i2,t1,t2,sep='\n')

def IoU(y_true, y_pred, eps=1e-6):
    '''mIoU or IoU ?'''
    # if np.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = torch.sum(y_true * y_pred, dim=(1,2,3))
    union = torch.sum(y_true, dim=(1,2,3)) + torch.sum(y_true, dim=(1,2,3)) - intersection
    return -torch.mean( (intersection + eps) / (union + eps), dim=0)


def dice_coef_loss(y_true, y_pred):
    '''就是Diceloss的算法'''
    def dice_coef(y_true, y_pred, smooth=1):
        '''input：image(b c h w), mask(b c h w), (1,3,512,512)'''
        # y_true, y_pred = y_true.to('cpu'), y_pred.to('cpu')
        # y_true,y_pred = y_true.numpy(), y_pred.numpy()
        multiple = y_true * y_pred
        intersection = torch.sum(multiple, dim=(2, 3)) # shape=(1,3)
        # intersection = np.sum(multiple, axis=[1, 2, 3])
        union = torch.sum(y_true, dim=(2,3)) + torch.sum(y_pred, dim=(2,3))
        return torch.mean((2. * intersection + smooth) / (union + smooth), dim=1)
    out = 1. - dice_coef(y_true, y_pred, smooth=1)
    # print(out, out.grad, sep='\t')
    return out

def weight_cross_entropy(output, target, wce_beta):
    '''

    :param output: 輸入影像(B,H,W,C)
    :param target: 輸入mask(H,W,C) or (B,H,W,C)
    :return: loss(float)
    '''

    def clip_by_tensor(t, t_min, t_max):
        """
        實現與tf.clip_by_value相似功能

        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t = t.float()
        t_min = float(t_min)
        t_max = float(t_max)

        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    output = clip_by_tensor(output, 1e-07, 1 - 1e-07)
    output = torch.log(output / (1 - output))

    # print('output.shape：',output.shape)
    # print('target.shape：',target.shape)
    # one-hot encoding to label (sparse to non-sparse)
    # output *= [0, 1]
    # target *= [0, 1]

    # output *= torch.tensor([0, 1])
    # target *= torch.tensor([0, 1])

    loss = target * -torch.log(torch.sigmoid(output)) * wce_beta + (1 - target) * -torch.log(1 - torch.sigmoid(output))
    loss = torch.sum(loss, axis=0)
    loss = torch.mean(loss)
    return loss

class LogNLLLoss(_WeightedLoss):
    '''來自MedT metrics.py'''
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    # 預設y_input=(b,c,h,w),y_target=(b,1,h,w)，計算方式為攤平後再CE
    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        y_input = y_input.flatten(2)  # 新增
        y_target = y_target.flatten(2).squeeze(1) # 新增
        return cross_entropy(y_input, y_target.long(), weight=self.weight,
                             ignore_index=self.ignore_index)
    # def forward(self, y_input, y_target):
    #     if y_target.dim() == 4:
    #         y_target = y_target.squeeze(0) # target的size需要是(N,H,W), output的channel數量是target的內部值(整數)
    #     # y_input = torch.log(y_input + np.finfo(np.float32).eps)
    #     return cross_entropy(y_input, y_target.long(), weight=self.weight,
    #                          ignore_index=self.ignore_index)

def classwise_iou(output, gt):
    """
    來自MedT metrics.py
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    EPSILON = 1e-32
    dims = (0, *range(2, len(output.shape)))
    gt = gt.squeeze(1)# to (B,H,W)
    gt = gt.type(torch.int64)
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    o = gt.shape
    intersection = output*gt
    union = output + gt - intersection
    # pdb.set_trace()
    iou_loss = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)
    if not len(iou_loss) == 1:
        iou_loss = iou_loss.sum() / len(iou_loss)

    return iou_loss

def classwise_f1(output, gt):
    """
    來自MedT metrics.py
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1

def binary_cross_entropy(x, target):
    '''
    input :
        x: tensor.shape:(1,2,h,w)
        y: tensor.shape:(1,1,h,w)

    '''


    # x = (torch.Size([1, 2, 256, 256])
    *i,h,w = target.shape
    x = x.squeeze(0)
    target = target.resize(h,w) # (torch.Size([1, 256, 256]) to (256,256)
    output = 0
    for i in x:
        x = x.float()
        output = F.binary_cross_entropy_with_logits(i, target.float())
        # output = output / x[0,1]
    return output


if __name__ == '__main__':
    '''
    單元測試階段，測試loss_fn的可行性
    目前問題：
        1. BCE直接使用dataset會產生多個loss，但loss必須要是scalar才行。
        2. WCE無法準確學習到data的資訊，前面的epoch還具有分割效果，到大約3個epoch就會產生梯度爆炸的問題
        3. LogNllloss
    
    '''
    from Dataloader_breastUS import ImageToImage2D
    from torch.utils.data import DataLoader
    from zoo.MedT.lib.models.axialnet import gated
    import argparse

    # dataset_path = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset'
    # mask1 = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/masks/testA_27.bmp'
    # mask2 = r'../(Dataset)Gland Segmentation in Colon Histology Images Challenge/dataset/masks/testA_28.bmp'
    # #
    # us_dataset_image1 = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT\benign_new\images\benign (1).png"
    # us_dataset_mask1 = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT\benign_new\masks\benign (1).png"
    us_dataset = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT"


    parser = argparse.ArgumentParser(description='Transformer Test Version')
    parser.add_argument('-is', '--imgsize', type=int, default=128, help='圖片大小')
    parser.add_argument('-ic', '--imgchan', type=int, default=2, help='訓練影像通道數')
    args = parser.parse_args()

    dataset = ImageToImage2D(us_dataset, img_size=(args.imgsize,args.imgsize))
    DL = DataLoader(dataset, batch_size=4, shuffle=True)
    model = gated(args).cuda()

    for i, (image, mask) in enumerate(DL):
        x = model(image.cuda())
        # assert x.shape == (1,2,256,256), f'模型輸出格式與loss輸入格式不符合，輸入格式為{x.shape},應為(1,2,256,256)'
        # o = weight_cross_entropy(x, mask, wce_beta=1e-07)
        o = LogNLLLoss()(x, mask.cuda())
        print(o)
        if i == 1:
            break



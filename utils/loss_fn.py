import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import matplotlib.pyplot as plt

from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss # 用於LogNLLLoss

'''
loss_fn ver1.0

此loss script作為loss函數選擇使用。預設傳入之特徵(torch.tensor)大小: B,classes,H,W
Focalloss:
dice_coef_loss:
weight_cross_entropy(wce):
LogNLLLoss(lll):
IoU:
Classwise IoU:
classwise_f1:
binary_cross_entropy(bce):


更新紀錄:
    ver1.0
        pass




'''

def Binarization(x, th):
    x = x*255
    x[x>=th] = 1
    x[x<th] = 0
    return x

def scaling(x):
    max = x.max()
    min = x.min()
    dist = max-min
    if dist == 0:
        return x
    return (x - min) / dist

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

def sigmoid_scaling(x):
    x = torch.sigmoid(x) # 用torch內建方法
    return x

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
    out = out.sum() / len(y_true)
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

    # assert output.shape[1] == 1
    assert target.shape[1] == 1
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
    '''
    來自MedT metrics.py
    此loss function限定使用輸出2 class，不然會跳出CUDA error
    '''
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.nll = nn.NLLLoss()

    # 預設y_input=(b,c,h,w),y_target=(b,1,h,w)，計算方式為攤平後再CE
    def forward(self, y_input, y_target):

        # y_input = torch.log(y_input + EPSILON)
        y_input = y_input.flatten(2)  # 新增
        y_target = y_target.flatten(2).squeeze(1) # 新增
        return self.nll(y_input, y_target.long())
        return cross_entropy(y_input, y_target.long(), weight=self.weight,
                             ignore_index=self.ignore_index)
    # def forward(self, y_input, y_target):
    #     if y_target.dim() == 4:
    #         y_target = y_target.squeeze(0) # target的size需要是(N,H,W), output的channel數量是target的內部值(整數)
    #     # y_input = torch.log(y_input + np.finfo(np.float32).eps)
    #     return cross_entropy(y_input, y_target.long(), weight=self.weight,
    #                          ignore_index=self.ignore_index)

def IoU(y_pred ,y_true,  eps=1e-8, threshold=0.333):
    '''IoU
    y_pred:(b,2,h,w)
    y_true:(b,1,h,w)
    '''
    # assert y_true.shape == (args.batchsize, 1, args.imgsize, args.imgsize)
    # assert y_pred.shape == (args.batchsize, 2, args.imgsize, args.imgsize)

    # --- 歸一化+二值化 ---
    y_pred = scaling(y_pred)
    y_pred = (y_pred>threshold).int()

    intersection = y_pred * y_true
    union = y_pred + y_true - intersection
    intersection = intersection.sum(dim=(0,2,3)) # (b,c*h*w), clip by channels, expect shape :[c nums]
    union = union.sum(dim=(0,2,3)) # (b,c*h*w), clip by channels, expect shape :[c nums]
    output = (intersection + eps) / (union + eps)
    output = torch.max(output)
    return output

def classwise_iou(output, gt):
    """
    來自MedT metrics.py
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, 1, image.shape)
    """
    # assert gt.shape == (args.batchsize, 1, args.imgsize, args.imgsize)
    # assert output.shape == (args.batchsize, 2, args.imgsize, args.imgsize)
    EPSILON = 1e-32
    # EPSILON = 0
    dims = (0, *range(2, len(output.shape)))
    gt = gt.squeeze(1)# to (B,H,W)
    gt = gt.type(torch.int64)
    # 調整gt的class數，變成和output相同，(4,2,h,w), 兩個dim為相反陣列()
    # EX：[[1,1,0,0,1,0,1],[0,0,1,1,0,1,0]]
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection
    iou_loss = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)
    if not len(iou_loss) == 1:
        iou_loss = iou_loss.sum() / len(iou_loss)

    return iou_loss

def classwise_f1(output, gt, threshold=0.333):
    """
    來自MedT metrics.py
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    # --- 歸一化+二值化 ---
    output = scaling(output)
        # 設定閥值
    output = (output > threshold).int()

    epsilon = 1e-8
    n_classes = output.shape[1]
    # pdb.set_trace()
    # ---- modified f1 ----
    # -- if want source code
    # to github https://github.com/jeya-maria-jose/Medical-Transformer/blob/main/lib/metrics.py ---
    # output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i+1) * (gt == i+1)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i+1).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i+1).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)
    # print(f'true_positives:{true_positives}',f'selected:{selected}',
    #       f'precision:{precision}',f'recall:{recall}')
    if not len(classwise_f1) == 1:
        classwise_f1 = classwise_f1.sum() / len(classwise_f1)

    return classwise_f1

def binary_cross_entropy(x, target):
    '''
    input :
        x: tensor.shape:(1,2,h,w)
        y: tensor.shape:(1,1,h,w)

    '''


    # x = (torch.Size([1, 2, 256, 256])
    b,c,h,w = target.shape
    x = x[:,-1,:,:].squeeze(0)
    if x.shape == (h,w):
        x = x.unsqueeze(0)
    target = target.reshape(b,h,w) # (torch.Size([1, 256, 256]) to (256,256)
    x = x.float()
    output = F.binary_cross_entropy_with_logits(x, target.float())
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
    from Use_model import Use_model
    from zoo.MedT.lib.models.axialnet import MedT
    import argparse
    import torch
    import segmentation_models_pytorch as smp
    import sys



    # assign parameters
    parser = argparse.ArgumentParser(description='Testers')
    us_dataset = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT"
    test_ds = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\val_ds2"

    parser.add_argument('-is', '--imgsize', type=int, default=128, help='圖片大小')
    parser.add_argument('-ic', '--imgchan', type=int, default=2, help='訓練影像通道數')
    parser.add_argument('-b', '--batchsize', type=int, default=1, help='batchsize')
    parser.add_argument('-mn', '--modelname', default='unet++_resnet34')
    parser.add_argument('--device', default='cuda', help='是否使用GPU訓練')

    args = parser.parse_args()

    # build dataloader and model
    dataset = ImageToImage2D(test_ds, img_size=(args.imgsize,args.imgsize), merge_train=False)
    DL = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    # model = MedT(args).cuda()
    model = Use_model(args)

    # load model from trained model
    model_state = r"D:\Programming\AI&ML\model\TotalResult_HAND\20220223\test1\best_model.pth"
    model.load_state_dict(torch.load(model_state))
    model.eval()

    for i, (image, mask) in enumerate(DL):
        oriout = out = model(image.cuda())

        print(f'輸入影像大小:{image.shape}')
        assert image.shape == (args.batchsize, 3, args.imgsize, args.imgsize), f'correct:{image.shape},'# confirm input format
        assert mask.shape == (args.batchsize, 1, args.imgsize, args.imgsize), f'correct:{mask.shape},' # confirm input format


        out = sigmoid_scaling(out) # 使用sigmoid歸一化
        out_iou = (out > 0.333).float() # 影像二值化
        # out_iou = out
        # pdb.set_trace()
        # assert x.shape == (1,2,256,256), f'模型輸出格式與loss輸入格式不符合，輸入格式為{x.shape},應為(1,2,256,256)'
        # o = weight_cross_entropy(x, mask, wce_beta=1e-07)
        iou_test = IoU(out_iou,mask.cuda())
        iou = classwise_iou(out_iou,mask.cuda())
        nll = LogNLLLoss()(out, mask.cuda())
        f1_s = classwise_f1(out, mask.cuda())
        wce = weight_cross_entropy(oriout, mask.cuda(), 1e-8)

        print('IoU: {:2f}'.format(iou_test))
        print('classwise_iou: {:2f}'.format(iou))
        print('LogNLLLoss:', nll)
        print('classwise_f1:', f1_s)



        out = out.permute(0,3,2,1).to('cpu').detach().numpy()
        oriout = oriout.permute(0,3,2,1).to('cpu').detach().numpy()
        out_iou = out_iou.permute(0,3,2,1).to('cpu').detach().numpy()
        #顯示影像
        # pdb.set_trace()
        plt.subplot(2,2,1)
        plt.axis('off')
        plt.xticks([]),plt.yticks([])
        plt.title('original')
        plt.imshow(image.squeeze(0).permute(2,1,0))

        plt.subplot(2, 2, 2)
        plt.axis('off')
        plt.xticks([]), plt.yticks([])
        plt.title('mask')
        plt.imshow(mask.squeeze(0).permute(2,1,0))

        plt.subplot(2,2,3)
        plt.axis('off')
        plt.xticks([]),plt.yticks([])
        plt.title('output')
        plt.imshow(oriout.squeeze(0)[:,:,1])

        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.xticks([]), plt.yticks([])
        plt.title('output bi')
        plt.imshow(out_iou.squeeze(0)[:, :, 1])

        plt.show()

        if i == 2:
            break
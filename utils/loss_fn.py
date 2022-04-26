from __future__ import absolute_import
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

def Accuracy(predict, target):
    '''
    -------------------Accuracy TEST-------------------
    :input
    predict的二值化影像
    mask

    :return:
    Accuracy
    '''
    classes = torch.max(predict).to(torch.int)
    acc, dice_c = 0, 0
    for cls in range(classes):
        cls = cls+1
        TP = torch.sum((predict == cls) * (target == cls))
        FP = torch.sum(predict == cls) - TP
        FN = torch.sum(target == cls) - TP
        TN = torch.sum((predict != cls) * (target != cls))
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        acc += (TP + TN) / (TP + FP + FN + TN)
        dice_c += (2 * TP + 1e-16) / (torch.sum(predict == cls) + torch.sum(target == cls) +1e-16)
    return acc / (classes+1), dice_c / (classes+1)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-16):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs.to(torch.float))

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def dice_loss(target, predictive, ep=1e-8):
    '''
    陳中銘老師推薦的指標

    '''
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

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
    # def forward(self, y_input, y_target):
    #
    #     # y_input = torch.log(y_input + EPSILON)
    #     y_input = y_input.flatten(2)  # 新增
    #     y_target = y_target.flatten(2).squeeze(1) # 新增
    #     return cross_entropy(y_input, y_target.long(), weight=self.weight,
    #                          ignore_index=self.ignore_index)
    def forward(self, y_input, y_target):
        if y_target.dim() == 4:
            y_target = y_target.squeeze(1) # target的size需要是(N,H,W), output的channel數量是target的內部值(整數)
        # y_input = torch.log(y_input + np.finfo(np.float32).eps)
        return cross_entropy(y_input.float(), y_target.long(), weight=self.weight,
                             ignore_index=self.ignore_index)

def IoU(y_pred ,y_true,  eps=1e-8, threshold=0.333):
    '''IoU
    y_pred:(b,2,h,w)
    y_true:(b,1,h,w)
    '''
    # assert y_true.shape == (args.batchsize, 1, args.imgsize, args.imgsize)
    # assert y_pred.shape == (args.batchsize, 2, args.imgsize, args.imgsize)

    # --- 歸一化+二值化 ---
    y_pred = scaling(y_pred)
    # y_pred = (y_pred>threshold).int()

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

def classwise_f1(output, gt, threshold=0.333, testing=False):
    """
    來自MedT metrics.py
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)

    測試：
    f1的輸入影像在此函數中改為channel=1，如果讓channel>1，會讓f1多0.5?
    """
    n,c,h,w = output.shape
    # print(f'mask: {gt.shape}') # 1,1,h,w
    # --- 歸一化+二值化 ---
    output = scaling(output)
    # 設定閥值，test時設定？
    if testing and c == 1:
        # output = (output > threshold).int()
        n_classes = output.shape[1]
    if c == 1:
        n_classes = output.shape[1]
    if c > 1:
        n_classes = output.shape[1]-1 # 2 - 1


    epsilon = 1e-8
    # output = torch.argmax(output, dim=1) # 1,H,W
    # ---- modified f1 ----
    # -- if want source code
    # to github https://github.com/jeya-maria-jose/Medical-Transformer/blob/main/lib/metrics.py ---
    # output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i+1) * (gt == i+1)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i+1).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i+1).sum() for i in range(n_classes)]).float()

    # pdb.set_trace()
    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)
    # print(f'true_positives:{true_positives}',f'selected:{selected}',
    #       f'precision:{precision}',f'recall:{recall}')
    if not len(classwise_f1) == 1:
        classwise_f1 = classwise_f1.sum() / len(classwise_f1)
    return classwise_f1

def binary_cross_entropy(x, target, _2class=False):
    '''
    input :
        x: tensor.shape:(1,2,h,w)
        y: tensor.shape:(1,1,h,w)

    :parameter
    _2class: 是否使用分層mask輸出(目標層、背景層)。mask.shape=(b,2,h,w)
    '''
    def f_2class(mask, dim=1):
        '''
        return
            mask背景層 = mask[:,1,:,:]
            mask病灶層 = mask[:,0,:,:]
            mask.shape = (b,2,h,w)
            background is last channel!!!!!!!!
        '''
        bg_mask = torch.as_tensor((mask) == 0, dtype=torch.int32)
        mask = torch.cat((mask, bg_mask), dim=dim)
        return mask
    b,c,h,w = target.shape
    x = x # b,c,h,w
    target = target.reshape(b,c,h,w)
    x = x.float()
    if _2class and x.shape[1] != 1:
        target = f_2class(target, dim=1) # (b,2,h,w)
    output = F.binary_cross_entropy_with_logits(x, target.float())
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
    import argparse
    import torch
    import segmentation_models_pytorch as smp
    import sys
    import yaml
    import os

    def parser_args(model_name=None):
        parser = argparse.ArgumentParser(description='test loss function')
        ds_path = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT"
        vds_path = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\val_ds2"

        'yaml test'
        parser.add_argument('--fname', type=str, help='name of config file to load',
                            default=r"D:\Programming\AI&ML\model\config\loss_config.yaml")

        def _process_main(fname):
            import logging, pprint
            logging.basicConfig()
            logger = logging.getLogger()
            params = None
            with open(fname, 'r') as y_file:
                params = yaml.load(y_file, Loader=yaml.FullLoader)
                logger.info('loaded params...')
                pp = pprint.PrettyPrinter(indent=4)
                pp.pprint(params)
            dump = os.path.join(fr'{params["save"]["direc"]}', 'test loss setting.yaml')
            with open(dump, 'w') as f:  # 寫入檔案
                yaml.dump(params, f)
            return params

        fname = r'D:\Programming\AI&ML\model\config\loss_config.yaml'
        args = _process_main(fname)
        print(args['meta']['modelname'])

        # Training parameter setting
        parser.add_argument('--epoch', default=args['optimization']['epochs'], type=int, help='需要跑的輪數')
        parser.add_argument('-bs', '--batchsize', default=args['optimization']['batchsize'], type=int)
        parser.add_argument('-is', '--imgsize', type=int, default=args['data']['imgsize'], help='圖片大小')
        parser.add_argument('-ic', '--imgchan', type=int, default=args['data']['imgchan'], help='使用資料的通道數，預設3(RGB)')
        parser.add_argument('-class', '--classes', type=int, default=args['data']['classes'],
                            help='model輸出影像通道數(grayscale)')
        parser.add_argument('-model', '--modelname', default=args['meta']['modelname'], type=str)
        parser.add_argument('-ds_path', '--train_dataset', default=fr'{args["data"]["ds_path"]}', type=str,
                            help='訓練資料集位置')
        parser.add_argument('-vd', '--val_dataset', type=str, default=args['data']['val_dataset'], help='驗證用資料集所在位置')
        parser.add_argument('--catagory', type=int, default=args['data']['catagory'], help='使用類別資料與否。如果使用，將輸出正常0，有腫瘤1')

        # Model training setting
        parser.add_argument('--device', type=str, default=args['meta']['device'], help='是否使用GPU訓練')
        parser.add_argument('-ds', '--dataset', choices=['BreastUS'], default=args['data']['dataset'],
                            help='選擇使用的資料集，默認GS，預設BreastUS')
        parser.add_argument('--use_autocast', type=bool, default=args['meta']['use_autocast'], help='是否使用混和精度訓練')
        parser.add_argument('--threshold', type=int, default=args['save']['threshold'],
                            help='設定model output後二值化的threshold, 介於0-1之間')
        parser.add_argument('--train_accumulation_steps', default=args['optimization']['train_accumulation_steps'],
                            type=int, help='多少iters更新一次權重(可減少顯存負擔)')
        parser.add_argument('--k_fold', type=int, default=args['optimization']['k_fold'], help='使用k_fold訓練')
        parser.add_argument('--deep_supervise', type=bool, default=args['optimization']['deep_supervise'],
                            help='使用深層監督')
        parser.add_argument('--training', type=bool, default=args['meta']['training'], help='訓練狀態??')


        # Optimizer Setting
        parser.add_argument('--lr', type=float, default=args['optimization']['lr'], help='learning rate')
        parser.add_argument('--scheduler', type=str, default=args['criterion']['scheduler'], help='使用的scheduler')
        parser.add_argument('-opt', '--optimizer', type=str, default=args['criterion']['optimizer'],
                            help='使用的optimizer')

        # Loss function and Loss schedule
        parser.add_argument('-loss', '--loss_fn', type=str, default=args['criterion']['loss'],
                            choices=['wce', 'dice_coef_loss', 'IoU', 'FocalLoss', 'bce', 'lll', 'clsiou'])
        parser.add_argument('-wce', '--wce_beta', type=float, default=1e-04,
                            help='wce_loss的wce_beta值，如果使用wce_loss時需要設定')

        # Save Setting
        parser.add_argument('-sf', '--save_freq', type=int, default=args['save']['save_frequency'],
                            help='多少個epoch儲存一次checkpoint')
        parser.add_argument('--save_state_dict', type=bool, default=args['save']['save_state_dict'],
                            help='是否只儲存權重，默認為權重')
        parser.add_argument('--savemodel', type=bool, default=args['save']['savemodel'], help='是否儲存模型')
        parser.add_argument('-r', '--run_formal', type=bool, default=args['save']['run_formal'],
                            help='是否是正式訓練(if not, train 8 iters for each epoch)')
        parser.add_argument('--direc', type=str, default=args['save']['direc'], help='directory to save')
        parser.add_argument('--savefig_resize', type=bool, default=args['save']['savefig_resize'],
                            help='savefig resize')
        parser.add_argument('--load_state_dict', default=False, type=bool, help='')
        parser.add_argument('--model_state', default=args['save']['model_state'], type=str, help='')

        args = parser.parse_args()

        return args

    # assign parameters
    us_dataset = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT"
    test_ds = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\val_ds2"

    args = parser_args()

    # build dataloader and model
    dataset = ImageToImage2D(test_ds, img_size=(args.imgsize,args.imgsize), merge_train=False)
    DL = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    model = Use_model(args)

    # load model from trained model
    if args.load_state_dict:
        model_state = r"D:\Programming\AI&ML\model\TotalResult_HAND\20220223\test1\best_model.pth"
        model.load_state_dict(torch.load(model_state))
    if not args.training:
        model.eval()

    for i, (image, mask) in enumerate(DL):
        oriout = out = model(image.cuda())

        # print(f'輸入影像大小:{image.shape}')
        assert image.shape == (args.batchsize, 3, args.imgsize, args.imgsize), f'correct:{image.shape},'# confirm input format
        assert mask.shape == (args.batchsize, 1, args.imgsize, args.imgsize), f'correct:{mask.shape},' # confirm input format


        out = sigmoid_scaling(out) # 使用sigmoid歸一化
        out_iou = (out > 0.333).float() # 影像二值化
        # out_iou = out
        assert out.shape == (args.batchsize,args.classes,args.imgsize,args.imgsize), f'模型輸出格式與loss輸入格式不符合，輸入格式為{out.shape},應為(1,2,256,256)'
        # o = weight_cross_entropy(x, mask, wce_beta=1e-07)
        iou_test = IoU(out_iou,mask.cuda())
        iou = classwise_iou(out_iou,mask.cuda())
        nll = LogNLLLoss()(out, mask.cuda())
        f1_s = classwise_f1(out, mask.cuda())
        wce = weight_cross_entropy(oriout, mask.cuda(), 1e-8)
        bce = binary_cross_entropy(out, mask.cuda(), _2class=True)
        dice_loss = dice_loss(out, mask.cuda())

        # print('IoU: {:2f}'.format(iou_test))
        # print('classwise_iou: {:2f}'.format(iou))
        # print('LogNLLLoss:', nll)
        # print('classwise_f1:', f1_s)
        print('dice_loss:', dice_loss)
        print('f1_s:', f1_s)
        print('iou:',  iou)



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
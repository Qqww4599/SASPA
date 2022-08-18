import torch
from torch import nn
import torch.nn.functional as F
"""Code from https://github.com/achaiah/pywick/blob/master/pywick/losses.py, thank you!!"""


class WeightedBCELoss2d(nn.Module):
    def __init__(self, **_):
        super(WeightedBCELoss2d, self).__init__()

    @staticmethod
    def forward(logits, labels, weights, **_):
        w = weights.view(-1)            # (-1 operation flattens all the dimensions)
        z = logits.view(-1)             # (-1 operation flattens all the dimensions)
        t = labels.view(-1)             # (-1 operation flattens all the dimensions)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss


class WeightedSoftDiceLoss(torch.nn.Module):
    def __init__(self, **_):
        super(WeightedSoftDiceLoss, self).__init__()

    @staticmethod
    def forward(logits, labels, weights, **_):
        probs = torch.sigmoid(logits)
        num   = labels.size(0)
        w     = weights.view(num,-1)
        w2    = w*w
        m1    = probs.view(num,-1)
        m2    = labels.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
        score = 1 - score.sum()/num
        return score


class BCEWithLogitsViewLoss(nn.BCEWithLogitsLoss):
    '''
    Silly wrapper of nn.BCEWithLogitsLoss because BCEWithLogitsLoss only takes a 1-D array
    '''
    def __init__(self, weight=None, **_):
        super().__init__(weight=weight, reduction='mean')

    def forward(self, input_, target, **_):
        '''
        :param input_:
        :param target:
        :return:
        Simply passes along input.view(-1), target.view(-1)
        '''
        return super().forward(input_.view(-1), target.view(-1))


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **_):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels, **_):
        num = labels.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)

        # smooth = 1.

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class FocalLoss(nn.Module):
    """
    Weighs the contribution of each sample to the loss based in the classification error.
    If a sample is already classified correctly by the CNN, its contribution to the loss decreases.
    :eps: Focusing parameter. eps=0 is equivalent to BCE_loss
    """
    def __init__(self, l=0.5, eps=1e-6, **_):
        super(FocalLoss, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, labels, **_):
        labels = labels.view(-1)
        probs = torch.sigmoid(logits).view(-1)

        losses = -(labels * torch.pow((1. - probs), self.l) * torch.log(probs + self.eps) + \
                   (1. - labels) * torch.pow(probs, self.l) * torch.log(1. - probs + self.eps))
        loss = torch.mean(losses)

        return loss


# BCEDicePenalizeBorderLoss
class BCEDicePenalizeBorderLoss(nn.Module):
    def __init__(self, kernel_size=55, **_):
        super(BCEDicePenalizeBorderLoss, self).__init__()
        self.bce = WeightedBCELoss2d()
        self.dice = WeightedSoftDiceLoss()
        self.kernel_size = kernel_size

    def to(self, device):
        super().to(device=device)
        self.bce.to(device=device)
        self.dice.to(device=device)

    def forward(self, logits, labels, **_):
        a = F.avg_pool2d(labels, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = torch.ones(a.size()).to(device=logits.device)

        w0 = weights.sum()
        weights = weights + ind * 2
        w1 = weights.sum()
        weights = weights / w1 * w0

        loss = self.bce(logits, labels, weights) + self.dice(logits, labels, weights)

        return loss


# BCEDiceFocalLoss
class BCEDiceFocalLoss(nn.Module):
    '''
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1,1]) Optional weighing (0.0-1.0) of the losses in order of [bce, dice, focal]
    '''
    def __init__(self, focal_param, weights=None, **kwargs):
        if weights is None:
            weights = [1.0,1.0,1.0]
        super(BCEDiceFocalLoss, self).__init__()
        self.bce = BCEWithLogitsViewLoss(weight=None, reduction='mean', **kwargs)
        self.dice = SoftDiceLoss(**kwargs)
        self.focal = FocalLoss(l=focal_param, **kwargs)
        self.weights = weights

    def forward(self, logits, labels, **_):
        return self.weights[0] * self.bce(logits, labels) + self.weights[1] * self.dice(logits, labels) + self.weights[2] * self.focal(logits.unsqueeze(1), labels.unsqueeze(1))


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

    b, c, h, w = target.shape
    x = x  # b,c,h,w
    target = target.reshape(b, c, h, w)
    x = x.float()
    if _2class and x.shape[1] != 1:
        target = f_2class(target, dim=1)  # (b,2,h,w)
    output = F.binary_cross_entropy_with_logits(x, target.float())
    return output


if __name__ == '__main__':
    testdata = torch.randn(1, 1, 128, 128)
    GT = torch.randint(0, 2, (1, 1, 128, 128)).to(torch.float32)
    metrics = BCEDiceFocalLoss(focal_param=0.5)
    metrics1 = binary_cross_entropy
    print(metrics(testdata, GT), metrics1(testdata, GT), )
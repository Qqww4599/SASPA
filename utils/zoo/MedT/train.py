# Code for MedT

import torch
import lib
import argparse
# import torch
# import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# import torch.nn.functional as F
import os
# import matplotlib.pyplot as plt
# import torch.utils.data as data
# from PIL import Image
import numpy as np
# from torchvision.utils import save_image
import torch
# import torch.nn.init as init
# from utils import JointTransform2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
# from utils import chk_mkdir, Logger, MetricList
import cv2
# from functools import partial
# from random import randint
import time

from GS_Dataloader import Make_Dataset
from Dataloader_breastUS import ImageToImage2D
from torch.utils.tensorboard import SummaryWriter



def main(args):
    gray_ = args.gray
    # aug = args.aug
    direc = args.direc
    modelname = args.modelname
    imgsize = args.imgsize

    if gray_ == "yes":
        from utils_gray import JointTransform2D, ImageToImage2D, Image2D
        imgchant = 1
    else:
        from utils import JointTransform2D, ImageToImage2D, Image2D
        imgchant = 3

    if args.crop is not None:
        crop = (args.crop, args.crop)
    else:
        crop = None

    tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
    tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
    # print('args.train_dataset是：',args.train_dataset,'type是：',type(args.train_dataset))
    train_dataset = Make_Dataset(args.train_dataset,img_size=(imgsize,imgsize),joint_transform=tf_train)
    val_dataset = Make_Dataset(args.val_dataset,img_size=(imgsize,imgsize),joint_transform=tf_train)
    predict_dataset = Image2D(args.val_dataset)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, 1, shuffle=True)

    device = torch.device("cuda")

    if modelname == "axialunet":
        model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
    elif modelname == "MedT":
        model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant)
    elif modelname == "gatedaxialunet":
        model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
    elif modelname == "logo":
        model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model,device_ids=[0,1]).cuda()
    model.to(device)

    criterion = LogNLLLoss()

    # 學習率一般調整方法
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                                 weight_decay=1e-5)
    # 學習率動態調整方法：lr_scheduler.OneCycleLR
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,max_lr=args.learning_rate,
    #     cycle_momentum=False,
    #     epochs=args.epochs,
    #     steps_per_epoch=int(np.ceil(len(dataloader) / args.batch_size)),
    # )
    # 學習率動態調整方法：ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=10,verbose=True)

    writer = SummaryWriter('./log')

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    seed = 3000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.set_deterministic(True)
    # random.seed(seed)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()

    time_start = time.time()
    for epoch in range(args.epochs):

        epoch_running_loss = 0

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
            optimizer.zero_grad()

            X_batch = Variable(X_batch.to(device ='cuda'))
            y_batch[y_batch != 0] = 1
            # print(y_batch.shape)
            y_batch = y_batch.squeeze(1)
            # print(y_batch)
            y_batch = Variable(y_batch.to(device='cuda',dtype=torch.int64))

            # ===================forward=====================

            with torch.cuda.amp.autocast():
                output = model(X_batch)
                loss = criterion(output, y_batch)
                print(loss)

            # print('X_batch.shape, y_batch.shape:', X_batch.shape, y_batch.shape)
            # tmp2 = y_batch.detach().cpu().numpy() # mask
            # tmp = output.detach().cpu().numpy() # output
            # tmp[tmp>=0.5] = 1
            # tmp[tmp<0.5] = 0
            # tmp2[tmp2>0] = 1
            # tmp2[tmp2<=0] = 0
            # tmp2 = tmp2.astype(int)
            # tmp = tmp.astype(int)
            #
            # yHaT = tmp
            # yval = tmp2

            # loss = criterion(output, y_batch)
            # ===================amp backward====================
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # scheduler.step()
            scaler.update()


            # ===================backward====================
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            epoch_running_loss += loss.item()

        # ===================log========================
        avg_loss = epoch_running_loss/(batch_idx+1)
        print('epoch [{}/{}], loss:{:.8f}'
              .format(epoch+1, args.epochs, avg_loss))
        writer.add_scalar('loss', scalar_value=avg_loss, global_step=epoch)


        if epoch == 10:
            for param in model.parameters():
                param.requires_grad =True
        if (epoch % args.save_freq) ==0:

            for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
                # print(batch_idx)
                # if isinstance(rest[0][0], str):
                #             image_filename = rest[0][0]
                # else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                X_batch = Variable(X_batch.to(device='cuda'))
                y_batch = Variable(y_batch.to(device='cuda'))
                # start = timeit.default_timer()
                y_out = model(X_batch)
                # stop = timeit.default_timer()
                # print('Time: ', stop - start)
                tmp2 = y_batch.detach().cpu().numpy()
                tmp = y_out.detach().cpu().numpy()
                tmp[tmp>=0.5] = 1
                tmp[tmp<0.5] = 0
                tmp2[tmp2>0] = 1
                tmp2[tmp2<=0] = 0
                tmp2 = tmp2.astype(int)
                tmp = tmp.astype(int)

                # print(np.unique(tmp2))
                yHaT = tmp
                yval = tmp2

                epsilon = 1e-20

                del X_batch, y_batch,tmp,tmp2, y_out

                yHaT[yHaT==1] =255
                yval[yval==1] =255
                fulldir = direc+"/{}/".format(epoch)
                # print(fulldir+image_filename)
                if not os.path.isdir(fulldir):

                    os.makedirs(fulldir)

                cv2.imwrite(fulldir+image_filename, yHaT[0,1,:,:])
                # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
            fulldir = direc+"/{}/".format(epoch)
            torch.save(model.state_dict(), fulldir+args.modelname+".pth")
            torch.save(model.state_dict(), direc+"final_model.pth")
        # 動態調整進入下一個step
        scheduler.step()
    time_end = time.time()
    print(f'training Done! Use {time_end-time_start} s')

if __name__ == '__main__':
    # 處理輸入參數，輸入參數實例化成args後傳給main使用
    parser = argparse.ArgumentParser(description='My train model')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run(default: 400)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='batch size (default: 1)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--train_dataset', required=True, type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--save_freq', type=int,default = 10)
    parser.add_argument('--modelname', default='MedT', type=str,
                        help='type of model')
    parser.add_argument('--cuda', default="on", type=str,
                        help='switch on/off cuda option (default: off)')
    parser.add_argument('--aug', default='off', type=str,
                        help='turn on img augmentation (default: False)')
    parser.add_argument('--load', default='default', type=str,
                        help='load a pretrained model')
    parser.add_argument('--save', default='default', type=str,
                        help='save the model')
    parser.add_argument('--direc', default='./medt', type=str,
                        help='directory to save')
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--imgsize', type=int, default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gray', default='no', type=str)
    args = parser.parse_args()
    print(args)
    main(args)

'==========================================================================================='

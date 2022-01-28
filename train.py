import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import no_grad
import os
from torch.cuda.amp import GradScaler, autocast
import time
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch.nn.functional as F
import shutil

from utils.GS_Dataloader import Make_Dataset,Image2D
from utils.Dataloader_breastUS import ImageToImage2D,Image2D
from utils import loss_fn, Use_model
from utils.Use_model import Use_model
from show_img import Save_image

import argparse

def main(args):
    device = args.device
    save_freq = args.save_freq
    args.use_autocast = bool(args.use_autocast)
    # 選擇使用的資料集：選擇有
    # [Gland Segmentation in Colon Histology Images Challenge]：用GS_Dataloader.py導入，
    # [breast Ultrasound lmage Dataset]：用Dataloader_breastUS.py導入
    if args.dataset == 'GS':
        train_dataset = Make_Dataset(args.train_dataset,img_size=(args.imgsize,args.imgsize))
        train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'BreastUS':
        train_dataset = ImageToImage2D(args.train_dataset,img_size=(args.imgsize,args.imgsize))
        train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 確定使用model種類
    model = Use_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 學習率動態調整方法：ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    scaler = GradScaler(enabled=args.use_autocast)
    # 初始化訓練結果資料夾(清空)
    def init_training_result_folder():
        path = './Model_Result'
        files = os.listdir(path)
        for file in files:
            cur_path = os.path.join(path,file)
            if os.path.isdir(cur_path):
                shutil.rmtree(cur_path)
            else:
                os.remove(cur_path)
    init_training_result_folder()

    if not os.path.exists('./Model_Result/log'):
        os.makedirs('./Model_Result/log')
    # 紀錄訓練狀況
    writer = SummaryWriter('./Model_Result/log')

    time_start = time.time()
    # n = model.parameters()
    # print(model.state_dict().keys())

    def train_epoch(test_epoch, safe_mode=1):
        '''
        :param
        test_epoch：需要訓練幾個epoch，代表最終訓練的epoch數量，屬於暫時型限制，最終訓練時test_epoch==arg.epoch
        save_freq：多少個epoch儲存一次checkpoint，預設為1
        '''
        epoch = args.epoch
        best_loss = 100
        best_model = ''

        def train_one_epoch(dataloader, model):
            '''
            training 1 epoch is packed in training epochs
            :param
                use_autocast: whether use automatic mixed precision training, default True
            '''
            model.cuda()
            model.train()
            train_accumulation_steps = 5 # iters per update
            for i, (image, mask) in enumerate(dataloader):
                image, mask = image.to(device), mask.to(device)  # b c h w
                with autocast(enabled=args.use_autocast):
                    output = model(image) # b,c,h,w
                    output = F.log_softmax(output, dim=1)
                    # print(image, image.shape, output, output.shape, sep='\n')
                    # 全部的loss函數計算
                    weight_cross_entropy_loss = loss_fn.weight_cross_entropy(output, mask, wce_beta=args.wce_beta)
                    dice_coef_loss_loss = loss_fn.dice_coef_loss(output, mask)
                    IoU_loss = loss_fn.IoU(output, mask)
                    FocalLoss_loss = loss_fn.FocalLoss()(output, mask)

                    if args.loss_fn == 'weight_cross_entropy':
                        loss = weight_cross_entropy_loss / train_accumulation_steps
                    elif args.loss_fn == 'dice_coef_loss':
                        loss = dice_coef_loss_loss / train_accumulation_steps
                    elif args.loss_fn == 'IoU':
                        loss = IoU_loss / train_accumulation_steps
                    elif args.loss_fn == 'FocalLoss':
                        loss = FocalLoss_loss / train_accumulation_steps

                # =================backward==================
                if args.use_autocast:
                    # 使用混合精度訓練
                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(optimizer)
                    scaler.update()
                if args.use_autocast == False:
                    loss.backward()
                    if i % train_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                # 僅供測試使用!!!實際訓練要關!!!
                # if i == 3:
                #     scheduler.step(loss)
                #     return weight_cross_entropy_loss, dice_coef_loss_loss, IoU_loss, FocalLoss_loss
                # return all losses, including weight_cross_entropy_loss,dice_coef_loss_loss,IoU_loss,FocalLoss_loss
                if i + 1 == len(dataloader):
                    # if args.use_autocast == False:
                    #     loss.backward()
                    #     optimizer.step()
                    scheduler.step(loss)
                    return weight_cross_entropy_loss, dice_coef_loss_loss, IoU_loss, FocalLoss_loss

        # 測試val資料集，比較結果包含原影像、model輸出、GT
        def eval(val_dataset, model, epoch):
            model.eval()
            model.to('cpu')
            print('start eval!!!')
            save_path = r'./Model_Result/val_images'
            save_path = fr'{save_path}/epoch{epoch}'
            test_loss = 0

            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            for i, (original_image, mask, original_size) in enumerate(val_dataset):
                x = original_image.to(torch.float)
                with no_grad():
                    pred = model(x) # b,c,h,w
                    # Use CE as loss function
                    test_loss += loss_fn.weight_cross_entropy(pred, mask, args.wce_beta)
                # print('pred.shape：',pred.shape)
                Save_image(original_image, pred, mask, save_path=fr'{save_path}/num{i+1}',original_size=original_size)
            test_loss = test_loss / len(val_dataset)
            print('eval finish!!!','avg_test_loss：{:.4f}'.format(test_loss),sep='\t')

        if args.use_autocast:  # only appear when first sample of epoch
            print('=' * 10, 'use_autocast!', '=' * 10)
        else:
            print('=' * 10, 'NO use_autocast!', '=' * 10)
        for i, _ in enumerate(range(epoch)):
            # return 4 type of loss
            weight_cross_entropy, dice_coef_loss, IoU, FocalLoss = train_one_epoch(train_dataset, model)
            writer.add_scalar(f'weight_cross_entropy loss', scalar_value=weight_cross_entropy, global_step=i)
            writer.add_scalar(f'dice_coef_loss loss', scalar_value=dice_coef_loss, global_step=i)
            writer.add_scalar(f'IoU loss', scalar_value=IoU, global_step=i)
            writer.add_scalar(f'FocalLoss loss', scalar_value=FocalLoss, global_step=i)

            if args.loss_fn == 'weight_cross_entropy':
                loss = weight_cross_entropy
            elif args.loss_fn == 'dice_coef_loss':
                loss = dice_coef_loss
            elif args.loss_fn == 'IoU':
                loss = IoU
            else:
                # args.loss_fn == 'FocalLoss'無法執行
                loss = FocalLoss
            print(f'1 epoch loss = {loss}')
            if i % int(save_freq) == 0 and i != 0:
                model.eval()
                # 載入驗證資料集
                val_dataset = Image2D(args.val_dataset)
                val_dataset = DataLoader(val_dataset)
                eval(val_dataset,model,i)
                print('1 epoch finish!!!')
                if args.save_state_dict:
                    if safe_mode == 0:
                        # 儲存多個模型(只要比前面好就儲存)
                        save_name = f"{args.direc}/num{i // save_freq}_model.pth"
                        torch.save(model.state_dict(), save_name)
                        print(f'best_model is in {i} epoch', f"num{i // save_freq}_model.pt are saved!", sep='\t')
                    if safe_mode == 1:
                        # 儲存單個模型(只儲存最好的)
                        save_name = f"{args.direc}/best_model.pth"
                        torch.save(model.state_dict(), save_name)
                        print(f'best_model is in {i} epoch', f"best_model.pt are saved!", sep='\t')
                else:
                    if safe_mode == 0:
                        # 儲存多個模型(只要比前面好就儲存)
                        save_name = f"{args.direc}/num{i // save_freq}_model.pth"
                        torch.save(model.state_dict(), save_name)
                        print(f'best_model is in {i} epoch', f"num{i // save_freq}_model.pt are saved!", sep='\t')
                    if safe_mode == 1:
                        # 儲存單個模型(只儲存最好的)
                        save_name = f"{args.direc}/best_model.pth"
                        torch.save(model.state_dict(), save_name)
                        print(f'best_model is in {i} epoch', f"best_model.pt are saved!", sep='\t')
                best_loss = loss

            if i+1 == epoch:  # 測試用的epoch, test_epoch=1 代表訓練兩個epoch
                print('='*10,'last one eval','='*10)
                val_dataset = Image2D(args.val_dataset)
                val_dataset = DataLoader(val_dataset)
                eval(val_dataset, model, i)
                print('='*10,'last one eval finish!!!!!','='*10)
                break
    # 訓練(迭代)階段
    train_epoch(args.epoch, safe_mode=1)
    # 計算時間(以秒為單位)
    time_end = time.time()
    print(f'training Done! Use {time_end - time_start} s')

    # def choose_loss_fn(output, target):
    #     # 選擇使用的loss function。
    #     # 經過測試可以使用的(在MedT環境中)：weight_cross_entropy, dice_coef_loss,IoU,FocalLoss
    #     # 目前狀態沒有被調用
    #     if args.loss_fn == 'weight_cross_entropy':
    #         loss = weight_cross_entropy(output, target, wce_beta=args.wce_beta)
    #         return loss
    #     elif args.loss_fn == 'dice_coef_loss':
    #         loss = dice_coef_loss(output, target)
    #         return loss.to(device)
    #     elif args.loss == 'IoU':
    #         loss = IoU(output, target)
    #         return loss
    #     elif args.loss == 'FocalLoss': # 這個criterion是用torch.nn.module建立，需要當作layer看待
    #         criterion = FocalLoss()
    #         return criterion








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Test Version')

    parser.add_argument('--epoch', default=3, type=int, help='需要跑的輪數')
    parser.add_argument('--train_dataset', required=True, type=str, help='訓練資料集位置')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--modelname', required=True, type=str)
    parser.add_argument('--dataset', choices=['GS','BreastUS'],default='GS',help='選擇使用的資料集，默認GS，可以選擇BreastUS')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_freq', default=1,type=int, help='多少個epoch儲存一次checkpoint')
    parser.add_argument('--save_state_dict', type=bool, default=True, help='是否只儲存權重，默認為權重')
    parser.add_argument('--loss_fn', default='weight_cross_entropy',
                        choices=['weight_cross_entropy', 'dice_coef_loss', 'IoU', 'FocalLoss'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wce_beta', type=float, default=1e-04, help='wce_loss的wce_beta值，如果使用wce_loss時需要設定')
    parser.add_argument('--imgsize', type=int, default=256, help='圖片大小')
    parser.add_argument('--imgchan', type=int, default=3, help='訓練影像通道數')
    parser.add_argument('--direc', default='./Model_Result', type=str, help='directory to save')
    parser.add_argument('--val_dataset', type=str, help='驗證用資料集所在位置')
    parser.add_argument('--use_autocast', type=int, help='是否使用混和精度訓練')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)  # supress Userwarning
    main(args)




'''-------------------------------------------------------'''
# train_dataset = "D:\Programming\AI&ML\(Dataset)Gland Segmentation in Colon Histology Images Challenge\dataset"
# val_dataset = "D:\Programming\AI&ML\(Dataset)Gland Segmentation in Colon Histology Images Challenge\val_dataset"
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
from torch import no_grad
import os
from torch.cuda.amp import GradScaler, autocast
import time
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch.nn.functional as F
import shutil
from torchvision.utils import save_image
from sklearn.model_selection import KFold
import numpy as np

from utils.Dataloader_breastUS import ImageToImage2D,Image2D,JointTransform2D
from utils import loss_fn, Use_model, adaptation_th
from utils.Use_model import *
from show_img import Save_image
from utils.remove_readonly import remove_readonly

import argparse
import yaml





'''
Train script ver1.0

此model script為訓練模型所編寫，啟動時需要透過windows powershell啟動
使用時需要選擇使用的model名稱、loss函數、訓練週期等參數

注意事項:
pass

更新紀錄:
    tarin ver1.0。
        2022/4/18
        1. 加入 k_fold 訓練機制: Limitation: k_fold設定不可為1。
    train ver1.0.1
        2022/4/19
        1. yaml功能。新增資料讀取方法
        2. 資料視覺化與計算f1 score, iou計算影像相同
        3. 新增deep supervise功能。都用if else來寫，目前非常不好維護，後面需要修改。

'''

def main(args):
    save_freq = args.save_freq
    safe_mode = 1
    args.use_autocast = bool(args.use_autocast)
    train_dataset = ImageToImage2D(args.train_dataset, img_size=(args.imgsize, args.imgsize), get_catagory=args.catagory)
    # train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_dataset = Image2D(args.val_dataset, img_size=(args.imgsize, args.imgsize))
    # val_dataset = DataLoader(val_dataset)
    splits = KFold(n_splits=int(args.k_fold), shuffle=True, random_state=42)  # 設定random_state使輸出都一樣

    model = Use_model(args)
    optimizer = use_opt(args, model)
    scheduler = use_scheduler(args, optimizer)
    writer = SummaryWriter('./Model_Result/log')
    time_start = time.time()
    use_autocast = f'{"="*10} USE autocast! {"="*10}' if args.use_autocast else f'{"="*10} NO autocast! {"="*10}'
    warnings.warn(use_autocast)

    f_val_loss, f_f1, f_iou = 0.0, 0.0, 0.0
    for fold, (train_idx, val_idx) in enumerate(splits.split(train_dataset)):
        print('----- Fold {} -----'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=1, sampler=test_sampler)
        for i, epoch in enumerate(range(args.epoch)):
            loss = train_one_epoch(args,
                                   dataloader=train_loader,
                                   model=model,
                                   optimizer=optimizer,
                                   scheduler=scheduler)
            writer.add_scalar(f'training {args.loss_fn} loss', scalar_value=loss, global_step=i+(args.epoch*fold))
            if i % save_freq == 0:
                assert save_freq > 1, 'save_freq只能設定大於1。'
                val_dataset = Image2D(args.val_dataset, img_size=(args.imgsize, args.imgsize))
                final_val_dataset = DataLoader(val_dataset)
                folder_name = fr'./Model_Result/val_images/fold{fold}_epoch{epoch}'
                val_loss, f1, iou = eval(final_val_dataset, model, folder_name, binarization=False,
                                                   scaling=True, save=True)  # must set scaling and binarization
                writer.add_scalar(f'fold_{fold} val_loss', scalar_value=val_loss, global_step=i)
                writer.add_scalar(f'fold_{fold} f1 score', scalar_value=f1, global_step=i)
                writer.add_scalar(f'fold_{fold} mIoU score', scalar_value=iou, global_step=i)
                # if args.savemodel:
                #     save_name = ''
                #     save_model_mode(model, safe_mode, save_freq, i)
            if i + 1 == args.epoch:  # 測試用的epoch, test_epoch=1 代表訓練兩個epoch
                print('=' * 10, 'last one eval', '=' * 10)
                val_dataset = Image2D(args.val_dataset, img_size=(args.imgsize, args.imgsize))
                final_val_dataset = DataLoader(val_dataset)
                folder_name = fr'./Model_Result/val_images/fold{fold}_epoch{epoch}'
                val_loss, f1, iou = eval(final_val_dataset, model, folder_name, binarization=False, scaling=False, save=True)
                f_val_loss += val_loss
                f_f1 += f1
                f_iou += iou
                print('=' * 10, 'last one eval finish!!!!!', '=' * 10)
                if args.savemodel:
                    save_name = f'./Model_Result/model_fold_{fold+1}.pth'
                    save_model_mode(model, save_name)
                    # 模型初始化
                    model = Use_model(args)
                break
    f_val_loss, f_f1, f_iou = f_val_loss/(fold+1), f_f1/(fold+1), f_iou/(fold+1)
    print('f_val_loss:{:8f}, f_f1:{:8f}, f_iou:{:8f}'.format(f_val_loss.item(), f_f1.item(), f_iou.item()))

    time_end = time.time()
    print('training Done! Use {:.2f} s'.format(time_end - time_start))

def train_one_epoch(args, dataloader, model, optimizer, scheduler):
    scaler = GradScaler(enabled=args.use_autocast)
    model.to(args.device)
    model.train()
    train_accumulation_steps = args.train_accumulation_steps
    for i, (image, mask) in enumerate(dataloader):
        image = image.to(args.device) if args.device == 'cuda:0' else image
        mask = mask.to(args.device) if args.device == 'cuda:0' else mask
        with autocast(enabled=args.use_autocast):
            f_loss = 0.
            if args.deep_supervise:
                # Use deep supervise
                output, *f = model(image)
                f = f[0]
                f = f.clone().detach().requires_grad_(True)
                f_loss = deep_supervise(f, mask=mask)
            else:
                output = model(image)
            loss = (choose_loss_fn(output, mask) + f_loss) / train_accumulation_steps
        if args.use_autocast:
            FutureWarning('AMP is ON, precision will a little down')
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        if not args.use_autocast:
            loss.backward()
            if i % train_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        if not args.run_formal and i == 2:
            print('----Not formal training!----')
            scheduler.step(loss)
            return loss
        if i + 1 == len(dataloader):
            scheduler.step(loss)
            return loss


def eval(val_dataset, model, folder_name, binarization=False, scaling=False, save=False):
    model.eval()
    model.to('cuda')
    print('start eval!!!')
    save_path = folder_name
    test_loss, f1, iou = 0., 0., 0.

    original_size = torch.tensor((128,128))

    for i, (original_image, mask) in enumerate(val_dataset):
        original_image = original_image.to('cuda') if torch.cuda.is_available() else original_image
        x = original_image.to(torch.float32)
        with no_grad():
            if args.deep_supervise:
                pred, *_ = model(x) # b,c,h,w
            else:
                pred = model(x) # b,c,h,w

            pred = pred.to('cpu')
            # Use loss function
            if scaling:
                pred = loss_fn.sigmoid_scaling(pred)
            if binarization:
                pred = (pred > args.threshold).float() # 1, 1, 128, 128
            pred = adaptation_th.THRESH_BINARY_for_pred(pred, return_tensor=True)
            test_loss += choose_loss_fn(pred, mask)
            f1 += loss_fn.classwise_f1(pred, mask)
            iou += loss_fn.IoU(pred, mask)
        if save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            Save_image(original_image.to('cpu'), pred, mask,
                       save_path=fr'{save_path}/num{i + 1}',
                       original_size=original_size,
                       channel=args.classes,
                       th=int(args.threshold * 255),
                       resize=args.savefig_resize
                       )

    val_loss = test_loss / len(val_dataset)
    f1 = f1 / len(val_dataset)
    iou = iou / len(val_dataset)
    print(f'epoch validation.',
          'avg_eval_loss：{:.4f}, '
          'f1 score：{:.4f},'
          ' mIoU score：{:.4f}'.format(test_loss,f1.item(),iou.item()),
          sep='\t')
    return val_loss, f1, iou

    # =============================額外增加功能放在這邊=============================
def save_training_images(image):
    '''儲存訓練過程產生的影像，假設輸入影像大小為(1,c,h,w)'''
    b,c,h,w = image.shape
    assert c in [1,2,3], f'c is not in [1,2,3], c is {c}'
    if torch.is_tensor(image):
        image = image.squeeze(0)
    save_image(image, './Model_Result/test_files/test_image.png')
def choose_loss_fn(output, target):
    # 選擇使用的loss function。
    # 經過測試可以使用的(在MedT環境中)：weight_cross_entropy, dice_coef_loss,IoU,FocalLoss
    # validation階段調用

    if args.loss_fn == 'wce':
        # wce 可以接受channel=1的output
        loss_fn_name = 'wce'
        loss = loss_fn.weight_cross_entropy(output, target, wce_beta=args.wce_beta)
    elif args.loss_fn == 'bce':
        # wce 可以接受channel=1的output
        loss_fn_name = 'bce'
        loss = loss_fn.binary_cross_entropy(output, target, _2class=bool(args.classes-1))
    elif args.loss_fn == 'dice_coef_loss':
        loss_fn_name = 'dice_coef_loss'
        loss = loss_fn.dice_coef_loss(output, target)
    elif args.loss_fn == 'IoU':
        loss_fn_name = 'IoU'
        loss = loss_fn.IoU(output, target)
    elif args.loss_fn == 'FocalLoss':  # 這個criterion是用torch.nn.module建立，需要當作layer看待
        loss_fn_name = 'FocalLoss'
        loss = loss_fn.FocalLoss()(output, target)
    elif args.loss_fn == 'lll':
        loss_fn_name = 'lll'
        loss = loss_fn.LogNLLLoss()(output, target)
    elif args.loss_fn == 'diceloss':
        loss_fn_name = 'diceloss'
        loss = loss_fn.dice_loss(output, target)
    elif args.loss_fn == 'clsiou':
        loss_fn_name = 'clsiou'
        loss = loss_fn.classwise_iou(output, target)
    # print('----- loss_fn_name: ',loss_fn_name, '-----')
    return loss
def init_training_result_folder():
    '''初始化並創建資料夾'''
    path = './Model_Result'
    files = os.listdir(path)
    for file in files:
        cur_path = os.path.join(path,file)
        if os.path.isdir(cur_path):
            shutil.rmtree(cur_path, onerror=remove_readonly)
        else:
            os.remove(cur_path)
    if not os.path.exists('./Model_Result/log'):
        os.makedirs('./Model_Result/log')
    if not os.path.exists('./Model_Result/test_files'):
        os.makedirs('./Model_Result/test_files')
def save_model_mode(model, model_name):
    '''
    儲存model的模式設定：
    # safe_mode == 0 代表儲存多個模型；safe_mode == 1 儲存單一模型。已棄用。

    此區域上未完工！！
    '''
    # 儲存模型
    save_name = model_name
    torch.save(model.state_dict(), save_name)
    print(f"1 epoch finish!!! {model_name} are saved!", sep='\t')
def model_output_dim_adjust(model_output):
    mo = model_output
    ch, size = 2, args.imgsize # use 2 if MedT, use 1 if timm(ViT)
    if mo.shape == (1, ch, size, size):
        return mo
    elif mo.dim() == 3:
        assert len(mo) == ch
        mo = mo.squeeze(0) # add batch dim
        return mo
    elif mo.dim() == 2:
        assert len(mo) == 1 and len(mo[0]) != 0
        mo = mo.view(1, 1, size, size)
        return mo
def deep_supervise(*feature, mask):
    '''
    feature: 輸入影像之特徵(tuple of tensor), Any size of feature.
        default (B,C,H,W): (8, 16, 32, 32)
        default(N,B,C,H,W): ((8,128,32,32),(8,64,64,64))
    mask:
        default(B,1,H,W)

    :return f_loss (scalar)
    '''
    # f_weight = torch.nn.Parameter(torch.randn(len(feature))) #可學習權重
    f_weight = torch.ones(len(feature)) #不可學習權重
    f_loss = []
    for f in feature:
        b,c,h,w = mask.shape
        f = f.to(args.device) if args.device == 'cuda:0' else f
        f = F.interpolate(f, size=(h,w), mode='bilinear', align_corners=True)
        f_loss.append(choose_loss_fn(f, mask))

    f_loss = torch.mul(f_weight, torch.tensor(f_loss)).sum()
    return f_loss
def parser_args(model_name):
    parser = argparse.ArgumentParser(description='Transformer Test Version')
    ds_path = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\Dataset_BUSI_with_GT"
    vds_path = r"D:\Programming\AI&ML\(Dataset)breast Ultrasound lmage Dataset\archive\val_ds2"

    'yaml test'
    parser.add_argument('--fname', type=str, help='name of config file to load', default=r'.\config\train_config.yaml')

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
        dump = os.path.join(fr'{params["save"]["direc"]}', 'training setting.yaml')
        with open(dump, 'w') as f: # 寫入檔案
            yaml.dump(params, f)
        return params

    fname = r'D:\Programming\AI&ML\model\config\train_config.yaml'
    args = _process_main(fname)
    print(args['meta']['modelname'])

    # Training parameter setting
    parser.add_argument('--epoch', default=args['optimization']['epochs'], type=int, help='需要跑的輪數')
    parser.add_argument('-bs', '--batch_size', default=args['optimization']['batchsize'], type=int)
    parser.add_argument('-is', '--imgsize', type=int, default=args['data']['imgsize'], help='圖片大小')
    parser.add_argument('-ic', '--imgchan', type=int, default=args['data']['imgchan'], help='使用資料的通道數，預設3(RGB)')
    parser.add_argument('-class', '--classes', type=int, default=args['data']['classes'], help='model輸出影像通道數(grayscale)')
    parser.add_argument('-model', '--modelname', default=args['meta']['modelname'], type=str)
    parser.add_argument('-ds_path', '--train_dataset', default=fr'{args["data"]["ds_path"]}', type=str, help='訓練資料集位置')
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
    parser.add_argument('--deep_supervise', type=bool, default=args['optimization']['deep_supervise'], help='使用深層監督')
    parser.add_argument('--pos', type=bool, default=args['optimization']['pos'], help='位置編碼')


    # Optimizer Setting
    parser.add_argument('--lr', type=float, default=args['optimization']['lr'], help='learning rate')
    parser.add_argument('--scheduler', type=str, default=args['criterion']['scheduler'], help='使用的scheduler')
    parser.add_argument('-opt', '--optimizer', type=str, default=args['criterion']['optimizer'], help='使用的optimizer')

    # Loss function and Loss schedule
    parser.add_argument('-loss', '--loss_fn', type=str, default=args['criterion']['loss'],
                        choices=['wce', 'diceloss', 'IoU', 'FocalLoss', 'bce', 'lll', 'clsiou'])
    parser.add_argument('-wce', '--wce_beta', type=float, default=1e-04, help='wce_loss的wce_beta值，如果使用wce_loss時需要設定')

    # Save Setting
    parser.add_argument('-sf', '--save_freq', type=int, default=args['save']['save_frequency'],
                        help='多少個epoch儲存一次checkpoint')
    parser.add_argument('--save_state_dict', type=bool, default=args['save']['save_state_dict'],
                        help='是否只儲存權重，默認為權重')
    parser.add_argument('--savemodel', type=bool, default=args['save']['savemodel'], help='是否儲存模型')
    parser.add_argument('-r', '--run_formal', type=bool, default=args['save']['run_formal'],
                        help='是否是正式訓練(if not, train 8 iters for each epoch)')
    parser.add_argument('--direc', type=str, default=args['save']['direc'], help='directory to save')
    parser.add_argument('--savefig_resize', type=bool, default=args['save']['savefig_resize'], help='savefig resize')
    # parser.add_argument('--savefig', default=False, type=bool, help='儲存影像的設定，目前沒有用到。需更改。')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    model_name = 'medt'

    init_training_result_folder()
    args = parser_args(model_name)
    main(args)
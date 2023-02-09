from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import no_grad
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from utils.Dataloader_breastUS import ImageToImage2D, Image2D
from utils import loss_fn
from utils.Use_model import *
from show_img import Save_image
from utils import Other_utils as ou

import os
import time
import argparse
import yaml
import logging
import pprint
import pickle

'''
Training script version 2.0.0

使用時於train_config設定model、loss、epochs等參數，DatasetConfig設定資料集路徑

更新紀錄:
    Training script version 1。
        1. 加入 k_fold 訓練機制: Limitation: k_fold設定不可為1。
        2. yaml功能。新增資料讀取方法
        3. 資料視覺化與計算f1 score, iou計算影像相同
        4. 新增deep supervise功能。都用if else來寫，目前非常不好維護，後面需要修改。
        5. 更改程式架構: 分成Optimizer、K Fold、Training部分，增加註解
        6. init_training_result_folder與save_model_mode移到新檔案Other_utils中
        8. 加入訓練初始化
        9. 加入cudnn_benchmark，加速運算
        
    Training script version 2.0.0
        1. 加入Pretrain選項，於DatasetConfig選擇要使用PretrainDataset/TrainDataset以及使用PretrainWeight

'''
print('Training script version 2.0.0', 'Last edit in 20220830', sep=', ')


def main(args):
    save_freq = args.save_freq  # Freq of save model and valid model performance
    gray = False if args.imgchan == 3 else True    # Training images is RGB or gray scale

    # When pretraining on pretrain dataset. Modify dataset in ./config/DatasetConfig
    if args.TrainState == 'Pretrain':
        print('<<Pretraining.....>>')
        train_dataset = ImageToImage2D(args.PretrainDataset, img_size=(args.imgsize, args.imgsize),
                                   get_catagory=args.catagory, Gray=gray, datasetname=args.PretrainDatasetName)
    else:
        train_dataset = ImageToImage2D(args.train_dataset, img_size=(args.imgsize, args.imgsize),
                                       get_catagory=args.catagory, Gray=gray, datasetname=args.TrainDatasetName)
    splits = KFold(shuffle=True, n_splits=int(args.k_fold), random_state=42)  # 設定random_state使輸出都一樣

    model = use_model(args)
    criteria = use_loss_fn(args)
    optimizer = use_opt(args, model)
    scheduler = use_scheduler(args, optimizer)
    writer = SummaryWriter(f'{args.direc}/log')
    time_start = time.time()

    f_val_loss, f_f1, f_iou = 0.0, 0.0, 0.0

    for fold, (train_idx, val_idx) in enumerate(splits.split(train_dataset)):
        # Add pretrain function
        if args.LOAD_PRETRAIN_WEIGHT:
            print(f'load pretrainWeight: {args.pretrainWeightName}')
            model_weight = torch.load(args.pretrainWeight)
            model.load_state_dict(model_weight)

        print('----- Fold {} -----'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=1, sampler=test_sampler)
        one_fold_start_time = time.time()
        for i, epoch in enumerate(range(args.epoch)):
            loss = train_one_epoch(args,
                                   dataloader=train_loader,
                                   model=model,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   lossfn=criteria)
            writer.add_scalar(f'fold{fold} training {args.loss_fn} loss', scalar_value=loss, global_step=i + (args.epoch * fold))
            if save_freq != 0:  # 驗證是否固定頻率紀錄訓練狀況
                if i % save_freq == 0:
                    assert save_freq > 1, 'save_freq只能設定大於1。'
                    val_dataset = Image2D(args.val_dataset, img_size=(args.imgsize, args.imgsize), Gray=gray)
                    final_val_dataset = DataLoader(val_dataset)
                    folder_name = fr'{args.direc}/val_images/fold{fold + 1}_epoch{epoch + 1}'
                    save_model_name = f'{args.direc}/model_fold_{fold + 1}_{epoch + 1}.pth'
                    val_loss, f1, iou = eval(final_val_dataset, model, folder_name,
                                             binarization=True,
                                             scaling=True,  # must set scaling and binarization
                                             save_valid_img=args.save_valid_img,
                                             lossfn=criteria,
                                             args=args,
                                             save_model=False,
                                             save_model_name=save_model_name)
                    writer.add_scalar(f'fold_{fold} val_loss', scalar_value=val_loss, global_step=i)
                    writer.add_scalar(f'fold_{fold} f1 score', scalar_value=f1, global_step=i)
                    writer.add_scalar(f'fold_{fold} mIoU score', scalar_value=iou, global_step=i)
            if i + 1 == args.epoch:
                print('=' * 10, 'last one eval', '=' * 10)
                folder_name = fr'{args.direc}/val_images/fold{fold + 1}_epoch{epoch + 1}'
                save_model_name = f'{args.direc}/model_fold_{fold + 1}.pth'
                val_loss, f1, iou = eval(val_loader, model, folder_name,
                                         binarization=True,
                                         scaling=True,
                                         save_valid_img=True,
                                         lossfn=criteria,
                                         args=args,
                                         save_model=args.savemodel,
                                         save_model_name=save_model_name)
                f_val_loss += val_loss
                f_f1 += f1
                f_iou += iou
                one_fold_end_time = time.time()
                writer.add_scalar(f'fold Training time',
                                  scalar_value=one_fold_end_time - one_fold_start_time,
                                  global_step=fold + 1)
                print('=' * 10, 'last one eval finish!!!!!', '=' * 10)
                print('=' * 10, f'One fold training time: {one_fold_end_time - one_fold_start_time}', '=' * 10)
                #
                # ---------- 模型初始化 ------------
                model = use_model(args)
                criteria = use_loss_fn(args)
                optimizer = use_opt(args, model)
                scheduler = use_scheduler(args, optimizer)
                # -------------- Finish --------------
                break
    f_val_loss, f_f1, f_iou = f_val_loss / (args.k_fold + 1), f_f1 / (args.k_fold + 1), f_iou / (args.k_fold + 1)
    print('f_val_loss:{:8f}, f_f1:{:8f}, f_iou:{:8f}'.format(f_val_loss, f_f1, f_iou))

    time_end = time.time()
    print('training Done! Use {:.2f} s'.format(time_end - time_start))
    if args.SaveAsPKL:  # 儲存成可部屬模型格式
        with open(f'{args.direc}/model.pkl', 'wb') as f:
            pkl_model = pickle.dumps(model, f)

    path = os.path.join(args.direc, 'TrainingLogger.txt')
    with open(path, 'w') as f:
        f.write(f"Training time: \t{time_end - time_start}\n\n")
        f.write('final_val_loss:\t{:8f}\n final_f1:\t{:8f}\n final_iou:\t{:8f}'.format(f_val_loss, f_f1, f_iou))


def train_one_epoch(args, dataloader, model, lossfn, optimizer, scheduler):
    model.to(args.device)
    model.train()
    train_accumulation_steps = args.train_accumulation_steps
    for i, (image, mask) in enumerate(dataloader):
        image = image.to(args.device) if args.device == 'cuda:0' else image
        mask = mask.to(args.device) if args.device == 'cuda:0' else mask

        f_loss = 0.
        output = model(image)
        loss = (lossfn(output, mask) + f_loss) / train_accumulation_steps
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


def eval(val_dataset, model, folder_name, lossfn, args=None, binarization=False, scaling=False,
         save_valid_img=False, save_model=False, save_model_name=None):
    model.eval()
    model.to('cuda')
    print('start eval!!!')
    save_path = folder_name
    test_loss, f1, iou = 0., 0., 0.

    original_size = torch.tensor((args.imgsize, args.imgsize))

    for i, (original_image, mask) in enumerate(val_dataset):
        original_image = original_image.to('cuda') if torch.cuda.is_available() else original_image
        x = original_image.to(torch.float32)
        with no_grad():
            pred = model(x)  # b,c,h,w
            pred = pred.to('cpu')
            # Use loss function
            if scaling:
                pred = loss_fn.sigmoid_scaling(pred)

            if save_valid_img:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                Save_image(original_image.to('cpu'), pred, mask,
                           save_path=fr'{save_path}/num{i + 1}',
                           original_size=original_size,
                           channel=args.classes,
                           resize=args.savefig_resize
                           )
            if binarization:
                th = pred.max() * args.threshold - pred.min() * (args.threshold - 1)
                pred = (pred > th).float()
                # pred = (pred > args.threshold).float()  # 1, 1, 128, 128
                # pred = OU.THRESH_BINARY_for_pred(pred, return_tensor=True)
            test_loss += lossfn(pred, mask)
            f1 += loss_fn.classwise_f1(pred, mask)
            iou += loss_fn.IoU(pred, mask)
            del pred

    if save_model and save_model_name:
        ou.save_model_mode(model, save_model_name)

    val_loss = test_loss / len(val_dataset)
    f1 = float(f1 / len(val_dataset))
    iou = float(iou / len(val_dataset))
    print(f'epoch validation.',
          'avg_eval_loss：{:.4f}, '
          'f1 score：{:.4f},'
          ' mIoU score：{:.4f}'.format(test_loss, f1, iou),
          sep='\t')
    return val_loss, f1, iou


def parser_args():
    """yaml test"""
    parser = argparse.ArgumentParser(description=' Version')
    parser.add_argument('--fname', type=str, help='name of config file to load',
                        default=r'.\config\train_config.yaml')

    def _process_main(filename):
        logging.basicConfig()
        logger = logging.getLogger()
        with open(filename, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)
        try:
            os.makedirs(params["save"]["direc"]) if os.path.exists(params["save"]["direc"]) is False else print(
            'folder is exist')
        except:
            pass
        return params

    absFilePath = os.path.abspath(__file__)
    fname = os.path.join(os.path.split(absFilePath)[0], 'config/train_config.yaml')
    dataConfig = os.path.join(os.path.split(absFilePath)[0], 'config/DatasetConfig.yaml')
    args = _process_main(fname)
    data = _process_main(dataConfig)
    print(args['meta']['modelname'])
    if args["save"]["run_formal"]:
        ou.Double_check_training_setting()
        ou.init_training_result_folder(args["save"]["direc"])
    dump = os.path.join(fr'{args["save"]["direc"]}', 'training setting.yaml')
    with open(dump, 'w') as f:  # 寫入檔案
        print('writing!!')
        yaml.dump(args, f)
        yaml.dump(data, f)
    # Training parameter setting
    parser.add_argument('--epoch', default=args['optimization']['epochs'], type=int, help='需要跑的輪數')
    parser.add_argument('-bs', '--batch_size', default=args['optimization']['batchsize'], type=int)
    parser.add_argument('-is', '--imgsize', type=int, default=args['data']['imgsize'], help='圖片大小')
    parser.add_argument('-ic', '--imgchan', type=int, default=args['data']['imgchan'], help='使用資料的通道數，預設3(RGB)')
    parser.add_argument('-class', '--classes', type=int, default=args['data']['classes'],
                        help='model輸出影像通道數(grayscale)')
    parser.add_argument('-model', '--modelname', default=args['meta']['modelname'], type=str)
    parser.add_argument('-ds_path', '--train_dataset', default=fr'{data["TrainDataset"]["Path"]["Training data"]}', type=str,
                        help='訓練資料集位置')
    parser.add_argument( '--TrainDatasetName', default=fr'{data["TrainDataset"]["Name"]}', type=str,
                        help='訓練資料集名稱')
    parser.add_argument('-vd', '--val_dataset', type=str, default=data["TrainDataset"]["Path"]["Validation data"], help='驗證用資料集所在位置')
    parser.add_argument('--catagory', type=int, default=args['data']['catagory'], help='使用類別資料與否') # 目前已棄用
    parser.add_argument('--TrainState', type=str, choices=['Pretrain', 'NormalTrain'], default=args['meta']['TrainState'],
                        help='訓練狀態，Pretrain/NormalTrain')
    parser.add_argument('--LOAD_PRETRAIN_WEIGHT', type=bool, default=args['meta']['LOAD_PRETRAIN_WEIGHT'], help='是否載入預訓練權重')

    parser.add_argument('--PretrainDataset', type=str, default=data["PretrainDataset"]["Path"]["Training data"], help='預訓練資料路徑')
    parser.add_argument('--PretrainDatasetName', type=str, default=data["PretrainDataset"]["Name"], help='預訓練資料集名稱')
    parser.add_argument('--pretrainWeight', type=str, default=data["PretrainWeight"]["Path"], help='預訓練權重，未填入視同無')
    parser.add_argument('--pretrainWeightName', type=str, default=data["PretrainWeight"]["Name"], help='預訓練權重來源名稱')

    # Model training setting
    parser.add_argument('--training_details', type=str, default=args['meta']['Name'], help='訓練項目細節。training details')
    parser.add_argument('--device', type=str, default=args['meta']['device'], help='是否使用GPU訓練')
    parser.add_argument('-ds', '--dataset', choices=['BreastUS'], default=args['data']['dataset'],
                        help='(已棄用，改為使用DatasetConfig取資料)選擇使用的資料集，默認GS，預設BreastUS')
    parser.add_argument('--threshold', type=int, default=args['save']['threshold'],
                        help='設定model output後二值化的threshold, 介於0-1之間')
    parser.add_argument('--train_accumulation_steps', default=args['optimization']['train_accumulation_steps'],
                        type=int, help='多少iters更新一次權重(可減少顯存負擔)')
    parser.add_argument('--k_fold', type=int, default=args['optimization']['k_fold'], help='使用k_fold訓練')
    parser.add_argument('--pos', type=bool, default=args['optimization']['pos'], help='位置編碼')

    # Optimizer Setting
    parser.add_argument('--lr', type=float, default=args['optimization']['lr'], help='learning rate')
    parser.add_argument('--scheduler', type=str, default=args['criterion']['scheduler'], help='使用的scheduler')
    parser.add_argument('-opt', '--optimizer', type=str, default=args['criterion']['optimizer'],
                        help='使用的optimizer')
    parser.add_argument('--weight_decay', type=float, default=args['optimization']['weight_decay'],
                        help='Optimizer weight decay')

    # Loss function and Loss schedule
    parser.add_argument('-loss', '--loss_fn', type=str, default=args['criterion']['loss'],
                        choices=['wce', 'diceloss', 'IoU', 'FocalLoss', 'bce', 'lll', 'clsiou'])
    parser.add_argument('-wce', '--wce_beta', type=float, default=1e-04,
                        help='wce_loss的wce_beta值，如果使用wce_loss時需要設定')

    # Save Setting
    parser.add_argument('-sf', '--save_freq', type=int, default=args['save']['save_frequency'],
                        help='多少個epoch儲存一次checkpoint。Set 0 將不會進行測試')
    parser.add_argument('--save_state_dict', type=bool, default=args['save']['save_state_dict'],
                        help='是否只儲存權重，默認為權重')
    parser.add_argument('--savemodel', type=bool, default=args['save']['savemodel'], help='是否儲存模型')
    parser.add_argument('-r', '--run_formal', type=bool, default=args['save']['run_formal'],
                        help='是否是正式訓練(if not, train 8 iters for each epoch)')
    parser.add_argument('--direc', type=str, default=args['save']['direc'], help='directory to save')
    parser.add_argument('--savefig_resize', type=bool, default=args['save']['savefig_resize'], help='savefig resize')
    parser.add_argument('--save_valid_img', type=bool, default=args['save']['save_valid_img'],
                        help='save validation result(in every save freq)')
    parser.add_argument('--SaveAsPKL', type=bool, default=False, help='是否將模型儲存成.pkl(部署使用格式)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(args=parser_args())

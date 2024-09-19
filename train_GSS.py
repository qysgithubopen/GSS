#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models_GSS as models
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax

from GSS_choice import singel_teacher, double_teacher

from random_mask import random_mask

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='./BRATS2020_Training_none_npy/', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='./result/', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=20, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--TEMP', default=8, type=float)
parser.add_argument('--mask_ratio', default=0.2, type=float)
parser.add_argument('--TP', default=0.75, type=float)
parser.add_argument('--TM', default=0.85, type=float)
parser.add_argument('--DTP', default=0.3, type=float)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = models.Model(num_cls=num_cls)
    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train3.txt'
        test_file = 'test3.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        test_score = AverageMeter()
        with torch.no_grad():
            logging.info('###########test set wi post process###########')
            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score = test_softmax(
                                test_loader,
                                model,
                                dataname = args.dataname,
                                feature_mask = mask,
                                mask_name = mask_name[::-1][i])
                test_score.update(dice_score)
            logging.info('Avg scores: {}'.format(test_score.avg))
            exit(0)
    startepoch=0
    ################ Pretrain
    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        startepoch = checkpoint['epoch']

        
    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    temp = args.TEMP
    TP = args.TP
    TM = args.TM
    P = args.DTP
    mask_rato = args.mask_ratio
    max_iter = args.num_epochs * iter_per_epoch
    # print('startepoch', startepoch)
    for epoch in range(args.num_epochs):
        # epoch = startepoch
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):


            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds, seg_logits = model(x, mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            

            flair_logits, t1ce_logits, t1_logits, t2_logits = seg_logits
            flair_logits = F.softmax(flair_logits/temp, dim=1)
            t1ce_logits = F.softmax(t1ce_logits/temp, dim=1)
            t1_logits = F.softmax(t1_logits/temp, dim=1)
            t2_logits = F.softmax(t2_logits/temp, dim=1)

            flair_pred, t1ce_pred, t1_pred, t2_pred = sep_preds

            with torch.no_grad():
                soft_target_1= singel_teacher(flair_pred[:,0:1,:,:,:].clone(), t1ce_pred[:,0:1,:,:,:].clone(),   t1_pred[:,0:1,:,:,:].clone(), t2_pred[:,0:1,:,:,:].clone(), step, max_iter, flair_logits[:,0:1,:,:,:].clone(), t1ce_logits[:,0:1,:,:,:].clone(),   t1_logits[:,0:1,:,:,:].clone(), t2_logits[:,0:1,:,:,:].clone(),TP,TM)
                soft_target_2 = singel_teacher(t1ce_pred[:,1:2,:,:,:].clone(), flair_pred[:,1:2,:,:,:].clone(), t1_pred[:,1:2,:,:,:].clone(), t2_pred[:,1:2,:,:,:].clone(), step, max_iter, t1ce_logits[:,1:2,:,:,:].clone(), flair_logits[:,1:2,:,:,:].clone(), t1_logits[:,1:2,:,:,:].clone(), t2_logits[:,1:2,:,:,:].clone(),TP,TM)
                soft_target_3 = double_teacher(flair_pred[:,2:3,:,:,:].clone(), t2_pred[:,2:3,:,:,:].clone(), t1ce_pred[:,2:3,:,:,:].clone(), t1_pred[:,2:3,:,:,:].clone(), step, max_iter, flair_logits[:,2:3,:,:,:].clone(), t2_logits[:,2:3,:,:,:].clone(), t1ce_logits[:,2:3,:,:,:].clone(), t1_logits[:,2:3,:,:,:].clone(),TP,TM,P)
                soft_target_4 = singel_teacher(t1ce_pred[:,3:4,:,:,:].clone(), flair_pred[:,3:4,:,:,:].clone(), t1_pred[:,3:4,:,:,:].clone(), t2_pred[:,3:4,:,:,:].clone(), step, max_iter, t1ce_logits[:,3:4,:,:,:].clone(), flair_logits[:,3:4,:,:,:].clone(), t1_logits[:,3:4,:,:,:].clone(), t2_logits[:,3:4,:,:,:].clone(),TP,TM)
                soft_target_sum = soft_target_1+soft_target_2+soft_target_3+soft_target_4
                soft_target = torch.cat([soft_target_1/soft_target_sum, soft_target_2/soft_target_sum, soft_target_3/soft_target_sum, soft_target_4/soft_target_sum], dim=1)

            soft_sep_cross_loss = torch.zeros(1).cuda().float()
            soft_sep_dice_loss = torch.zeros(1).cuda().float()


      
            soft_target_f,flair_logits = random_mask(soft_target,flair_logits, mask_rato)
            soft_sep_cross_loss += criterions.softmax_weighted_loss(flair_logits, soft_target_f, num_cls=num_cls)
            
            soft_target_t1c,t1ce_logits = random_mask(soft_target,t1ce_logits, mask_rato)
            soft_sep_cross_loss += criterions.softmax_weighted_loss(t1ce_logits, soft_target_t1c, num_cls=num_cls)

            soft_target_t1,t1_logits = random_mask(soft_target,t1_logits, mask_rato)
            soft_sep_cross_loss += criterions.softmax_weighted_loss(t1_logits, soft_target_t1, num_cls=num_cls)

            soft_target_t2,t2_logits = random_mask(soft_target,t2_logits, mask_rato)            
            soft_sep_cross_loss += criterions.softmax_weighted_loss(t2_logits, soft_target_t2, num_cls=num_cls)



            soft_sep_loss = soft_sep_cross_loss

            
            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss


            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss
            loss = fuse_loss + 0.7*sep_loss + prm_loss + soft_sep_loss*0.3*temp*temp/10

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('soft_sep_cross_loss', soft_sep_cross_loss.item(), global_step=step)
            writer.add_scalar('soft_sep_dice_loss', soft_sep_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += 'soft_sep_cross_loss:{:.4f}, soft_sep_dice_loss:{:.4f},'.format(soft_sep_cross_loss.item(), soft_sep_dice_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)
        
        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()

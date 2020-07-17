import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data

from dataset import IC15Trainset_PAN
from models import resnet50
from models import Model
from models.loss import dice_loss
from models.loss import get_pull_push_loss
from models.loss import get_pull_push_loss_PAN
from models.loss import PANLoss
from myutils import Logger
from myutils import AverageMeter
from myutils import RunningScore
from myutils import ohem_single, ohem_batch
from myutils import adjust_learning_rate_StepLR
from myutils import adjust_learning_rate_Poly
from myutils import PolynomialLR
import os
import sys
import time
# import pyclipper
# import Polygon as plg
import collections

import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text > 0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def cal_kernel_score(kernel, gt_kernel, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks.float()).data.cpu().numpy()
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel > 0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel




def train(model, trainloader, optimizer, epoch, scheduler, criterion):
    log_file = '/home/data1/zhm/ic15_PAN_res18_Poly_log.txt'
    print('Epoch:', epoch)
    with open(log_file, 'a') as f:
        f.write('Epoch:' + str(epoch) + '\n')
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernel = AverageMeter()
    # losses_top = AverageMeter()
    # losses_bot = AverageMeter()
    losses_pull = AverageMeter()
    losses_push = AverageMeter()
    current_time = time.time()
    running_metric_text = RunningScore(2)
    running_metric_kernel = RunningScore(2)
    for batch_idx, (imgs, gt_texts, gt_kernels, gt_texts_labeled, gt_kernels_labeled, training_masks) in enumerate(trainloader):
        t = time.time()
        data_time.update(t - current_time)
        # print(t - current_time)

        imgs = Variable(imgs.cuda())

        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        # gt_tops = Variable(gt_tops.cuda())
        # gt_bots = Variable(gt_bots.cuda())

        gt_texts_labeled = Variable(gt_texts_labeled.cuda())
        gt_kernels_labeled = Variable(gt_kernels_labeled.cuda())
        # gt_tops_labeled = Variable(gt_tops_labeled.cuda())
        # gt_bots_labeled = Variable(gt_bots_labeled.cuda())

        training_masks = Variable(training_masks.cuda())

        outputs = model(imgs)
        output_texts = outputs[:, 0, :, :]
        output_kernels = outputs[:, 1, :, :]
        # output_sim_vectors = outputs[:, 2:, :, :]
        #
        # selected_text_masks = ohem_batch(output_texts, gt_texts, training_masks)
        # selected_text_masks = Variable(selected_text_masks.cuda())
        #
        # mask_training = training_masks.data.cpu().numpy()
        # mask_gt_text = gt_texts.data.cpu().numpy()
        # mask_pred_text = torch.sigmoid(output_texts).data.cpu().numpy()
        # selected_kernel_masks = ((mask_training > 0.5) & (mask_gt_text > 0.5)).astype('float32')
        # selected_kernel_masks = torch.from_numpy(selected_kernel_masks).float()
        # selected_kernel_masks = Variable(selected_kernel_masks.cuda())
        # selected_kernel_masks = ohem_batch(output_kernels, gt_kernels, training_masks)

        # loss_pull, loss_push = get_pull_push_loss_PAN(outputs, gt_texts_labeled, gt_kernels_labeled)
        #
        # loss_text = dice_loss(output_texts, gt_texts, selected_text_masks)
        # loss_kernel = dice_loss(output_kernels, gt_kernels, selected_kernel_masks)
        #
        # loss = 1.0 * loss_text + 0.5 * loss_kernel + 0.25 * (loss_pull + loss_push)

        loss, loss_text, loss_kernel, loss_pull, loss_push = criterion(outputs, gt_texts_labeled, gt_kernels_labeled, training_masks)


        losses.update(loss.item(), imgs.shape[0])
        losses_text.update(loss_text.item(), imgs.shape[0])
        losses_kernel.update(loss_kernel.item(), imgs.shape[0])
        # losses_top.update(loss_top.item(), imgs.shape[0])
        # losses_bot.update(loss_bot.item(), imgs.shape[0])
        losses_pull.update(loss_pull.item(), imgs.shape[0])
        losses_push.update(loss_push.item(), imgs.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, PolynomialLR):
                # print('updating')
                scheduler.step()

        score_text = cal_text_score(texts=output_texts, gt_texts=gt_texts, training_masks=training_masks, running_metric_text=running_metric_text)
        score_kernel = cal_kernel_score(kernel=output_kernels, gt_kernel=gt_kernels, gt_texts=gt_texts, training_masks=training_masks,
                                        running_metric_kernel=running_metric_kernel)
        acc = score_text['Mean Acc']
        iou_text = score_text['Mean IoU']
        iou_kernel = score_kernel['Mean IoU']

        batch_time.update(time.time() - current_time)
        if (batch_idx + 1) % 20 == 0:
            output_log = '({batch}/{size}) Batch: {dt:.3f}s {bt:.3f}s {lr} | TOTAL: {total:.0f}s | ETA: {eta:.0f}s | Loss: {loss:.4f} |' \
                         ' {loss_text:.4f} | {loss_kernel:.4f} | {loss_top:.4f} | {loss_bot:.4f} | {loss_pull:.4f} | {loss_push:.4f} | acc:{acc:.4f} text_iou:{iou1:.4f} kernel_iou{iou2:.4f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                dt=data_time.avg,
                bt=batch_time.avg,
                lr=current_lr,
                total=batch_time.avg * batch_idx,
                eta=batch_time.avg * (len(trainloader) - batch_idx),
                loss=losses.avg,
                loss_text=losses_text.avg,
                loss_kernel=losses_kernel.avg,
                loss_top=0,
                loss_bot=0,
                loss_pull=losses_pull.avg,
                loss_push=losses_push.avg,
                acc=acc,
                iou1=iou_text,
                iou2=iou_kernel)
            print(output_log)
            sys.stdout.flush()
            with open(log_file, 'a') as f:
                f.write(output_log + '\n')
        current_time = time.time()
    return losses.avg

def main(args):
    num_classes = 6 # gt_text, gt_kernel, sim_vector
    trainset = IC15Trainset_PAN(kernel_scale=0.5, with_coord=False)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1,
                                              drop_last=True,
                                              pin_memory=True)
    # data_time = AverageMeter()
    # batch_time = AverageMeter()
    # current_time = time.time()
    # for i in range(5):
    #     for batch_idx, (imgs, gt_texts, gt_kernels, gt_texts_labeled, gt_kernels_labeled, training_masks) in enumerate(
    #             trainloader):
    #         data_time.update(time.time() - current_time)
    #         batch_time.update(time.time() - current_time)
    #         if (batch_idx + 1) % 10 == 0:
    #             print(data_time.avg)
    #             print(batch_time.avg)
    #         current_time = time.time()
    if args.backbone == 'res50':
        model = resnet50(pretrained=True, num_classes=num_classes)
    elif args.backbone == 'res18':
        model = Model()
    else:
        raise NotImplementedError

    max_epoch = 600
    start_epoch = 0
    start_lr = args.lr


    if args.resume is not None:
        if os.path.exists(args.resume):
            print('Resume From:', args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            start_lr = args.lr * (1 - start_epoch / max_epoch) ** 0.9
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print('No Such Checkpoint File at', args.resume)
    else:
        print('Training From the Beginning')
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    scheduler = PolynomialLR(optimizer=optimizer, max_iter=max_epoch * len(trainloader), power=0.9)
    # if args.resume is not None:
    #     scheduler.step(start_epoch * len(trainloader))
    criterion = PANLoss()
    min_loss = 1000
    for epoch in range(start_epoch, max_epoch):
        # adjust_learning_rate_StepLR(args, optimizer, epoch)
        loss = train(model, trainloader, optimizer, epoch, scheduler=scheduler, criterion=criterion)
        checkpoint_info = {'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'lr': args.lr,
                           'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_info, '/home/data1/zhm/ic15_PAN_res18_Poly.pth.tar')
        if loss < min_loss:
            min_loss = loss
            torch.save(checkpoint_info, '/home/data1/zhm/best_models/ic15_PAN_res18_Poly_best.pth.tar')



if __name__ == '__main__':
    seed_everything(911)
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    # parser.add_argument('--schedule', nargs='+', type=int, default=[200, 400])
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    args = parser.parse_args()
    main(args)
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data

from dataset import IC15Dataset
from models import resnet50
from models.loss import dice_loss
from myutils import Logger
from myutils import AverageMeter
from myutils import RunningScore
from myutils import ohem_single, ohem_batch
from myutils import adjust_learning_rate_StepLR
import os
import sys
import time
import pyclipper
import Polygon as plg

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text >  0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks).data.cpu().numpy()
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel >  0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel


def train(model, trainloader, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    running_metric_text = RunningScore(n_classes=2)
    running_metric_kernel = RunningScore(n_classes=2)
    start_time = time.time()
    for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(trainloader):

        data_time.update(time.time() - start_time)
        num_sample = imgs.shape[0]

        imgs = Variable(imgs.cuda())
        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        training_masks = Variable(training_masks.cuda())

        outputs = model(imgs)

        output_texts = outputs[:, 0, :, :]
        output_kernels = outputs[:, 1:, :, :]

        selected_masks = ohem_batch(output_texts, gt_texts, training_masks)
        selected_masks = Variable(selected_masks.cuda())

        loss_texts = criterion(output_texts, gt_texts, selected_masks)
        loss_kernels = []
        mask0 = torch.sigmoid(output_texts).data.cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks.cuda())
        for i in range(6):
            kernel_i = output_kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
            loss_kernels.append(loss_kernel_i)

        loss_kernels = sum(loss_kernels) / len(loss_kernels)
        loss = 0.7 * loss_texts + 0.3 * loss_kernels

        losses.update(loss.item(), n=num_sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score_text = cal_text_score(output_texts, gt_texts, training_masks, running_metric_text)
        score_kernel = cal_kernel_score(output_kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        # TODO: use the logger to save log information
        if batch_idx % 50 == 0:
            output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(trainloader) - batch_idx) / 60.0,
                loss=losses.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            print(output_log)
            sys.stdout.flush()


    return losses.avg, score_text['Mean Acc'], score_kernel['Mean Acc'], score_text['Mean IoU'], score_kernel['Mean IoU']

def main(args):
    num_kernel = 7
    min_scale = 0.4
    trainset = IC15Dataset(num_kernel=num_kernel)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=1,
                                              drop_last=True,
                                              pin_memory=True)
    if args.backbone == 'res50':
        model = resnet50(pretrained=True, num_classes=num_kernel)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    logger = Logger('log.txt', title='icdar2015')
    logger.set_names(['Learning-Rate', 'Train-Loss', 'Train-Acc', 'Train-IOU'])

    n_epoch = 100
    for epoch in range(n_epoch):
        # ajust learning rate
        adjust_learning_rate_StepLR(args, optimizer, epoch)
        avg_losses, text_acc, kernel_acc, text_iou, kernel_iou = train(model, trainloader, dice_loss, optimizer, epoch)
        # TODO: save the checkpoint and the corresponding training info
        checkpoint_info = {'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'lr': args.lr,
                           'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_info, '/home/data1/zhm/checkpoint.pth.tar')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    parser.add_argument('--schedule', nargs='+', type=int, default=[50, 100])
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)

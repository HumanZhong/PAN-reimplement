import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data

from dataset import CTW1500Trainset_Bound
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
# import pyclipper
# import Polygon as plg

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
    print('Epoch:', epoch)
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernel = AverageMeter()
    losses_top = AverageMeter()
    losses_bot = AverageMeter()
    # losses_top_left = AverageMeter()
    # losses_top_right = AverageMeter()
    # losses_bot_right = AverageMeter()
    # losses_bot_left = AverageMeter()
    running_metric_kernel = RunningScore(2)
    current_time = time.time()
    # for batch_idx, (imgs, gt_texts, gt_kernels, gt_top_lefts, gt_top_rights, gt_bot_rights, gt_bot_lefts, training_masks) in enumerate(trainloader):
    for batch_idx, (imgs, gt_texts, gt_kernels, gt_tops, gt_bots, training_masks) in enumerate(trainloader):
        data_time.update(time.time() - current_time)

        imgs = Variable(imgs.cuda())
        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())

        gt_tops = Variable(gt_tops.cuda())
        gt_bots = Variable(gt_bots.cuda())

        # gt_top_lefts = Variable(gt_top_lefts.cuda())
        # gt_top_rights = Variable(gt_top_rights.cuda())
        # gt_bot_rights = Variable(gt_bot_rights.cuda())
        # gt_bot_lefts = Variable(gt_bot_lefts.cuda())

        training_masks = Variable(training_masks.cuda())

        # i_channels = Variable(i_channels.cuda())
        # j_channels = Variable(j_channels.cuda())

        outputs = model(imgs)
        output_texts = outputs[:, 0, :, :]
        output_kernels = outputs[:, 1, :, :]

        output_tops = outputs[:, 2, :, :]
        output_bots = outputs[:, 3, :, :]

        # output_top_lefts = outputs[:, 2, :, :]
        # output_top_rights = outputs[:, 3, :, :]
        # output_bot_rights = outputs[:, 4, :, :]
        # output_bot_lefts = outputs[:, 5, :, :]

        # attention: -----------------generating training masks for each part---------------------
        selected_text_masks = ohem_batch(output_texts, gt_texts, training_masks)
        selected_text_masks = Variable(selected_text_masks.cuda())

        # TODO: think twice whether to use ohem or the method used in the original PSENet paper
        selected_kernel_masks = ohem_batch(output_kernels, gt_kernels, training_masks)
        selected_kernel_masks = Variable(selected_kernel_masks.cuda())
        # mask_training = training_masks.data.cpu().numpy()

        # selected_top_masks = ohem_batch(output_tops, gt_tops, training_masks)

        mask_training = training_masks.data.cpu().numpy()
        mask_gt_top = gt_tops.data.cpu().numpy()
        mask_gt_text = gt_texts.data.cpu().numpy()
        mask_pred_top = torch.sigmoid(output_tops).data.cpu().numpy()
        selected_top_masks = ((mask_training > 0.5) & ((mask_gt_top > 0.5) | (mask_gt_text > 0.5) | (mask_pred_top > 0.5))).astype('float32')
        selected_top_masks = torch.from_numpy(selected_top_masks).float()
        selected_top_masks = Variable(selected_top_masks.cuda())

        # selected_bot_masks = ohem_batch(output_bots, gt_bots, training_masks)

        mask_training = training_masks.data.cpu().numpy()
        mask_gt_bot = gt_bots.data.cpu().numpy()
        mask_gt_text = gt_texts.data.cpu().numpy()
        mask_pred_bot = torch.sigmoid(output_bots).data.cpu().numpy()
        selected_bot_masks = ((mask_training > 0.5) & ((mask_gt_bot > 0.5) | (mask_gt_text > 0.5) | (mask_pred_bot > 0.5))).astype('float32')
        selected_bot_masks = torch.from_numpy(selected_bot_masks).float()
        selected_bot_masks = Variable(selected_bot_masks.cuda())

        # mask_training = training_masks.data.cpu().numpy()
        # mask_gt_top_left = gt_top_lefts.data.cpu().numpy()
        # mask_gt_top_right = gt_top_rights.data.cpu().numpy()
        # mask_gt_bot_right = gt_bot_rights.data.cpu().numpy()
        # mask_gt_bot_left = gt_bot_lefts.data.cpu().numpy()
        # mask_gt_text = gt_texts.data.cpu().numpy()

        # selected_masks_top_left = ((mask_training > 0.5) & ((mask_gt_top_left > 0.5) | (mask_gt_text > 0.5))).astype('float32')
        # selected_masks_top_left = torch.from_numpy(selected_masks_top_left).float()
        # selected_masks_top_left = Variable(selected_masks_top_left.cuda())
        #
        # selected_masks_top_right = ((mask_training > 0.5) & ((mask_gt_top_right > 0.5) | (mask_gt_text > 0.5))).astype('float32')
        # selected_masks_top_right = torch.from_numpy(selected_masks_top_right).float()
        # selected_masks_top_right = Variable(selected_masks_top_right.cuda())
        #
        # selected_masks_bot_right = ((mask_training > 0.5) & ((mask_gt_bot_right > 0.5) | (mask_gt_text > 0.5))).astype('float32')
        # selected_masks_bot_right = torch.from_numpy(selected_masks_bot_right).float()
        # selected_masks_bot_right = Variable(selected_masks_bot_right.cuda())
        #
        # selected_masks_bot_left = ((mask_training > 0.5) & ((mask_gt_bot_left > 0.5) | (mask_gt_text > 0.5))).astype('float32')
        # selected_masks_bot_left = torch.from_numpy(selected_masks_bot_left).float()
        # selected_masks_bot_left = Variable(selected_masks_bot_left.cuda())

        # TODO: to complete the whole project, an embedding vector and its corresponding loss is needed(should be done before 04.01)



        loss_text = criterion(output_texts, gt_texts, selected_text_masks)
        loss_kernel = criterion(output_kernels, gt_kernels, selected_kernel_masks)

        loss_top = criterion(output_tops, gt_tops, selected_top_masks)
        loss_bot = criterion(output_bots, gt_bots, selected_bot_masks)

        # loss_top_left = criterion(output_top_lefts, gt_top_lefts, selected_masks_top_left)
        # loss_top_right = criterion(output_top_rights, gt_top_rights, selected_masks_top_right)
        # loss_bot_right = criterion(output_bot_rights, gt_bot_rights, selected_masks_bot_right)
        # loss_bot_left = criterion(output_bot_lefts, gt_bot_lefts, selected_masks_bot_left)

        # loss = 0.2 * loss_kernel + 1.0 * loss_top + 1.0 * loss_bot
        loss = 1.0 * loss_text + 0.2 * loss_kernel + 1.0 * (loss_top + loss_bot)
        # loss = 1.0 * loss_text + 0.2 * loss_kernel + 0.5 * (loss_top_left + loss_top_right + loss_bot_right + loss_bot_left)

        losses.update(loss.item(), imgs.shape[0])
        losses_text.update(loss_text.item(), imgs.shape[0])
        losses_kernel.update(loss_kernel.item(), imgs.shape[0])
        losses_top.update(loss_top.item(), imgs.shape[0])
        losses_bot.update(loss_bot.item(), imgs.shape[0])
        # losses_top_left.update(loss_top_left.item(), imgs.shape[0])
        # losses_top_right.update(loss_top_right.item(), imgs.shape[0])
        # losses_bot_right.update(loss_bot_right.item(), imgs.shape[0])
        # losses_bot_left.update(loss_bot_left.item(), imgs.shape[0])


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # score_kernel = cal_kernel_score(output_kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)
        batch_time.update(time.time() - current_time)
        if (batch_idx + 1) % 20 == 0:
            output_log = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}s | ETA: {eta:.0f}s | Loss: {loss:.4f} | {loss_text:.4f} | {loss_kernel:.4f} | {loss_top:.4f} | {loss_bot:.4f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx,
                eta=batch_time.avg * (len(trainloader) - batch_idx),
                loss=losses.avg,
                loss_text=losses_text.avg,
                loss_kernel=losses_kernel.avg,
                loss_top=losses_top.avg,
                loss_bot=losses_bot.avg)
            print(output_log)
            sys.stdout.flush()
        # if (batch_idx + 1) % 20 == 0:
        #     output_log = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}s | ETA: {eta:.0f}s | Loss: {loss:.4f} |' \
        #                  ' {loss_text:.4f} | {loss_kernel:.4f} | {loss_top_left:.4f} | {loss_top_right:.4f} | {loss_bot_right:.4f} |' \
        #                  ' {loss_bot_left:.4f}'.format(
        #         batch=batch_idx + 1,
        #         size=len(trainloader),
        #         bt=batch_time.avg,
        #         total=batch_time.avg * batch_idx,
        #         eta=batch_time.avg * (len(trainloader) - batch_idx),
        #         loss=losses.avg,
        #         loss_text=losses_text.avg,
        #         loss_kernel=losses_kernel.avg,
        #         loss_top_left=losses_top_left.avg,
        #         loss_top_right=losses_top_right.avg,
        #         loss_bot_right=losses_bot_right.avg,
        #         loss_bot_left=losses_bot_left.avg)
        #     print(output_log)
        #     sys.stdout.flush()
        current_time = time.time()




def main(args):
    num_classes = 4 # gt_text, gt_kernel, gt_top, gt_bot
    # num_classes = 6 # gt_text, gt_kernel, gt_top_left, gt_top_right, gt_bot_right, gt_bot_left
    trainset = CTW1500Trainset_Bound(with_coord=False)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=1,
                                              drop_last=True,
                                              pin_memory=True)
    if args.backbone == 'res50':
        model = resnet50(pretrained=True, num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    n_epoch = 300
    for epoch in range(n_epoch):
        adjust_learning_rate_StepLR(args, optimizer, epoch)
        # TODO: train func
        _ = train(model, trainloader, dice_loss, optimizer, epoch)
        checkpoint_info = {'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'lr': args.lr,
                           'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_info, '/home/data1/zhm/ctw_purebound_checkpoint_erodedwidth20_rotate_flip_jitter_0402_300e.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    parser.add_argument('--schedule', nargs='+', type=int, default=[150, 240])
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)


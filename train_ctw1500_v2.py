import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data

from dataset import CTW1500Trainset_BoundE2E
from models import resnet50
from models.loss import dice_loss
from models.loss import get_pull_push_loss
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


def train(model, trainloader, criterion, optimizer, epoch, scheduler):
    log_file = '/home/data1/zhm/ctw_purebound_checkpoint_poly_0417_baseline_predmask_nobound_moreaug_600e_log.txt'
    print('Epoch:', epoch)
    with open(log_file, 'a') as f:
        f.write('Epoch: ' + str(epoch) + '\n')
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernel = AverageMeter()
    losses_top = AverageMeter()
    losses_bot = AverageMeter()
    losses_pull = AverageMeter()
    losses_push = AverageMeter()
    # losses_top_left = AverageMeter()
    # losses_top_right = AverageMeter()
    # losses_bot_right = AverageMeter()
    # losses_bot_left = AverageMeter()
    running_metric_kernel = RunningScore(2)
    current_time = time.time()
    # for batch_idx, (imgs, gt_texts, gt_kernels, gt_top_lefts, gt_top_rights, gt_bot_rights, gt_bot_lefts, training_masks) in enumerate(trainloader):
    for batch_idx, (imgs, gt_texts, gt_kernels, gt_tops, gt_bots, gt_texts_labeled, gt_kernels_labeled, gt_tops_labeled, gt_bots_labeled, training_masks) in enumerate(trainloader):
        data_time.update(time.time() - current_time)

        imgs = Variable(imgs.cuda())

        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        gt_tops = Variable(gt_tops.cuda())
        gt_bots = Variable(gt_bots.cuda())

        gt_texts_labeled = Variable(gt_texts_labeled.cuda())
        gt_kernels_labeled = Variable(gt_kernels_labeled.cuda())
        gt_tops_labeled = Variable(gt_tops_labeled.cuda())
        gt_bots_labeled = Variable(gt_bots_labeled.cuda())

        training_masks = Variable(training_masks.cuda())

        # i_channels = Variable(i_channels.cuda())
        # j_channels = Variable(j_channels.cuda())

        outputs = model(imgs)
        output_texts = outputs[:, 0, :, :]
        output_kernels = outputs[:, 1, :, :]
        output_tops = outputs[:, 2, :, :]
        output_bots = outputs[:, 3, :, :]
        output_sim_vectors = outputs[:, 4:, :, :]


        # attention: -----------------generating training masks for each part---------------------
        selected_text_masks = ohem_batch(output_texts, gt_texts, training_masks)
        selected_text_masks = Variable(selected_text_masks.cuda())

        # TODO: think twice whether to use ohem or the method used in the original PSENet paper
        mask_training = training_masks.data.cpu().numpy()
        mask_gt_text = gt_texts.data.cpu().numpy()
        mask_pred_text = output_texts.data.cpu().numpy()
        selected_kernel_masks = ((mask_training > 0.5) & (mask_pred_text > 0.5)).astype('float32')
        selected_kernel_masks = torch.from_numpy(selected_kernel_masks).float()
        selected_kernel_masks = Variable(selected_kernel_masks.cuda())
        # selected_kernel_masks = ohem_batch(output_kernels, gt_kernels, training_masks)


        # selected_top_masks = ohem_batch(output_tops, gt_tops, training_masks)

        # mask_training = training_masks.data.cpu().numpy()
        # mask_gt_top = gt_tops.data.cpu().numpy()
        # mask_gt_text = gt_texts.data.cpu().numpy()
        # mask_pred_top = torch.sigmoid(output_tops).data.cpu().numpy()
        # selected_top_masks = ((mask_training > 0.5) & ((mask_gt_top > 0.5) | (mask_gt_text > 0.5) | (mask_pred_top > 0.5))).astype('float32')
        # selected_top_masks = torch.from_numpy(selected_top_masks).float()
        # selected_top_masks = Variable(selected_top_masks.cuda())

        # selected_bot_masks = ohem_batch(output_bots, gt_bots, training_masks)

        # mask_training = training_masks.data.cpu().numpy()
        # mask_gt_bot = gt_bots.data.cpu().numpy()
        # mask_gt_text = gt_texts.data.cpu().numpy()
        # mask_pred_bot = torch.sigmoid(output_bots).data.cpu().numpy()
        # selected_bot_masks = ((mask_training > 0.5) & ((mask_gt_bot > 0.5) | (mask_gt_text > 0.5) | (mask_pred_bot > 0.5))).astype('float32')
        # selected_bot_masks = torch.from_numpy(selected_bot_masks).float()
        # selected_bot_masks = Variable(selected_bot_masks.cuda())


        # TODO: to complete the whole project, an embedding vector and its corresponding loss is needed(should be done before 04.01)
        loss_pull, loss_push = get_pull_push_loss(outputs, gt_texts_labeled, gt_kernels_labeled, gt_tops_labeled, gt_bots_labeled)


        loss_text = criterion(output_texts, gt_texts, selected_text_masks)
        loss_kernel = criterion(output_kernels, gt_kernels, selected_kernel_masks)

        # loss_top = criterion(output_tops, gt_tops, selected_top_masks)
        # loss_bot = criterion(output_bots, gt_bots, selected_bot_masks)

        # loss = 0.2 * loss_kernel + 1.0 * loss_top + 1.0 * loss_bot
        # loss = 1.0 * loss_text + 0.5 * loss_kernel + 1.0 * (loss_top + loss_bot) + 0.25 * (loss_pull + loss_push)
        loss = 1.0 * loss_text + 0.5 * loss_kernel + 0.25 * (loss_pull + loss_push)

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
        if isinstance(scheduler, PolynomialLR):
            # print('updating')
            scheduler.step()

        # score_kernel = cal_kernel_score(output_kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)
        batch_time.update(time.time() - current_time)
        if (batch_idx + 1) % 20 == 0:
            output_log = '({batch}/{size}) Batch: {dt:.3f}s {bt:.3f}s {lr} | TOTAL: {total:.0f}s | ETA: {eta:.0f}s | Loss: {loss:.4f} |' \
                         ' {loss_text:.4f} | {loss_kernel:.4f} | {loss_top:.4f} | {loss_bot:.4f} | {loss_pull:.4f} | {loss_push:.4f}'.format(
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
                loss_push=losses_push.avg)
            print(output_log)
            sys.stdout.flush()
            with open(log_file, 'a') as f:
                f.write(output_log + '\n')
        current_time = time.time()




def main(args):
    num_classes = 8 # gt_text, gt_kernel, gt_top, gt_bot, sim_vector(n_channels:4)
    trainset = CTW1500Trainset_BoundE2E(with_coord=False)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1,
                                              drop_last=True,
                                              pin_memory=True)
    if args.backbone == 'res50':
        model = resnet50(pretrained=True, num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    n_epoch = 600
    scheduler = PolynomialLR(optimizer=optimizer, max_iter=n_epoch * len(trainloader), power=0.9)
    for epoch in range(n_epoch):
        # adjust_learning_rate_StepLR(args, optimizer, epoch)
        # adjust_learning_rate_Poly(args, 1e-3, optimizer, epoch, n_epoch, 0.9)
        # TODO: train func
        _ = train(model, trainloader, dice_loss, optimizer, epoch, scheduler)
        checkpoint_info = {'epoch': epoch + 1,
                           'state_dict': model.state_dict(),
                           'lr': args.lr,
                           'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_info, '/home/data1/zhm/ctw_purebound_checkpoint_poly_0417_baseline_predmask_nobound_moreaug_600e.pth.tar')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    # parser.add_argument('--schedule', nargs='+', type=int, default=[200, 400])
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)


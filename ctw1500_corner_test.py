import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data

from pypse import pypse
from dataset import CTW1500Testset_Bound
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
import collections
import pyclipper
import Polygon as plg
import cv2
from tqdm import tqdm

def generate_img_result(image, result_filename, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    result_filepath = os.path.join(root_path, 'result_%s.jpg'%(result_filename))
    cv2.imwrite(result_filepath, image)

def test(args):
    testset = CTW1500Testset_Bound()
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             drop_last=True)
    if args.backbone == 'res50':
        model = resnet50(pretrained=True, num_classes=6)

    for param in model.parameters():
        param.requires_grad = False
    model = model.cuda()

    if args.resume is not None:
        if os.path.exists(args.resume):
            print('Load from', args.resume)
            checkpoint = torch.load(args.resume)
            # 这里为什么不直接用model.load_state_dict(checkpoint['state_dict'])
            # 是因为训练时使用多卡训练，模型中各个参数的名字前面有个前缀，需要去除该前缀
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print('No such checkpoint file at', args.resume)

    model.eval()

    for idx, (img, original_img) in tqdm(enumerate(testloader)):
        img = Variable(img.cuda())

        original_img = original_img.numpy().astype('uint8')[0]
        original_img = original_img.copy()

        outputs = model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - 1.0) + 1) / 2

        output_text = outputs[:, 0, :, :]
        kernel = outputs[:, 1, :, :]
        # top = outputs[:, 2, :, :]
        # bot = outputs[:, 3, :, :]
        top_left = outputs[:, 2, :, :]
        top_right = outputs[:, 3, :, :]
        bot_right = outputs[:, 4, :, :]
        bot_left = outputs[:, 5, :, :]

        output_kernel = outputs[:, 1, :, :] * output_text

        # output_top = outputs[:, 2, :, :] * output_text
        # output_bot = outputs[:, 3, :, :] * output_text
        output_top_left = top_left * output_text
        output_top_right = top_right * output_text
        output_bot_right = bot_right * output_text
        output_bot_left = bot_left * output_text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        output_text = output_text.data.cpu().numpy()[0].astype(np.uint8)
        output_kernel = output_kernel.data.cpu().numpy().astype(np.uint8)
        # output_top = output_top.data.cpu().numpy().astype(np.uint8)
        # output_bot = output_bot.data.cpu().numpy().astype(np.uint8)
        output_top_left = output_top_left.data.cpu().numpy().astype(np.uint8)
        output_top_right = output_top_right.data.cpu().numpy().astype(np.uint8)
        output_bot_right = output_bot_right.data.cpu().numpy().astype(np.uint8)
        output_bot_left = output_bot_left.data.cpu().numpy().astype(np.uint8)


        kernel = kernel.data.cpu().numpy()[0].astype(np.uint8)
        # top = top.data.cpu().numpy().astype(np.uint8)
        # bot = bot.data.cpu().numpy().astype(np.uint8)


        # pred = pypse(output_kernel, 10)

        pred_top = pypse(output_top_left, 0, connectivity=8)

        # scale = (original_img.shape[1] / pred.shape[1], original_img.shape[0] / pred.shape[0])
        # bboxes = []
        # num_label = np.max(pred) + 1
        # for i in range(1, num_label):
        #     points_loc = np.array(np.where(pred == i)).transpose((1, 0))
        #     # points = points[:,::-1]
        #     if points_loc.shape[0] < 300:
        #         continue
        #     score_i = np.mean(score[pred == i])
        #     if score_i < 0.93:
        #         continue
        #
        #     binary = np.zeros(pred.shape, dtype='uint8')
        #     binary[pred == i] = 1
        #
        #     contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     contour = contours[0]
        #     contour = contour * scale
        #     if contour.shape[0] <= 2:
        #         continue
        #     contour = contour.astype('int32')
        #     bboxes.append(contour.reshape(-1))


        scale = (original_img.shape[1] / pred_top.shape[1], original_img.shape[0] / pred_top.shape[0])

        # pred_top = np.reshape(pred_top, (pred_top.shape[0], pred_top.shape[1], 1))
        # pred_top = cv2.resize(pred_top, dsize=(original_img.shape[0], original_img.shape[1]))
        # pred_top = pred_top[:, :, 0]
        # original_img[pred_top > 0.5, :] = (0, 0, 255)


        bboxes = []
        num_label = np.max(pred_top) + 1
        for i in range(1, num_label):
            points_loc = np.array(np.where(pred_top == i)).transpose((1, 0))
            # points = points[:,::-1]
            # if points_loc.shape[0] < 1:
            #     continue
            # score_i = np.mean(score[pred_top == i])
            # if score_i < 0.93:
            #     continue

            binary = np.zeros(pred_top.shape, dtype='uint8')
            binary[pred_top == i] = 1

            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            contour = contour * scale
            if contour.shape[0] <= 2:
                continue
            contour = contour.astype('int32')
            bboxes.append(contour.reshape(-1))


        torch.cuda.synchronize()
        for bbox in bboxes:
            cv2.drawContours(original_img, [bbox.reshape(int(bbox.shape[0] / 2), 2)], -1, (0, 0, 255), 1)
        image_name = testset.img_paths[idx].split('/')[-1].split('.')[0]
        # generate_txt_result_ctw(bboxes, image_name, 'outputs/result_ctw_txt_wh_new')
        generate_img_result(original_img, image_name, 'outputs/result_ctw_img_my_top_left')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    # parser.add_argument('--resume', nargs='?', type=str, default='/home/data1/zhm/ctw_bound_full_new_checkpoint_0322_200e.pth.tar')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/data1/zhm/ctw_corner_checkpoint_0325_100e.pth.tar')

    args = parser.parse_args()
    test(args)
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data

from pypse import pypse
from dataset import CTW1500Testset
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
#from pse import pse

def generate_txt_result_ctw(bboxes, result_filename, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    result_filepath = os.path.join(root_path, 'result_%s.txt'%(result_filename))
    with open(result_filepath, 'w') as f:
        lines = []
        for bbox in bboxes:
            bbox = [int(v) for v in bbox]
            #line = '%d, %d, %d, %d, %d, %d, %d, %d\n' % tuple(bbox)
            line = '%d'%bbox[0]
            for idx in range(1, len(bbox)):
                line += ', %d'%bbox[idx]
            line += '\n'
            lines.append(line)
        f.writelines(lines)

def generate_img_result(image, result_filename, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    result_filepath = os.path.join(root_path, 'result_%s.jpg'%(result_filename))
    cv2.imwrite(result_filepath, image)


def test(args):
    testset = CTW1500Testset()
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             drop_last=True)
    if args.backbone == 'res50':
        model = resnet50(pretrained=True, num_classes=7)

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
        output_kernels= outputs[:, 0:3, :, :] * output_text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        output_text = output_text.data.cpu().numpy()[0].astype(np.uint8)
        output_kernels = output_kernels.data.cpu().numpy()[0].astype(np.uint8)

        pred = pypse(output_kernels, 10)
        # pred = pse(output_kernels, 10)

        scale = (original_img.shape[1] / pred.shape[1], original_img.shape[0] / pred.shape[0])
        bboxes = []
        num_label = np.max(pred) + 1
        for i in range(1, num_label):
            points_loc = np.array(np.where(pred == i)).transpose((1, 0))
            # points = points[:,::-1]
            if points_loc.shape[0] < 300:
                continue
            score_i = np.mean(score[pred == i])
            if score_i < 0.93:
                continue

            binary = np.zeros(pred.shape, dtype='uint8')
            binary[pred == i] = 1

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
        generate_txt_result_ctw(bboxes, image_name, 'outputs/result_ctw_txt_wh_new')
        generate_img_result(original_img, image_name, 'outputs/result_ctw_img_wh_new')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/data1/zhm/pretrained_models/ctw1500_res50.pth.tar')
    args = parser.parse_args()
    test(args)
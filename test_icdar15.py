import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils import data

from pypse import pypse
from dataset import IC15Dataset
from dataset import IC15TestDataset
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

def generate_txt_result(bboxes, result_filename, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    result_filepath = os.path.join(root_path, 'result_%s.txt'%(result_filename))
    with open(result_filepath, 'w') as f:
        lines = []
        for bbox in bboxes:
            bbox = [int(v) for v in bbox]
            line = '%d, %d, %d, %d, %d, %d, %d, %d\n' % tuple(bbox)
            lines.append(line)
        f.writelines(lines)

def generate_img_result(image, result_filename, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    result_filepath = os.path.join(root_path, 'result_%s.jpg'%(result_filename))
    cv2.imwrite(result_filepath, image)

def test(args):
    testset = IC15TestDataset()
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
        original_img = original_img.numpy().astype('uint8')[0]
        img = Variable(img.cuda(), volatile=True)

        torch.cuda.synchronize()
        outputs = model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])

        outputs = (torch.sign(outputs - 1.0) + 1) / 2
        output_text = outputs[:, 0, :, :]

        # attention: 为什么这里是从0开始，是因为output_text可以看作是最外层的kernel，而所有kernel在后续的pse过程都会被使用到，因此这里是从0开始
        output_kernels = outputs[:, 0:7, :, :] * output_text
        #output_kernels2 = outputs[:, 1:, :, :] * output_text

        score = score.data.cpu().numpy()[0].astype(np.float)
        output_text = output_text.data.cpu().numpy()[0].astype(np.int8)
        output_kernels = output_kernels.data.cpu().numpy()[0].astype(np.int8)
        #output_kernels2 = output_kernels2.data.cpu().numpy()[0].astype(np.int8)
        # TODO: implement the Progressive Scale Expansion algorithm
        pred = pypse(kernels=output_kernels, min_area=4.0)
        #pred2 = pypse(kernels=output_kernels2, min_area=4.0)


        num_label = np.max(pred) + 1
        bboxes = []
        for i in range(1, num_label):
            points_loc = np.array(np.where(pred == i)).transpose((1, 0))
            #这段代码有什么用吗？
            #points_loc = points_loc[:, ::-1]
            if points_loc.shape[0] < 4:
                continue
            if np.mean(score[pred == i]) < 0.9:
                continue
            rect = cv2.minAreaRect(points_loc)
            bbox = cv2.boxPoints(rect).astype('int32')
            bboxes.append(bbox.reshape(-1))
        torch.cuda.synchronize()
        image_name = testset.img_paths[idx].split('/')[-1].split('.')[0]
        generate_txt_result(bboxes, image_name, 'outputs/result_ic15_txt_wh')
        output_img = None
        for bbox in bboxes:
            cv2.drawContours(original_img, [bbox.reshape(4, 2)[:, [1, 0]]], -1, (0, 0, 255), 1)
        generate_img_result(original_img, image_name, 'outputs/result_ic15_img_wh')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/data3/human/ic15_res50.pth.tar')
    args = parser.parse_args()
    test(args)
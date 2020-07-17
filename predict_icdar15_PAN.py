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
from dataset import IC15TestDataset
from models import resnet50
from models.post_processing import generate_result_purebound
from models.post_processing import generate_result_purebound_v2
from models.post_processing import generate_result_purebound_baseline
from models.post_processing import generate_result_PAN
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
    # if os.path.exists(result_filepath):
    #     return
    cv2.imwrite(result_filepath, image)

def generate_txt_result_PAN(bboxes, result_filename, root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    result_filepath = os.path.join(root_path, 'res_%s.txt'%(result_filename))
    # if os.path.exists(result_filepath):
    #     return
    with open(result_filepath, 'w') as f:
        lines = []
        for bbox in bboxes:
            # bbox = np.reshape(bbox, (20, ))
            bbox = [int(v) for v in bbox]
            #line = '%d, %d, %d, %d, %d, %d, %d, %d\n' % tuple(bbox)
            line = '%d'%bbox[0]
            for idx in range(1, len(bbox)):
                line += ', %d'%bbox[idx]
            line += '\n'
            lines.append(line)
        f.writelines(lines)



def predict(args):
    testset = IC15TestDataset()
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             drop_last=True)
    if args.backbone == 'res50':
        model = resnet50(pretrained=True, num_classes=6)
    else:
        raise NotImplementedError
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

        bboxes = generate_result_PAN(outputs, original_img, threshold=1.0)

        for i in range(len(bboxes)):
            bboxes[i] = bboxes[i].reshape(4, 2)[:, [1, 0]].reshape(-1)

        for bbox in bboxes:
            cv2.drawContours(original_img, [bbox.reshape(4, 2)], -1, (0, 255, 0), 1)

        image_name = testset.img_paths[idx].split('/')[-1].split('.')[0]
        generate_txt_result_PAN(bboxes, image_name, 'outputs/result_ic15_txt_PAN_res50fpn_Polyv2_4_85')
        generate_img_result(original_img, image_name, 'outputs/result_ic15_img_PAN_res50fpn_Polyv2_4_85')

    cmd = 'cd %s;zip -j %s %s/*' % ('./outputs/', 'submit_ic15.zip', 'result_txt_ic15_PAN_baseline');
    print(cmd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', nargs='?', type=str, default='res50')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/data1/zhm/ic15_PAN_res50fpn_Polyv2.pth.tar')
    args = parser.parse_args()

    predict(args)

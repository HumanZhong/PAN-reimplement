import numpy as np
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import os
import scipy.io as scio

def read_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path + 'not found')
    return img

def resize_syntext(img, word_bbox):
    h, w = img.shape[0:2]


def generate_bound_from_4bbox(bbox):
    pass
def generate_bound_from_14bbox():
    pass



class SynthText_Trainset(data.Dataset):
    root_path = "/home/data1/zhm/dataset/SynthText"

    def __init__(self):
        gt_path = os.path.join(self.root_path, "gt.mat")
        gt_info = scio.loadmat(gt_path)
        self.is_train = True
        self.gt = {}
        if self.is_train:
            self.gt["txt"] = gt_info["txt"][0][0:-1][:-10000]
            self.gt["imnames"] = gt_info["imnames"][0][:-10000]
            self.gt["charBB"] = gt_info["charBB"][0][:-10000]
            self.gt["wordBB"] = gt_info["wordBB"][0][:-10000]


    def __len__(self):
        return self.gt["txt"].shape[0]

    def __getitem__(self, index):
        img_name = self.gt["imnames"][index][0]
        img_path = os.path.join(self.root_path, img_name)
        img = read_image(img_path)

        # transpose the format from 2x4xn_word to n_wordx4x2
        word_bbox = self.gt["wordBB"][index].transpose(2, 1, 0)


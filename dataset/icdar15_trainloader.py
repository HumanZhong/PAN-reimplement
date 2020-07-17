import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
#import util
import os

import time
from myutils.misc import AverageMeter
import skimage.color as color

# random.seed(911)

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    random_scale_rate = np.array([0.5, 1.0, 2.0])
    scale = np.random.choice(random_scale_rate)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img, scale

def random_horizontal_flip(imgs):
    random_num = random.random()
    # print(random_num)
    if random_num < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    # print(angle)
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def random_crop(img_infos, img_size):
    h, w = img_infos[0].shape[0:2]
    # print(h, w)
    th, tw = img_size, img_size
    if th == h and tw == w:
        return img_infos
    random_num = random.random()
    # random_num = 0.1
    # print(random_num)

    if random_num > 3.0 / 8.0 and np.max(img_infos[1]) > 0:
        tl = np.min(np.where(img_infos[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        # br = np.max(np.where(img_infos[1] > 0), axis=1) - img_size
        br = np.max(np.where(img_infos[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        # print(i, j)
    for idx in range(len(img_infos)):
        if len(img_infos[idx].shape) == 3:
            img_infos[idx] = img_infos[idx][i:i + th, j:j + tw, :]
        else:
            img_infos[idx] = img_infos[idx][i:i + th, j:j + tw]
    return img_infos


def scale_shortside(img, short_size=320):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    resize_h = h * scale
    resize_w = w * scale
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    scale_x = resize_w / w
    scale_y = resize_h / h
    img = cv2.resize(img, dsize=None, fx=scale_x, fy=scale_y)
    return img

def read_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path + 'not found')
    return img

def read_anno(img, anno_path):
    h, w = img.shape[0:2]
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        bboxes = []
        tags = []
        for line in lines:
            line = line.replace('\ufeff', '')
            info = line.split(sep=',')
            if info[-1][0] == '#':
                tags.append(False)
                # print('#')
            elif info[-1][0] == '*':
                tags.append(False)
                # print('*')
            else:
                tags.append(True)
            bbox = [int(info[i]) for i in range(8)]
            bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 4)
            bboxes.append(bbox)
        return np.array(bboxes), tags

def read_anno_bound(img, anno_path):
    h, w = img.shape[0:2]
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        top_bounds = []
        bot_bounds = []
        bboxes = []
        tags = []
        for line in lines:
            line = line.replace('\ufeff', '')
            info = line.split(',')
            if info[-1][0] == '#':
                tags.append(False)
            else:
                tags.append(True)
            bbox = [np.int(info[i]) for i in range(len(info))]
            bbox = np.asarray(bbox)
            bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 4)
            top_bound = bbox[:4]
            bot_bound = bbox[4:]
            bboxes.append(bbox)
            top_bounds.append(top_bound)
            bot_bounds.append(bot_bound)
            tags.append(True)
    return np.array(bboxes), np.array(top_bounds), np.array(bot_bounds), tags


def dist(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def perimeter(bbox):
    '''
    calculate the perimeter of the current bbox
    :param bbox: shape(num_point, 2)
    :return: the perimeter of this bbox
    '''
    peri = 0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pyco = pyclipper.PyclipperOffset()
        pyco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        # offset = min((int)((area * (1 - rate * rate)) / (peri + 0.001) + 0.5), max_shr)
        # offset = min((int)((area * (1 - rate)) / (peri + 0.001) + 0.5), max_shr)
        offset = (int)((area * (1 - rate)) / (peri + 0.001) + 0.5)
        # offset = (int)((area * (1 - rate * rate)) / (peri + 0.001) + 0.5)
        shrinked_bbox = pyco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bboxes.append(shrinked_bbox)
    return np.array(shrinked_bboxes)

class IC15Dataset(data.Dataset):
    ic15_root_dir = '/home/data1/IC19/ICDAR2015/Challenge4/'
    ic15_train_data_dir = ic15_root_dir + 'ch4_training_images/'
    ic15_train_anno_dir = ic15_root_dir + 'ch4_training_localization_transcription_gt/'

    def __init__(self, num_kernel=7):
        self.img_size = None
        train_data_dir = self.ic15_train_data_dir
        train_anno_dir = self.ic15_train_anno_dir
        self.img_paths = list()
        self.anno_paths = list()
        self.num_kernel = num_kernel
        self.min_scale = 0.4
        # TODO: implement transform
        self.is_transform = False

        img_names = os.listdir(train_data_dir)
        for idx, img_name in enumerate(img_names):
            img_path = train_data_dir + img_name
            self.img_paths.append(img_path)
            anno_path = train_anno_dir + 'gt_' + img_name.split('.')[0] + '.txt'
            self.anno_paths.append(anno_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        anno_path = self.anno_paths[index]
        img = read_image(img_path)
        bboxes, tags = read_anno(img, anno_path)

        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),(bboxes.shape[0], int(bboxes.shape[1] / 2), 2)).astype(int)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if tags[i] is False:
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
        # gt_kernels中的kernel从大到小进行排序
        gt_kernels = []
        for i in range(1, self.num_kernel):
            rate = 1.0 - (1.0 - self.min_scale) / (self.num_kernel - 1) * i
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            #ori_gt = gt_kernel
            kernel_bboxes = shrink(bboxes=bboxes, rate=rate)
            for j in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[j]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        # TODO: implement transform
        if self.is_transform:
            pass


        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()

        return img, gt_text, gt_kernels, training_mask

class IC15TestDataset(data.Dataset):
    ic15_root_dir = '/home/data1/IC19/ICDAR2015/Challenge4/'
    ic15_test_data_dir = ic15_root_dir + 'ch4_test_images/'
    ic15_test_anno_dir = ic15_root_dir + 'ch4_test_localization_transcription_gt/'

    def __init__(self):
        self.img_paths = []
        self.anno_paths = []
        test_data_dir = self.ic15_test_data_dir
        test_anno_dir = self.ic15_test_anno_dir
        img_names = os.listdir(test_data_dir)
        for img_name in img_names:
            self.img_paths.append(test_data_dir + img_name)
        # TODO: read from annotation?

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = read_image(img_path)
        original_img = img[:, :, [2, 1, 0]]
        # TODO:
        # according the PSENet paper, images in IC15 should be
        # rescaled to longer side 2240 during inference

        img = scale_shortside(img, short_size=736)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        # why [2, 1, 0]?
        return img, original_img

class IC15Trainset_PAN(data.Dataset):
    ic15_root_dir = '/home/data1/IC19/ICDAR2015/Challenge4/'
    ic15_train_data_dir = ic15_root_dir + 'ch4_training_images/'
    ic15_train_anno_dir = ic15_root_dir + 'ch4_training_localization_transcription_gt/'

    def __init__(self, kernel_scale=0.5, with_coord=False):
        self.kernel_scale = 0.5
        self.with_coord = with_coord
        train_data_dir = self.ic15_train_data_dir
        train_anno_dir = self.ic15_train_anno_dir
        self.img_paths = []
        self.anno_paths = []
        img_names = os.listdir(train_data_dir)
        for idx, img_name in enumerate(img_names):
            img_path = train_data_dir + img_name
            self.img_paths.append(img_path)
            anno_path = train_anno_dir + 'gt_' + img_name.split('.')[0] + '.txt'
            self.anno_paths.append(anno_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index, debug=False):
        start_time = time.time()
        img_path = self.img_paths[index]
        anno_path = self.anno_paths[index]
        img = read_image(img_path)
        # cv2.imwrite('demo.jpg', img)
        bboxes, tags = read_anno(img, anno_path)
        img, scale_rate = random_scale(img, 640)
        training_mask = np.ones(shape=img.shape[0:2], dtype='uint8')

        # generating gt_text
        gt_text = np.zeros(shape=img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                                (bboxes.shape[0], int(bboxes.shape[1] / 2), 2))
            bboxes = bboxes.astype(int)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if debug:
                    cv2.drawContours(img, [bboxes[i]], -1, (0, 0, 255), -1)
                if tags[i] is False:
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        # generating gt_kernel
        gt_kernel = np.zeros(shape=img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(bboxes=bboxes, rate=self.kernel_scale)
        for i in range(len(kernel_bboxes)):
            cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, i + 1, -1)
            if debug:
                cv2.drawContours(img, [kernel_bboxes[i]], -1, (255, 0, 0), -1)
        img_name = img_path.split('/')[-1].split('.')[0]
        demo_name = img_name + '_demo.jpg'
        crop_name = img_name + '_crop.jpg'
        demo_path = '/home/data1/zhm/vis/' + demo_name
        crop_path = '/home/data1/zhm/vis/' + crop_name

        # cv2.imwrite(demo_path, img)
        # cv2.imwrite('demo.jpg', img)

        img_infos = [img, gt_text, gt_kernel, training_mask]
        img_infos = random_horizontal_flip(img_infos)
        img_infos = random_rotate(img_infos)
        img_infos = random_crop(img_infos, 640)
        img, gt_text, gt_kernel, training_mask = img_infos[0], img_infos[1], img_infos[2], img_infos[3]

        # cv2.imwrite(crop_path, img)
        # cv2.imwrite('crop.jpg', img)

        gt_text_labeled = gt_text.copy()
        gt_kernel_labeled = gt_kernel.copy()

        gt_text[gt_text > 0] = 1
        gt_kernel[gt_kernel > 0] = 1

        img = Image.fromarray(img)
        img = img.convert('RGB')

        img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        img = transforms.ToTensor()(img)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        text = color.label2rgb(gt_text_labeled)
        kernel = color.label2rgb(gt_kernel_labeled)
        mask = color.label2rgb(training_mask)

        # cv2.imwrite('text.jpg', text * 255)
        # cv2.imwrite('kernel.jpg', kernel * 255)
        # cv2.imwrite('mask.jpg', mask * 255)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernel = torch.from_numpy(gt_kernel).float()

        gt_text_labeled = torch.from_numpy(gt_text_labeled).float()
        gt_kernel_labeled = torch.from_numpy(gt_kernel_labeled).float()

        training_mask = torch.from_numpy(training_mask).float()

        # print('time per img', time.time() - start_time)

        # print(torch.sum(gt_text), torch.sum(gt_kernel))
        # print(torch.sum((gt_text > 0.5) & (training_mask > 0.5)), torch.sum((gt_kernel > 0.5) & (training_mask > 0.5)))
        # print()


        return img, gt_text, gt_kernel, gt_text_labeled, gt_kernel_labeled, training_mask



if __name__ == '__main__':
    # print('testing IC15Dataset')
    # print([1 * 1.0, 2 * 1.0] * 4)
    # print('\ufeff')
    # dataset = IC15Trainset_PAN()
    # trainloader = data.DataLoader(dataset=dataset,
    #                              batch_size=16,
    #                              shuffle=True,
    #                              num_workers=1,
    #                              drop_last=True,
    #                              pin_memory=True)
    # # dataset.__getitem__(index=1, debug=True)
    # data_time = AverageMeter()
    # current_time = time.time()
    # for batch_idx, (imgs, gt_texts, gt_kernels, gt_texts_labeled, gt_kernels_labeled, training_masks) in enumerate(trainloader):
    #     data_time.update(time.time() - current_time)
    #     # print(time.time() - current_time)
    #     if (batch_idx + 1) % 20 == 0:
    #         print(data_time.avg)
    #     current_time = time.time()
    # print('Done')
    dataset = IC15TestDataset()
    # img, original_img = dataset.__getitem__(0)
    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False)
    for idx, (img, original_img) in enumerate(dataloader):
        print(img)
        print(original_img)

    # print(img)
    # print(original_img)

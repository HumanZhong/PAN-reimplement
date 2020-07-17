import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import os

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
            info = line.split(sep=',')

            # 矩形文本框左上角坐标
            x1 = np.int(info[0])
            y1 = np.int(info[1])

            # 弯曲文本框各点对于左上角坐标的偏移值
            bbox = [np.int(info[i]) for i in range(4, 32)]
            bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
            bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)
            bboxes.append(bbox)
            tags.append(True)
    return np.array(bboxes), tags

def read_anno_bound(img, anno_path):
    h, w = img.shape[0:2]
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        top_bounds = []
        bot_bounds = []
        bboxes = []
        tags =[]
        for line in lines:
            info = line.split(',')
            x1 = np.int(info[0])
            y1 = np.int(info[1])

            # top_bound = [np.int(info[i]) for i in range(4, 18)]
            # bot_bound = [np.int(info[i]) for i in range(18, 32)]
            # top_bound = np.asarray(top_bound) + ([x1 * 1.0, y1 * 1.0] * 7)

            bbox = [np.int(info[i]) for i in range(4, 32)]
            bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
            bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)
            top_bound = bbox[:14]
            bot_bound = bbox[14:]
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

def shrink(bboxes, rate):
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pyco = pyclipper.PyclipperOffset()
        pyco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)((area * (1 - rate * rate)) / (peri + 0.001) + 0.5), 20)
        shrinked_bbox = pyco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        shrinked_bboxes.append(shrinked_bbox)
    return np.array(shrinked_bboxes)

def uniform_scale(img, target_size):
    if isinstance(target_size, tuple):
        (target_h, target_w) = target_size
    else:
        target_h = target_size
        target_w = target_size
    h, w = img.shape[0:2]
    scale_x = target_w * 1.0 / w
    scale_y = target_h * 1.0 / h
    img = cv2.resize(img, dsize=None, fx=scale_x, fy=scale_y)
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    random_scale_rate = np.array([0.5, 1.0, 2.0])
    scale = np.random.choice(random_scale_rate)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img, scale

def random_crop(img_infos, img_size):
    h, w = img_infos[0].shape[0:2]
    th, tw = img_size, img_size
    if th == h and tw == w:
        return img_infos
    if random.random() > 3.0 / 8.0 and np.max(img_infos[1]) > 0:
        tl = np.min(np.where(img_infos[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(img_infos[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    for idx in range(len(img_infos)):
        if len(img_infos[idx].shape) == 3:
            img_infos[idx] = img_infos[idx][i:i + th, j:j + tw, :]
        else:
            img_infos[idx] = img_infos[idx][i:i + th, j:j + tw]
    return img_infos

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

class CTW1500TrainDataset(data.Dataset):
    ctw_root_dir = '/home/data1/IC19/CTW1500/'
    ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
    ctw_train_anno_dir = ctw_root_dir + 'train/text_label_curve/'

    def __init__(self, num_kernel=7):
        # attention: num_kernel是包括最外层gt_text在内的总个数
        self.num_kernel = 7
        self.min_scale = 0.4

        self.img_paths = []
        self.anno_paths = []
        train_data_dir = self.ctw_train_data_dir
        train_anno_dir = self.ctw_train_anno_dir
        img_names = os.listdir(train_data_dir)
        for img_name in img_names:
            img_path = train_data_dir + img_name
            self.img_paths.append(img_path)
            anno_name = img_name.split('.')[0] + '.txt'
            anno_path = train_anno_dir + anno_name
            self.anno_paths.append(anno_path)
        assert len(self.img_paths) == len(self.anno_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        anno_path = self.anno_paths[index]
        img = read_image(img_path=img_path)
        bboxes, tags = read_anno(img=img, anno_path=anno_path)

        img, scale_rate = random_scale(img, 640)

        gt_text = np.zeros(shape=img.shape[0:2], dtype='uint8')
        training_mask = np.ones(shape=img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14), (bboxes.shape[0], int(bboxes.shape[1] / 2), 2))
            bboxes = bboxes.astype(int)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)

        gt_kernels = []
        for i in range(1, self.num_kernel):
            rate = 1.0 - (1.0 - self.min_scale) / (self.num_kernel - 1) * i
            gt_kernel = np.zeros(shape=img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes=bboxes, rate=rate)
            for j in range(len(kernel_bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[j]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        # TODO: implement transform (randomflip, randomcrop .etc)
        img_infos = [img, gt_text, training_mask]
        img_infos.extend(gt_kernels)
        img_infos = random_crop(img_infos, 640)
        img, gt_text, training_mask, gt_kernels = img_infos[0], img_infos[1], img_infos[2], img_infos[3:]


        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()

        return img, gt_text, gt_kernels, training_mask

def scale(img, long_size=1280):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    resize_h = h * scale
    resize_w = w * scale
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    scale_x = resize_w / w
    scale_y = resize_h / h
    img = cv2.resize(img, dsize=None, fx=scale_x, fy=scale_y)
    return img

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


def scale_demo(img, size=640):
    h, w = img.shape[0:2]
    scale_x = 640 / w
    scale_y = 640 / h
    img = cv2.resize(img, dsize=None, fx=scale_x, fy=scale_y)
    return img



class CTW1500Testset(data.Dataset):
    ctw_root_dir = '/home/data1/IC19/CTW1500/'
    ctw_test_data_dir = ctw_root_dir + 'test_ctw/text_image/'
    ctw_test_anno_dir = ctw_root_dir + 'test_ctw/text_label_curve/'

    def __init__(self):
        self.img_paths = []
        self.anno_paths = []
        test_data_dir = self.ctw_test_data_dir
        test_anno_dir = self.ctw_test_anno_dir
        img_names = os.listdir(test_data_dir)
        for img_name in img_names:
            img_path = test_data_dir + img_name
            self.img_paths.append(img_path)
            anno_name = img_name.split('.')[0] + '.txt'
            anno_path = test_anno_dir + anno_name
            self.anno_paths.append(anno_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        anno_path = self.anno_paths[index]

        img = read_image(img_path)
        original_img = img[:, :, [2, 1, 0]]

        img = scale(img)

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, original_img


def generate_bounds(bounds, min_lengths, img_shape):
    assert bounds.shape[1] == 7
    result = np.zeros(shape=(bounds.shape[0], 14, 2))
    for idx in range(bounds.shape[0]):
        for i in range(7):
            current_x = bounds[idx, i, 0]
            current_y = bounds[idx, i, 1]
            if i == 0:
                x1 = bounds[idx, i, 0]
                y1 = bounds[idx, i, 1]
                x2 = bounds[idx, i + 1, 0]
                y2 = bounds[idx, i + 1, 1]
            elif i == 6:
                x1 = bounds[idx, i - 1, 0]
                y1 = bounds[idx, i - 1, 1]
                x2 = bounds[idx, i, 0]
                y2 = bounds[idx, i, 1]
            else:
                x1 = bounds[idx, i - 1, 0]
                y1 = bounds[idx, i - 1, 1]
                x2 = bounds[idx, i + 1, 0]
                y2 = bounds[idx, i + 1, 1]
            if y1 == y2:
                tmp_x = 0
                tmp_y = 1
            else:
                tmp_x = 1
                tmp_y = (x1 - x2) / (y2 - y1)

            vector_x = tmp_x / np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
            vector_y = tmp_y / np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)

            #length = 4

            # TODO: to avoid the x-shape fault, we can simply put the point whose y-coord is smaller than the other point as the upper point

            target_x1 = current_x + min_lengths[idx] * vector_x
            target_y1 = current_y + min_lengths[idx] * vector_y
            target_x2 = current_x - min_lengths[idx] * vector_x
            target_y2 = current_y - min_lengths[idx] * vector_y

            target_x1 = int(target_x1 + 0.5)
            target_y1 = int(target_y1 + 0.5)
            target_x2 = int(target_x2 + 0.5)
            target_y2 = int(target_y2 + 0.5)

            (h, w) = img_shape
            target_x1 = np.clip(target_x1, 0, w)
            target_y1 = np.clip(target_y1, 0, h)
            target_x2 = np.clip(target_x2, 0, w)
            target_y2 = np.clip(target_y2, 0, h)

            if target_y1 < target_y2:
                result[idx, i, 0] = target_x1
                result[idx, i, 1] = target_y1
                result[idx, 13 - i, 0] = target_x2
                result[idx, 13 - i, 1] = target_y2
            else:
                result[idx, i, 0] = target_x2
                result[idx, i, 1] = target_y2
                result[idx, 13 - i, 0] = target_x1
                result[idx, 13 - i, 1] = target_y1
    return result.astype(int)

def generate_bounds_shrinked(bounds, min_lengths, img_shape, top=True):
    assert bounds.shape[1] == 7
    result = np.zeros(shape=(bounds.shape[0], 14, 2))
    for idx in range(bounds.shape[0]):
        for i in range(7):
            current_x = bounds[idx, i, 0]
            current_y = bounds[idx, i, 1]
            if i == 0:
                x1 = bounds[idx, i, 0]
                y1 = bounds[idx, i, 1]
                x2 = bounds[idx, i + 1, 0]
                y2 = bounds[idx, i + 1, 1]
            elif i == 6:
                x1 = bounds[idx, i - 1, 0]
                y1 = bounds[idx, i - 1, 1]
                x2 = bounds[idx, i, 0]
                y2 = bounds[idx, i, 1]
            else:
                x1 = bounds[idx, i - 1, 0]
                y1 = bounds[idx, i - 1, 1]
                x2 = bounds[idx, i + 1, 0]
                y2 = bounds[idx, i + 1, 1]
            if y1 == y2:
                tmp_x = 0
                tmp_y = 1
            else:
                tmp_x = 1
                tmp_y = (x1 - x2) / (y2 - y1)

            vector_x = tmp_x / np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
            vector_y = tmp_y / np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)

            # length = 4

            # TODO: to avoid the x-shape fault, we can simply put the point whose y-coord is smaller than the other point as the upper point

            target_x1 = current_x + min_lengths[idx] * vector_x
            target_y1 = current_y + min_lengths[idx] * vector_y
            target_x2 = current_x - min_lengths[idx] * vector_x
            target_y2 = current_y - min_lengths[idx] * vector_y

            target_x1 = int(target_x1 + 0.5)
            target_y1 = int(target_y1 + 0.5)
            target_x2 = int(target_x2 + 0.5)
            target_y2 = int(target_y2 + 0.5)

            (h, w) = img_shape
            target_x1 = np.clip(target_x1, 0, w)
            target_y1 = np.clip(target_y1, 0, h)
            target_x2 = np.clip(target_x2, 0, w)
            target_y2 = np.clip(target_y2, 0, h)

            if top:
                if target_y1 > current_y:
                    result[idx, i, 0] = current_x
                    result[idx, i, 1] = current_y
                    result[idx, 13 - i, 0] = target_x1
                    result[idx, 13 - i, 1] = target_y1
                else:
                    result[idx, i, 0] = current_x
                    result[idx, i, 1] = current_y
                    result[idx, 13 - i, 0] = target_x2
                    result[idx, 13 - i, 1] = target_y2
            else:
                if target_y1 < current_y:
                    result[idx, i, 0] = target_x1
                    result[idx, i, 1] = target_y1
                    result[idx, 13 - i, 0] = current_x
                    result[idx, 13 - i, 1] = current_y
                else:
                    result[idx, i, 0] = target_x2
                    result[idx, i, 1] = target_y2
                    result[idx, 13 - i, 0] = current_x
                    result[idx, 13 - i, 1] = current_y
    return result.astype(int)

def generate_corners(bboxes, corner_radiuses):
    top_lefts = []
    top_rights = []
    bot_rights = []
    bot_lefts = []
    for i in range(len(bboxes)):
        top_left = []
        top_right = []
        bot_right = []
        bot_left = []
        corner_radius = corner_radiuses[i]
        top_left_corner = bboxes[i, 0]
        top_right_corner = bboxes[i, 6]
        bot_left_corner = bboxes[i, -1]
        bot_right_corner = bboxes[i, 7]
        offset_x = [-corner_radius, corner_radius, corner_radius, -corner_radius]
        offset_y = [-corner_radius, -corner_radius, corner_radius, corner_radius]
        for j in range(4):
            top_left.append([int(top_left_corner[0] + offset_x[j] + 0.5), int(top_left_corner[1] + offset_y[j] + 0.5)])
            top_right.append([int(top_right_corner[0] + offset_x[j] + 0.5), int(top_right_corner[1] + offset_y[j] + 0.5)])
            bot_right.append([int(bot_right_corner[0] + offset_x[j] + 0.5), int(bot_right_corner[1] + offset_y[j] + 0.5)])
            bot_left.append([int(bot_left_corner[0] + offset_x[j] + 0.5), int(bot_left_corner[1] + offset_y[j] + 0.5)])
        top_lefts.append(top_left)
        top_rights.append(top_right)
        bot_rights.append(bot_right)
        bot_lefts.append(bot_left)
    top_lefts = np.array(top_lefts).astype(int)
    top_rights = np.array(top_rights).astype(int)
    bot_rights = np.array(bot_rights).astype(int)
    bot_lefts = np.array(bot_lefts).astype(int)
    return top_lefts, top_rights, bot_rights, bot_lefts

def generate_minlengths(bboxes):
    minlengths = []
    for i in range(len(bboxes)):
        minlength = dist(bboxes[i, 0], bboxes[i, 1])
        for j in range(1, len(bboxes[i])):
            minlength = min(minlength, dist(bboxes[i, j], bboxes[i, j - 1]))
        minlengths.append(minlength)
    minlengths = np.array(minlengths) * 0.1
    return minlengths

def get_radiuses(bboxes, scale=0.15):
    radiuses = []
    for i in range(len(bboxes)):
        radiuses.append((dist(bboxes[i, 0], bboxes[i, -1]) + dist(bboxes[i, 6], bboxes[i, 7])) / 2)
    return np.array(radiuses) * scale

def random_crop_bound(img_infos, img_size):
    '''
    random_crop function for boundary version
    :param img_infos: [img, gt_text, gt_kernel, gt_top, gt_bot, trainingmask]
    :param img_size: target_size
    :return: cropped img_infos
    '''
    h, w = img_infos[0].shape[0:2]
    th, tw = img_size, img_size
    if th == h and tw == w:
        return img_infos
    if random.random() > 3.0 / 8.0 and np.max(img_infos[1]) > 0:
        tl = np.min(np.where(img_infos[1] > 0), axis=1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(img_infos[1] > 0), axis=1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    for idx in range(len(img_infos)):
        if len(img_infos[idx].shape) == 3:
            img_infos[idx] = img_infos[idx][i:i + th, j:j + tw, :]
        else:
            img_infos[idx] = img_infos[idx][i:i + th, j:j + tw]
    return img_infos




class CTW1500Trainset_Bound(data.Dataset):
    ctw_root_dir = '/home/data1/IC19/CTW1500/'
    ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
    ctw_train_anno_dir = ctw_root_dir + 'train/text_label_curve/'
    ctw_train_anno_dir_e2e = '/home/data1/zhm/dataset/CTW1500/ctw1500_e2e_train/'

    def __init__(self, min_scale=0.6, with_coord=True):
        self.min_scale = min_scale
        self.with_coord = with_coord
        self.color_jitter = True

        self.img_paths = []
        self.anno_paths = []
        train_data_dir = self.ctw_train_data_dir
        train_anno_dir = self.ctw_train_anno_dir
        img_names = os.listdir(train_data_dir)
        for img_name in img_names:
            img_path = train_data_dir + img_name
            self.img_paths.append(img_path)
            anno_path = train_anno_dir + img_name.split('.')[0] + '.txt'
            self.anno_paths.append(anno_path)
        assert len(self.img_paths) == len(self.anno_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        anno_path = self.anno_paths[index]
        img = read_image(img_path)
        bboxes, top_bounds, bot_bounds, tags = read_anno_bound(img, anno_path)

        img, scale_rate = random_scale(img, 640)

        training_mask = np.ones(shape=img.shape[0:2], dtype='uint8')

        # TODO: implement an option for putting more weight on the 4 corners of the text instance
        training_weight = np.ones(shape=img.shape[0:2], dtype='uint8')




        # attention: generating full text instance region
        gt_text = np.zeros(shape=img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14),
                                (bboxes.shape[0], int(bboxes.shape[1] / 2), 2))
            bboxes = bboxes.astype(int)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)

        # min_lengths = generate_minlengths(bboxes)
        radiuses = get_radiuses(bboxes, scale=0.2)

        # attention: generating text kernel
        gt_kernel = np.zeros(shape=img.shape[0:2], dtype='uint8')
        rate = self.min_scale
        kernel_bboxes = shrink(bboxes=bboxes, rate=rate)
        for i in range(len(kernel_bboxes)):
            cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)

        # attention: generating top boundary of text instance
        gt_top = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # length = int(6 * scale_rate) # TODO: length should be an adaptive variable
        top_bounds = np.reshape(top_bounds * ([img.shape[1], img.shape[0]] * 7), (top_bounds.shape[0], 7, 2))
        expanded_top_bounds = generate_bounds_shrinked(top_bounds, radiuses, img_shape=img.shape[0:2], top=True)
        for i in range(len(expanded_top_bounds)):
            cv2.drawContours(gt_top, [expanded_top_bounds[i]], -1, 1, -1)
        # cv2.imwrite('demo.jpg', img)

        # attention: generating bottom boundary of text instance
        gt_bot = np.zeros(shape=img.shape[0:2], dtype='uint8')
        length = int(6 * scale_rate)
        bot_bounds = np.reshape(bot_bounds * ([img.shape[1], img.shape[0]] * 7), (bot_bounds.shape[0], 7, 2))
        expanded_bot_bounds = generate_bounds_shrinked(bot_bounds, radiuses, img_shape=img.shape[0:2], top=False)
        for i in range(len(expanded_bot_bounds)):
            cv2.drawContours(gt_bot, [expanded_bot_bounds[i]], -1, 1, -1)
        # cv2.imwrite('demo.jpg', img)

        # TODO: generating the 4 corners of text instance
        # gt_top_left = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # gt_top_right = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # gt_bot_left = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # gt_bot_right = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # top_lefts, top_rights, bot_lefts, bot_rights = generate_corners(bboxes, radiuses)
        # assert len(top_lefts) == len(top_rights)
        # for i in range(len(top_lefts)):
        #     cv2.drawContours(gt_top_left, [top_lefts[i]], -1, 1, -1)
        #     cv2.drawContours(gt_top_right, [top_rights[i]], -1, 1, -1)
        #     cv2.drawContours(gt_bot_left, [bot_lefts[i]], -1, 1, -1)
        #     cv2.drawContours(gt_bot_right, [bot_rights[i]], -1, 1, -1)

        # TODO: implement the coord map with the same size of the original img
        if self.with_coord:
            h, w = img.shape[0:2]
            i_channel = np.zeros(shape=(h, w)).astype(float)
            j_channel = np.zeros(shape=(h, w)).astype(float)
            for i in range(h):
                i_channel[i, :] = -1 + i / h * 2
            for j in range(w):
                j_channel[:, j] = -1 + j / w * 2
            img_infos = [img, gt_text, gt_kernel, gt_top, gt_bot, training_mask, i_channel, j_channel]
            img_infos = random_horizontal_flip(img_infos)
            img_infos = random_rotate(img_infos)
            img_infos = random_crop_bound(img_infos, 640)
            img, gt_text, gt_kernel, gt_top, gt_bot, training_mask, i_channel, j_channel = img_infos[0], img_infos[1], img_infos[2], img_infos[3], img_infos[4], img_infos[5], img_infos[6], img_infos[7]

        # attention: random_crop for corner version
        # img_infos = [img, gt_text, gt_kernel, gt_top_left, gt_top_right, gt_bot_right, gt_bot_left, training_mask]
        # img_infos = random_crop(img_infos, 640)
        # img, gt_text, gt_kernel, gt_top_left, gt_top_right, gt_bot_right, gt_bot_left, training_mask = img_infos[0], img_infos[1], img_infos[2], img_infos[3], img_infos[4], img_infos[5], img_infos[6], img_infos[7]

        # attention: random_crop for boundary version
        else:
            img_infos = [img, gt_text, gt_kernel, gt_top, gt_bot, training_mask]
            img_infos = random_horizontal_flip(img_infos)
            img_infos = random_rotate(img_infos)
            img_infos = random_crop_bound(img_infos, 640)
            img, gt_text, gt_kernel, gt_top, gt_bot, training_mask = img_infos[0], img_infos[1], img_infos[2], img_infos[3], img_infos[4], img_infos[5]

        gt_text[gt_text > 0] = 1
        gt_kernel = np.array(gt_kernel)

        if self.color_jitter:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).float()
        gt_kernel = torch.from_numpy(gt_kernel).float()

        gt_top = torch.from_numpy(gt_top).float()
        gt_bot = torch.from_numpy(gt_bot).float()

        # gt_top_left = torch.from_numpy(gt_top_left).float()
        # gt_top_right = torch.from_numpy(gt_top_right).float()
        # gt_bot_left = torch.from_numpy(gt_bot_left).float()
        # gt_bot_right = torch.from_numpy(gt_bot_right).float()

        training_mask = torch.from_numpy(training_mask).float()

        if self.with_coord:
            i_channel = cv2.resize(i_channel, dsize=None, fx=0.25, fy=0.25)
            j_channel = cv2.resize(j_channel, dsize=None, fx=0.25, fy=0.25)

            i_channel = np.expand_dims(i_channel, axis=0)
            j_channel = np.expand_dims(j_channel, axis=0)

            i_channel = torch.from_numpy(i_channel).float()
            j_channel = torch.from_numpy(j_channel).float()

            return img, gt_text ,gt_kernel, gt_top, gt_bot, training_mask, i_channel, j_channel
        return img, gt_text, gt_kernel, gt_top, gt_bot, training_mask

class CTW1500Testset_Bound(data.Dataset):
    ctw_root_dir = '/home/data1/IC19/CTW1500/'
    ctw_test_data_dir = ctw_root_dir + 'test_ctw/text_image/'
    ctw_test_anno_dir = ctw_root_dir + 'test_ctw/text_label_curve/'
    ctw_test_anno_dir_e2e = '/home/data1/zhm/dataset/CTW1500/ctw1500_e2e_test/'
    def __init__(self, with_coord=True):
        self.with_coord = with_coord

        self.img_paths = []
        self.anno_paths = []
        test_data_dir = self.ctw_test_data_dir
        test_anno_dir = self.ctw_test_anno_dir
        img_names = os.listdir(test_data_dir)
        for img_name in img_names:
            img_path = test_data_dir + img_name
            self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        # anno_path = self.anno_paths[index]

        img = read_image(img_path)
        original_img = img[:, :, [2, 1, 0]]

        # img = scale(img)
        img = scale_shortside(img, short_size=640)

        if self.with_coord:
            h, w = img.shape[0:2]
            i_channel = np.zeros(shape=(h, w)).astype(float)
            j_channel = np.zeros(shape=(h, w)).astype(float)
            for i in range(h):
                i_channel[i, :] = -1 + i / h * 2
            for j in range(w):
                j_channel[:, j] = -1 + j / w * 2

            i_channel = cv2.resize(i_channel, dsize=None, fx=0.25, fy=0.25)
            j_channel = cv2.resize(j_channel, dsize=None, fx=0.25, fy=0.25)

            i_channel = np.expand_dims(i_channel, axis=0)
            j_channel = np.expand_dims(j_channel, axis=0)

            i_channel = torch.from_numpy(i_channel).float()
            j_channel = torch.from_numpy(j_channel).float()

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.with_coord:
            return img, original_img, i_channel, j_channel
        return img, original_img




class CTW1500Trainset_Corner(data.Dataset):
    ctw_root_dir = '/home/data1/IC19/CTW1500/'
    ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
    ctw_train_anno_dir = ctw_root_dir + 'train/text_label_curve/'
    ctw_train_anno_dir_e2e = '/home/data1/zhm/dataset/CTW1500/ctw1500_e2e_train/'

    def __init__(self, min_scale=0.7):
        self.min_scale = min_scale

        self.img_paths = []
        self.anno_paths = []
        train_data_dir = self.ctw_train_data_dir
        train_anno_dir = self.ctw_train_anno_dir
        img_names = os.listdir(train_data_dir)
        for img_name in img_names:
            img_path = train_data_dir + img_name
            self.img_paths.append(img_path)
            anno_path = train_anno_dir + img_name.split('.')[0] + '.txt'
            self.anno_paths.append(anno_path)
        assert len(self.img_paths) == len(self.anno_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        anno_path = self.anno_paths[index]
        img = read_image(img_path)
        bboxes, top_bounds, bot_bounds, tags = read_anno_bound(img, anno_path)

        img, scale_rate = random_scale(img, 640)

        training_mask = np.ones(shape=img.shape[0:2], dtype='uint8')

        # TODO: implement an option for putting more weight on the 4 corners of the text instance
        training_weight = np.ones(shape=img.shape[0:2], dtype='uint8')

        # attention: generating full text instance region
        gt_text = np.zeros(shape=img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14),
                                (bboxes.shape[0], int(bboxes.shape[1] / 2), 2))
            bboxes = bboxes.astype(int)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)

        min_lengths = generate_minlengths(bboxes)
        corner_radiuses = get_radiuses(bboxes, scale=0.15)

        # attention: generating text kernel
        gt_kernel = np.zeros(shape=img.shape[0:2], dtype='uint8')
        rate = self.min_scale
        kernel_bboxes = shrink(bboxes=bboxes, rate=rate)
        for i in range(len(kernel_bboxes)):
            cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)

        # attention: generating top boundary of text instance
        # gt_top = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # # length = int(6 * scale_rate) # TODO: length should be an adaptive variable
        # top_bounds = np.reshape(top_bounds * ([img.shape[1], img.shape[0]] * 7), (top_bounds.shape[0], 7, 2))
        # expanded_top_bounds = generate_bounds(top_bounds, min_lengths)
        # for i in range(len(expanded_top_bounds)):
        #     cv2.drawContours(gt_top, [expanded_top_bounds[i]], -1, 1, -1)
        # cv2.imwrite('demo.jpg', img)

        # attention: generating bottom boundary of text instance
        # gt_bot = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # length = int(6 * scale_rate)
        # bot_bounds = np.reshape(bot_bounds * ([img.shape[1], img.shape[0]] * 7), (bot_bounds.shape[0], 7, 2))
        # expanded_bot_bounds = generate_bounds(bot_bounds, min_lengths)
        # for i in range(len(expanded_bot_bounds)):
        #     cv2.drawContours(gt_bot, [expanded_bot_bounds[i]], -1, 1, -1)

        # TODO: generating the 4 corners of text instance
        gt_top_left = np.zeros(shape=img.shape[0:2], dtype='uint8')
        gt_top_right = np.zeros(shape=img.shape[0:2], dtype='uint8')
        gt_bot_left = np.zeros(shape=img.shape[0:2], dtype='uint8')
        gt_bot_right = np.zeros(shape=img.shape[0:2], dtype='uint8')
        top_lefts, top_rights, bot_lefts, bot_rights = generate_corners(bboxes, corner_radiuses)
        assert len(top_lefts) == len(top_rights)
        for i in range(len(top_lefts)):
            cv2.drawContours(gt_top_left, [top_lefts[i]], -1, 1, -1)
            cv2.drawContours(gt_top_right, [top_rights[i]], -1, 1, -1)
            cv2.drawContours(gt_bot_left, [bot_lefts[i]], -1, 1, -1)
            cv2.drawContours(gt_bot_right, [bot_rights[i]], -1, 1, -1)



        # attention: random_crop for corner version
        img_infos = [img, gt_text, gt_kernel, gt_top_left, gt_top_right, gt_bot_right, gt_bot_left, training_mask]
        img_infos = random_crop(img_infos, 640)
        img, gt_text, gt_kernel, gt_top_left, gt_top_right, gt_bot_right, gt_bot_left, training_mask = img_infos[0], img_infos[1], img_infos[2], img_infos[3], img_infos[4], img_infos[5], img_infos[6], img_infos[7]

        # attention: random_crop for boundary version
        # img_infos = [img, gt_text, gt_kernel, gt_top, gt_bot, training_mask]
        # img_infos = random_crop_bound(img_infos, 640)
        # img, gt_text, gt_kernel, gt_top, gt_bot, training_mask = img_infos[0], img_infos[1], img_infos[2], img_infos[3], img_infos[4], img_infos[5]

        gt_text[gt_text > 0] = 1
        gt_kernel = np.array(gt_kernel)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).float()
        gt_kernel = torch.from_numpy(gt_kernel).float()

        # gt_top = torch.from_numpy(gt_top).float()
        # gt_bot = torch.from_numpy(gt_bot).float()

        gt_top_left = torch.from_numpy(gt_top_left).float()
        gt_top_right = torch.from_numpy(gt_top_right).float()
        gt_bot_left = torch.from_numpy(gt_bot_left).float()
        gt_bot_right = torch.from_numpy(gt_bot_right).float()

        training_mask = torch.from_numpy(training_mask).float()

        return img, gt_text, gt_kernel, gt_top_left, gt_top_right, gt_bot_right, gt_bot_left, training_mask




# def test_dataset_bound():
#     print('Testing CTW1500Trainset_Bound')
#     dataset = CTW1500Trainset_Bound()
#     dataset.__getitem__(0)

if __name__ == '__main__':
    # print('Testing CTW1500Dataset')
    # dataset = CTW1500TrainDataset()
    # dataset.__getitem__(0)
    dataset = CTW1500Trainset_Bound()
    dataset.__getitem__(0)
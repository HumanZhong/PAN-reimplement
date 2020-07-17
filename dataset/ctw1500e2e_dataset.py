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

vocab =  "<,.+:-?$ 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>"

def read_anno_bound_e2e(img, anno_path):
    h, w = img.shape[0:2]
    with open(anno_path, 'r') as f:
        lines = f.readlines()
        top_bounds = []
        bot_bounds = []
        bboxes = []
        tags = []
        n_bbox = int(lines[0])
        for i in range(1, len(lines)):
            line = lines[i]
            info = line.split(',')
            target_str = info[-1]




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

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    random_scale_rate = np.array([0.5, 1.0, 2.0])
    scale = np.random.choice(random_scale_rate)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img, scale

def get_radiuses(bboxes, scale=0.15):
    radiuses = []
    for i in range(len(bboxes)):
        radiuses.append((dist(bboxes[i, 0], bboxes[i, -1]) + dist(bboxes[i, 6], bboxes[i, 7])) / 2)
    return np.array(radiuses) * scale

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

# ------------------------------data augmentation--------------------

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



# TODO:this class is designed for e2e problem, but the e2e part hasn't been implemented
class CTW1500Trainset_BoundE2E(data.Dataset):
    ctw_root_dir = '/home/data1/IC19/CTW1500/'
    ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
    ctw_train_anno_dir = ctw_root_dir + 'train/text_label_curve/'
    ctw_train_anno_dir_e2e = '/home/data1/zhm/dataset/CTW1500/ctw1500_e2e_train/'
    def __init__(self, kernel_scale=0.7, with_coord=False):
        self.kernel_scale = kernel_scale
        self.with_coord = with_coord

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

        # attention: generating full text instance region
        gt_text = np.zeros(shape=img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14),
                                (bboxes.shape[0], int(bboxes.shape[1] / 2), 2))
            bboxes = bboxes.astype(int)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)

        radiuses = get_radiuses(bboxes, scale=0.25)

        # attention: generating text kernel
        gt_kernel = np.zeros(shape=img.shape[0:2], dtype='uint8')
        rate = self.kernel_scale
        kernel_bboxes = shrink(bboxes=bboxes, rate=rate)
        for i in range(len(kernel_bboxes)):
            cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, i + 1, -1)

        # attention: generating top boundary of text instance
        gt_top = np.zeros(shape=img.shape[0:2], dtype='uint8')
        # length = int(6 * scale_rate) # TODO: length should be an adaptive variable
        top_bounds = np.reshape(top_bounds * ([img.shape[1], img.shape[0]] * 7), (top_bounds.shape[0], 7, 2))
        expanded_top_bounds = generate_bounds_shrinked(top_bounds, radiuses, img_shape=img.shape[0:2], top=True)
        for i in range(len(expanded_top_bounds)):
            cv2.drawContours(gt_top, [expanded_top_bounds[i]], -1, i + 1, -1)
        # cv2.imwrite('demo.jpg', img)

        # attention: generating bottom boundary of text instance
        gt_bot = np.zeros(shape=img.shape[0:2], dtype='uint8')
        length = int(6 * scale_rate)
        bot_bounds = np.reshape(bot_bounds * ([img.shape[1], img.shape[0]] * 7), (bot_bounds.shape[0], 7, 2))
        expanded_bot_bounds = generate_bounds_shrinked(bot_bounds, radiuses, img_shape=img.shape[0:2], top=False)
        for i in range(len(expanded_bot_bounds)):
            cv2.drawContours(gt_bot, [expanded_bot_bounds[i]], -1, i + 1, -1)
        # cv2.imwrite('demo.jpg', img)

        assert len(expanded_top_bounds) == len(expanded_bot_bounds)
        assert len(bboxes) == len(kernel_bboxes)

        img_infos = [img, gt_text, gt_kernel, gt_top, gt_bot, training_mask]
        img_infos = random_horizontal_flip(img_infos)
        img_infos = random_rotate(img_infos)
        img_infos = random_crop_bound(img_infos, 640)
        img, gt_text, gt_kernel, gt_top, gt_bot, training_mask = img_infos[0], img_infos[1], img_infos[2], img_infos[3], \
                                                                 img_infos[4], img_infos[5]
        gt_text_labeled = gt_text.copy()
        gt_kernel_labeled = gt_kernel.copy()
        gt_top_labeled = gt_top.copy()
        gt_bot_labeled = gt_bot.copy()

        gt_text[gt_text > 0] = 1
        gt_kernel[gt_kernel > 0] = 1
        gt_top[gt_top > 0] = 1
        gt_bot[gt_bot > 0] = 1

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernel = torch.from_numpy(gt_kernel).float()
        gt_top = torch.from_numpy(gt_top).float()
        gt_bot = torch.from_numpy(gt_bot).float()

        gt_text_labeled = torch.from_numpy(gt_text_labeled).float()
        gt_kernel_labeled = torch.from_numpy(gt_kernel_labeled).float()
        gt_top_labeled = torch.from_numpy(gt_top_labeled).float()
        gt_bot_labeled = torch.from_numpy(gt_bot_labeled).float()

        training_mask = torch.from_numpy(training_mask).float()

        # assert torch.max(gt_top_labeled) == torch.max(gt_bot_labeled)

        return img, gt_text, gt_kernel, gt_top, gt_bot, gt_text_labeled, gt_kernel_labeled, gt_top_labeled, gt_bot_labeled, training_mask

class CTW1500E2E_Trainset(data.Dataset):
    ctw_root_dir = '/home/data1/IC19/CTW1500/'
    ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
    # ctw_train_anno_dir = ctw_root_dir + 'train/text_label_curve/'
    ctw_train_anno_dir_e2e = '/home/data1/zhm/dataset/CTW1500/ctw1500_e2e_train/'

    def __init__(self):
        self.img_paths = []
        self.anno_paths = []
        self.train_data_dir = self.ctw_train_data_dir
        self.train_anno_dir = self.ctw_train_anno_dir_e2e
        img_names = os.listdir(self.train_data_dir)
        for img_name in img_names:
            img_path = self.train_data_dir + img_name
            self.img_paths.append(img_path)
            anno_name = img_name.split('.')[0] + '.txt'
            anno_path = self.train_anno_dir + anno_name
            self.anno_paths.append(anno_path)
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        anno_path = self.anno_paths[index]
        img = read_image(img_path)
        img, scale_rate = random_scale(img, 640)



if __name__ == '__main__':
    dataset = CTW1500Trainset_BoundE2E()
    for i in range(100):
        dataset.__getitem__(i)



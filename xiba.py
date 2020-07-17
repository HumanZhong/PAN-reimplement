import matplotlib.pyplot as plt
import Polygon as plg
import numpy as np
import cv2
import torch
from models.post_processing import generate_result_purebound_baseline
from models.loss import get_pull_push_loss

from torch.autograd import Variable
# img = cv2.imread('/home/data3/IC19/ICDAR2015/Challenge4/ch4_test_images/img_128.jpg')
# #img = np.zeros(img.shape[0:2], dtype='uint8')
# original_img = img.copy()
# bound = [1135,67,1246,56,1246,74,1138,84]
# bound = np.array(bound).reshape(4,2)
# cv2.drawContours(img, [bound], -1, (255, 0, 0), 2)
# cv2.imwrite('caonima.jpg', img)


# a = np.array([[0,0,0,0,0],
#               [0,1,1,1,0],
#               [1,1,1,1,1],
#               [1,1,1,1,0],
#               [0,0,0,0,0]]).astype(np.float32)
# a = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).cuda()
# print(a)
# kernel = np.array([[-1],
#                    [1],
#                    [0]]).astype(np.float32)
# kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).cuda()
# print(kernel)
# output = torch.nn.functional.conv2d(a, kernel, padding=1)
# output = output[:,:,:,1:-1]
# result = output == a
# result = result.float()
# result = result * a
# print(result)
# b = np.array([[1,2,3,4,5,9],
#               [1,2,3,4,5,9],
#               [1,2,3,4,5,9],
#               [1,2,3,4,5,9]])
#
# n_sample = b.shape[1]
# b = np.expand_dims(b,1)
# b = np.tile(b, (1, 3, 1))
# print(b)
#
# kernel_list = np.array([[1, 1, 1, 1],
#                         [2, 2, 2, 2],
#                         [10, 10, 10, 10]])
# kernel_list = np.transpose(kernel_list, (1,0))
# kernel_list = np.expand_dims(kernel_list, 2)
# kernel_list = np.tile(kernel_list, (1,1,n_sample))
# print(kernel_list)
#
# diff = b - kernel_list
# print(diff)
#
# diff = np.linalg.norm(diff, axis=0)
# print(diff)
#
# diff = np.argmin(diff, axis=0)
# print(diff)
#
#
#
# c = [1, 1, 1, 0, 1, 1]
# c = np.array(c)
# category = [1, 2, 3, 4, 5]
# category = np.array(category)
# c[c > 0.5] = category
# print(c)
#
# d = np.random.rand(10, 10, 3)
# print(d)
# index = np.random.rand(10, 10)
# index[index > 0.5] = 1
# index[index <= 0.5] = 0
# index = index.astype(np.uint8)
# print(index)
# d[index == 1, :] = [5, 5, 5]
# print(d)


# -----------------------------------get_pull_push_loss test------------------------------
# outputs = np.zeros(shape=(1, 4, 10, 10))
# outputs[:, 0, 1:5, 1:5] = 1
# outputs[:, 0, 3:7, 6:10] = 1
# outputs[:, 1, 2:4, 2:4] = 1
# outputs[:, 1, 4:6, 7:9] = 1
# sim_vectors = np.random.randint(0, 10, size=(1, 1, 10, 10))
# # sim_vectors[sim_vectors > 0.5] = 1
# # sim_vectors[sim_vectors < 0.5] = 0
#
# print(outputs)
# print(sim_vectors)
#
# outputs = np.concatenate((outputs, sim_vectors), axis=1)
# print(np.shape(outputs))
#
# outputs = torch.from_numpy(outputs).float()
# outputs = Variable(outputs.cuda())
#
# gt_texts = np.zeros(shape=(1, 1, 10, 10))
# gt_texts[:, 0, 1:5, 1:5] = 1
# gt_texts[:, 0, 3:7, 6:10] = 2
# gt_kernels = np.zeros(shape=(1, 1, 10, 10))
# gt_kernels[:, 0, 2:4, 2:4] = 1
# gt_kernels[:, 0, 4:6, 7:9] = 2
# gt_tops = np.zeros(shape=(1, 1, 10, 10))
# gt_bots = np.zeros(shape=(1, 1, 10, 10))
# gt_tops[:, 0, 1:2, 1:5] = 1
# gt_tops[:, 0, 3:4, 6:10] = 2
# gt_bots[:, 0, 4:5, 1:5] = 1
# gt_bots[:, 0, 6:7, 6:10] = 2
#
# gt_texts = torch.from_numpy(gt_texts).float().cuda()
# gt_kernels = torch.from_numpy(gt_kernels).float().cuda()
# gt_tops = torch.from_numpy(gt_tops).float().cuda()
# gt_bots = torch.from_numpy(gt_bots).float().cuda()
#
#
#
# loss = get_pull_push_loss(outputs, gt_texts, gt_kernels, gt_tops, gt_bots)


# d = np.zeros(shape=(0))
# sim_vector = np.random.rand(4, 3)
# tmp = sim_vector[:, [False, False, False]]
# print(tmp)


e = np.array([[5, 0], [5, 1], [4, 1], [3, 1], [5, 2]])
idx = 1
y = e[idx, 0]
x = e[idx, 1]
loc = np.where(e[:, 1] == x)
print(np.shape(loc))
candidate = e[loc, 0]
print(candidate)
final_y = np.max(candidate)
print(final_y)

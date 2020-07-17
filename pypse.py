import numpy as np
import cv2
import queue

def pypse(kernels, min_area, connectivity=4):
    # attention: kernels中的各个kernel按照index从小到大的顺序应该是逐渐减小的
    # 也就是说kernels[0]是最大的kernel，kernels[num_kernel - 1]是最小的kernel

    num_kernel = len(kernels)
    pred = np.zeros(shape=kernels[0].shape, dtype='int32')

    # attention：cv2.connectedComponents函数只接受np.int8类型的数据，因此需要将kernels转变为np.int8
    label_num, label = cv2.connectedComponents(kernels[num_kernel - 1], connectivity=connectivity)

    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0

    q = queue.Queue(maxsize=0)
    next_q = queue.Queue(maxsize=0)

    points = np.array(np.where(label > 0)).transpose((1, 0))

    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        q.put((x, y, label[x, y]))
        pred[x, y] = label[x, y]

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for idx in range(num_kernel - 2, -1, -1):
        kernel = kernels[idx].copy()
        while not q.empty():
            (x, y, l) = q.get()
            is_edge = True
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernel.shape[0] or tmpy < 0 or tmpy >= kernel.shape[1]:
                    continue
                if kernel[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue
                q.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                next_q.put((x, y, l))
        q, next_q = next_q, q

    return pred

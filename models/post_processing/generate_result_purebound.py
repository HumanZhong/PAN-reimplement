import numpy as np
import torch
import cv2
import time
from queue import Queue

# generate: text bbox by clustering
def generate_result_purebound(output, ori_img):
    # 默认测试时batch_size=1
    score_text = torch.sigmoid(output[:, 0, :, :])
    output[:, 0:4, :, :] = (torch.sign(output[:, 0:4, :, :] - 1.0) + 1) / 2

    pred_text = output[:, 0, :, :]
    pred_kernel = output[:, 1, :, :]
    pred_top = output[:, 2, :, :]
    pred_bot = output[:, 3, :, :]
    pred_sim_vector = output[:, 4:, :, :]

    pred_kernel = pred_kernel * pred_text

    batch_size = pred_text.size(0)
    h = pred_text.size(1)
    w = pred_text.size(2)
    pred_text = pred_text.contiguous().reshape(batch_size, -1)
    # pred_kernel = pred_kernel.contiguous().reshape(batch_size, -1)
    # pred_top = pred_top.contiguous().reshape(batch_size, -1)
    # pred_bot = pred_bot.contiguous().reshape(batch_size, -1)
    pred_sim_vector = pred_sim_vector.contiguous().view(batch_size, 4, -1)

    score_text = score_text.data.cpu().numpy()[0].astype(np.float32)
    pred_text = pred_text.data.cpu().numpy()[0].astype(np.uint8)
    pred_kernel = pred_kernel.data.cpu().numpy()[0].astype(np.uint8)
    # pred_top = pred_top.data.cpu().numpy()[0].astype(np.uint8)
    # pred_bot = pred_bot.data.cpu().numpy()[0].astype(np.uint8)
    pred_sim_vector = pred_sim_vector.data.cpu().numpy()[0].astype(np.float32)

    # generate kernel result
    label_num, label = cv2.connectedComponents(pred_kernel, connectivity=4)
    label = label.reshape(batch_size, -1)[0]


    min_area = 5
    kernel_list = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
        current_kernel = label[label == label_idx]
        current_kernel_sim_vector = pred_sim_vector[:, label == label_idx]
        current_kernel_sim_vector = np.mean(current_kernel_sim_vector, axis=1)
        kernel_list.append(current_kernel_sim_vector)
    kernel_list = np.array(kernel_list)

    # print(np.shape(kernel_list))
    pred_top = pred_top.unsqueeze(0)
    pred_bot = pred_bot.unsqueeze(0)
    top_conv_kernel = np.array([[-1],
                                [1],
                                [0]]).astype(np.float32)
    top_conv_kernel = torch.from_numpy(top_conv_kernel).unsqueeze(0).unsqueeze(0).cuda()

    bot_conv_kernel = np.array([[0],
                                [1],
                                [-1]]).astype(np.float32)
    bot_conv_kernel = torch.from_numpy(bot_conv_kernel).unsqueeze(0).unsqueeze(0).cuda()
    output_top = torch.nn.functional.conv2d(pred_top, top_conv_kernel, padding=1)
    output_top = output_top[:, :, :, 1:-1]
    output_top = output_top == pred_top
    output_top = output_top.float() * pred_top
    output_bot = torch.nn.functional.conv2d(pred_bot, bot_conv_kernel, padding=1)
    output_bot = output_bot[:, :, :, 1:-1]
    output_bot = output_bot == pred_bot
    output_bot = output_bot.float() * pred_bot
    # print(output_top)
    # print(output_bot)
    top_mask = output_top[output_top > 0.5]
    bot_mask = output_bot[output_bot > 0.5]
    output_top = output_top.contiguous().view(batch_size, -1)
    output_top = output_top.data.cpu().numpy()[0].astype(np.uint8)
    output_bot = output_bot.contiguous().view(batch_size, -1)
    output_bot = output_bot.data.cpu().numpy()[0].astype(np.uint8)
    top_sim_vector = pred_sim_vector[:, output_top > 0.5]
    bot_sim_vector = pred_sim_vector[:, output_bot > 0.5]
    # print(np.shape(top_sim_vector))
    # print(np.shape(bot_sim_vector))
    n_top = np.shape(top_sim_vector)[1]
    n_bot = np.shape(bot_sim_vector)[1]
    n_kernel = np.shape(kernel_list)[0]
    top_sim_vector = np.expand_dims(top_sim_vector, 1)
    top_sim_vector = np.tile(top_sim_vector, (1, n_kernel, 1))
    bot_sim_vector = np.expand_dims(bot_sim_vector, 1)
    bot_sim_vector = np.tile(bot_sim_vector, (1, n_kernel, 1))
    kernel_list_transposed = np.transpose(kernel_list, (1, 0))
    kernel_list_transposed = np.expand_dims(kernel_list_transposed, 2)
    kernel_list_top = np.tile(kernel_list_transposed, (1, 1, n_top))
    kernel_list_bot = np.tile(kernel_list_transposed, (1, 1, n_bot))
    label_top = np.argmin(np.linalg.norm(top_sim_vector - kernel_list_top, axis=0), axis=0)
    label_bot = np.argmin(np.linalg.norm(bot_sim_vector - kernel_list_bot, axis=0), axis=0)
    # print(label_top)
    # print(label_bot)
    output_top[output_top > 0.5] = label_top + 1
    output_bot[output_bot > 0.5] = label_bot + 1
    output_top = np.reshape(output_top, (h, w))
    output_bot = np.reshape(output_bot, (h, w))
    bboxes = []
    for i in range(len(kernel_list)):
        top_index = np.nonzero(output_top == (i + 1))
        bot_index = np.nonzero(output_bot == (i + 1))
        top_index = np.transpose(top_index, (1, 0))
        bot_index = np.transpose(bot_index, (1, 0))
        top_index = top_index[np.argsort(top_index[:, 1])]
        bot_index = bot_index[np.argsort(bot_index[:, 1])]
        if len(top_index) == 0 or len(bot_index) == 0:
            continue
        n1 = len(top_index)
        n2 = len(bot_index)
        bbox = np.array([top_index[0], top_index[n1 // 3], top_index[n1 // 3 * 2], top_index[-1], bot_index[-1], bot_index[n2 // 3 * 2], bot_index[n2 // 3], bot_index[0]])
        bboxes.append(bbox[:, [1, 0]])
    # print(bboxes)
    return bboxes


# generate: text bbox by pypse
def generate_result_purebound_v2(output, ori_img):
    current_time = time.time()
    # 默认测试时batch_size=1
    score_text = torch.sigmoid(output[:, 0, :, :])
    output[:, 0:4, :, :] = (torch.sign(output[:, 0:4, :, :] - 1.0) + 1) / 2

    pred_text = output[:, 0, :, :]
    pred_kernel = output[:, 1, :, :]
    # pred_top = output[:, 2, :, :] * pred_text
    # pred_bot = output[:, 3, :, :] * pred_text
    pred_top = output[:, 2, :, :]
    pred_bot = output[:, 3, :, :]
    pred_sim_vector = output[:, 4:, :, :]

    pred_kernel = pred_kernel * pred_text

    batch_size = pred_text.size(0)
    h = pred_text.size(1)
    w = pred_text.size(2)
    # pred_sim_vector = pred_sim_vector.contiguous().view(batch_size, 4, -1)

    score_text = score_text.data.cpu().numpy()[0].astype(np.float32)
    pred_text = pred_text.data.cpu().numpy()[0].astype(np.uint8)
    pred_kernel = pred_kernel.data.cpu().numpy()[0].astype(np.uint8)
    # pred_top = pred_top.data.cpu().numpy()[0].astype(np.uint8)
    # pred_bot = pred_bot.data.cpu().numpy()[0].astype(np.uint8)
    top_cpu = pred_top.data.cpu().numpy()[0].astype(np.uint8)
    bot_cpu = pred_bot.data.cpu().numpy()[0].astype(np.uint8)
    pred_sim_vector = pred_sim_vector.data.cpu().numpy()[0].astype(np.float32)

    # generate kernel result
    label_num, label = cv2.connectedComponents(pred_kernel, connectivity=4)
    # label = label.reshape(batch_size, -1)[0]

    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < 10:
            label[label == label_idx] = 0
        if np.mean(score_text[label == label_idx]) < 0.92:
            label[label == label_idx] = 0

    print('connected_component:', time.time() - current_time)
    current_time = time.time()

    output_text = pypse_v2(pred_text, pred_sim_vector, label, label_num, top=top_cpu, bot=bot_cpu, dis_thres=6.0)

    print('pypse_v2', time.time() - current_time)
    current_time = time.time()

    top_conv_kernel = np.array([[-1],
                                [1],
                                [0]]).astype(np.float32)
    top_conv_kernel = torch.from_numpy(top_conv_kernel).unsqueeze(0).unsqueeze(0).cuda()
    bot_conv_kernel = np.array([[0],
                                [1],
                                [-1]]).astype(np.float32)
    bot_conv_kernel = torch.from_numpy(bot_conv_kernel).unsqueeze(0).unsqueeze(0).cuda()

    pred_top = pred_top.unsqueeze(0)
    pred_bot = pred_bot.unsqueeze(0)

    bboxes = []
    for i in range(1, label_num):
        tmp_text = np.zeros(output_text.shape)
        tmp_text[output_text == i] = 1
        if np.sum(tmp_text) == 0:
            continue
        tmp_text = torch.from_numpy(tmp_text).float().cuda()

        tmp_top = pred_top * tmp_text
        tmp_top = torch.nn.functional.conv2d(tmp_top, top_conv_kernel, padding=1)
        tmp_top = tmp_top[:, :, :, 1:-1]
        tmp_top = (tmp_top == pred_top)
        tmp_top = tmp_top.float() * pred_top

        tmp_bot = pred_bot * tmp_text
        tmp_bot = torch.nn.functional.conv2d(tmp_bot, bot_conv_kernel, padding=1)
        tmp_bot = tmp_bot[:, :, :, 1:-1]
        tmp_bot = (tmp_bot == pred_bot)
        tmp_bot = tmp_bot.float() * pred_bot

        # tmp_top = tmp_top.contiguous().view(batch_size, -1)
        # tmp_bot = tmp_bot.contiguous().view(batch_size, -1)
        tmp_top = tmp_top.data.cpu().numpy()[0, 0].astype(np.uint8)
        tmp_bot = tmp_bot.data.cpu().numpy()[0, 0].astype(np.uint8)

        top_index = np.nonzero(tmp_top > 0.5)
        top_index = np.transpose(top_index, (1, 0))
        bot_index = np.nonzero(tmp_bot > 0.5)
        bot_index = np.transpose(bot_index, (1, 0))
        top_index = top_index[np.argsort(top_index[:, 1])]
        bot_index = bot_index[np.argsort(bot_index[:, 1])]

        if len(top_index) == 0 or len(bot_index) == 0:
            continue
        n1 = len(top_index)
        n2 = len(bot_index)
        # TODO: 此处需要过滤掉那些不属于当前文本行的top和bot信息
        # bbox = np.array([top_index[0], top_index[n1 // 4], top_index[n1 // 2], top_index[n1 // 4 * 3], top_index[-1],
        #                  bot_index[-1], bot_index[n2 // 4 * 3], bot_index[n2 // 2], bot_index[n2 // 4], bot_index[0]])
        bbox = []
        t = [0, n1 // 4, n1 // 2, n1 // 4 * 3, -1]
        b = [-1, n2 // 4 * 3, n2 // 2, n2 // 4, 0]
        for k in t:
            top_coord = top_index[k]
            y = top_coord[0]
            x = top_coord[1]
            loc = np.where(top_index[:, 1] == x)
            if np.shape(loc)[1] > 1:
                candidate = top_index[loc, 0]
                final_y = np.min(candidate)
                y = final_y
                bbox.append([y, x])
            else:
                bbox.append([y, x])
        for k in b:
            bot_coord = bot_index[k]
            y = bot_coord[0]
            x = bot_coord[1]
            loc = np.where(bot_index[:, 1] == x)
            if np.shape(loc)[1] > 1:
                candidate = bot_index[loc, 0]
                final_y = np.max(candidate)
                y = final_y
                bbox.append([y, x])
            else:
                bbox.append([y, x])
        bbox = np.array(bbox)
        bboxes.append(bbox[:, [1, 0]])
    print('find_top&bot_total:', time.time() - current_time)
    print('find_top&bot_per:', (time.time() - current_time) / (label_num + 0.01))
    return bboxes


# generate: text segmentation
def generate_result_PAN(output, ori_img, threshold=1.0):
    score_text = torch.sigmoid(output[:, 0, :, :])
    output[:, 0:2, :, :] = (torch.sign(output[:, 0:2, :, :] - threshold) + 1) / 2
    pred_text = output[:, 0, :, :]
    pred_kernel = output[:, 1, :, :]
    pred_sim_vector = output[:, 2:, :, :]
    pred_kernel = pred_kernel * pred_text
    batch_size = pred_text.size(0)
    h = pred_text.size(1)
    w = pred_text.size(2)

    score_text = score_text.data.cpu().numpy()[0].astype(np.float32)
    pred_text = pred_text.data.cpu().numpy()[0].astype(np.uint8)
    pred_kernel = pred_kernel.data.cpu().numpy()[0].astype(np.uint8)
    pred_sim_vector = pred_sim_vector.data.cpu().numpy()[0].astype(np.float32)

    label_num, label = cv2.connectedComponents(pred_kernel, connectivity=4)
    output_text = pypse_v2(pred_text, pred_sim_vector, label, label_num, top=None, bot=None, dis_thres=6.0)

    bboxes = []
    scale = (ori_img.shape[1] / output_text.shape[1], ori_img.shape[0] / output_text.shape[0])
    for i in range(1, label_num):
        points_loc = np.array(np.where(output_text == i)).transpose((1, 0))
        # points = points[:,::-1]
        if points_loc.shape[0] < 10:
            continue
        score_i = np.mean(score_text[output_text == i])
        if score_i < 0.85:
            continue
        rect = cv2.minAreaRect(points_loc)
        bbox = cv2.boxPoints(rect) * scale
        bbox = bbox.astype(int)
        bboxes.append(bbox.reshape(-1))
    return bboxes


# generate: text segmentation + top segmentation + bot segmentation
def generate_result_purebound_baseline(output, ori_img, threshold=1.0):
    score_text = torch.sigmoid(output[:, 0, :, :])
    output[:, 0:4, :, :] = (torch.sign(output[:, 0:4, :, :] - threshold) + 1) / 2

    pred_text = output[:, 0, :, :]
    pred_kernel = output[:, 1, :, :]
    # pred_top = output[:, 2, :, :] * pred_text
    # pred_bot = output[:, 3, :, :] * pred_text
    pred_top = output[:, 2, :, :]
    pred_bot = output[:, 3, :, :]
    pred_sim_vector = output[:, 4:, :, :]

    pred_kernel = pred_kernel * pred_text

    batch_size = pred_text.size(0)
    h = pred_text.size(1)
    w = pred_text.size(2)
    # pred_sim_vector = pred_sim_vector.contiguous().view(batch_size, 4, -1)

    score_text = score_text.data.cpu().numpy()[0].astype(np.float32)
    pred_text = pred_text.data.cpu().numpy()[0].astype(np.uint8)
    pred_kernel = pred_kernel.data.cpu().numpy()[0].astype(np.uint8)
    pred_top = pred_top.data.cpu().numpy()[0].astype(np.uint8)
    pred_bot = pred_bot.data.cpu().numpy()[0].astype(np.uint8)
    pred_sim_vector = pred_sim_vector.data.cpu().numpy()[0].astype(np.float32)

    # generate kernel result
    label_num, label = cv2.connectedComponents(pred_kernel, connectivity=4)
    # label = label.reshape(batch_size, -1)[0]

    # for label_idx in range(1, label_num):
    #     if np.sum(label == label_idx) < 10:
    #         label[label == label_idx] = 0
    #     if np.mean(score_text[label == label_idx]) < 0.95:
    #         label[label == label_idx] = 0
    output_text = pypse_v2(pred_text, pred_sim_vector, label, label_num, top=None, bot=None, dis_thres=6.0)

    bboxes = []
    scale = (ori_img.shape[1] / output_text.shape[1], ori_img.shape[0] / output_text.shape[0])
    for i in range(1, label_num):
        points_loc = np.array(np.where(output_text == i)).transpose((1, 0))
        # points = points[:,::-1]
        if points_loc.shape[0] < 50:
            continue
        score_i = np.mean(score_text[output_text == i])
        if score_i < 0.90:
            continue
        binary = np.zeros(output_text.shape, dtype='uint8')
        binary[output_text == i] = 1
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        contour = contour * scale
        if contour.shape[0] <= 2:
            continue
        contour = contour.astype('int32')
        bboxes.append(contour.reshape(-1))
    pred_top = cv2.resize(pred_top, dsize=None, fx=scale[0], fy=scale[1])
    pred_bot = cv2.resize(pred_bot, dsize=None, fx=scale[0], fy=scale[1])
    ori_img[pred_top > 0.5, :] = [255, 0, 0]
    ori_img[pred_bot > 0.5, :] = [0, 0, 255]
    return bboxes, ori_img




def pypse_v2(text, sim_vector, label, label_values, top=None, bot=None, dis_thres=0.8):

    pred = np.zeros(text.shape)
    queue = Queue(maxsize=0)

    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        y, x = points[point_idx, 0], points[point_idx, 1]
        label_value = label[y, x]
        queue.put((y, x, label_value))
        pred[y, x] = label_value

    d = {}
    for i in range(1, label_values):
        kernel_idx = (label == i)
        if np.sum(kernel_idx) == 0:
            d[i] = np.zeros(shape=(0))
        else:
            kernel_sim_vector = sim_vector[:, kernel_idx]
            kernel_sim_vector = np.mean(kernel_sim_vector, axis=1)
            d[i] = kernel_sim_vector

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    kernel = text.copy()
    while not queue.empty():
        (y, x, label_value) = queue.get()
        current_kernel_sim_vector = d[label_value]
        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpx >= kernel.shape[1] or tmpy < 0 or tmpy >= kernel.shape[0]:
                continue
            if top is None and bot is None:
                if kernel[tmpy, tmpx] == 0 or pred[tmpy, tmpx] > 0:
                    continue
            else:
                if (kernel[tmpy, tmpx] == 0 and top[tmpy, tmpx] == 0 and bot[tmpy, tmpx] == 0) or pred[tmpy, tmpx] > 0:
                    continue
            distance = np.linalg.norm(sim_vector[:, tmpy, tmpx] - current_kernel_sim_vector)
            if distance >= dis_thres:
                # print(distance)
                continue
            queue.put((tmpy, tmpx, label_value))
            pred[tmpy, tmpx] = label_value
    return pred

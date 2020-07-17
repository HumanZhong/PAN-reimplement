import torch
import torch.nn as nn
import itertools
import numpy as np
from myutils.ohem import ohem_batch
from torch.autograd import Variable
from .dice_loss import dice_loss


def get_pull_push_loss_PAN(outputs, gt_texts, gt_kernels):
    pred_texts = outputs[:, 0, :, :]
    pred_kernels = outputs[:, 1, :, :]
    pred_sim_vectors = outputs[:, 2:, :, :]

    batch_size = pred_texts.size(0)
    sim_vector_channel = pred_sim_vectors.size(1)
    pred_texts = pred_texts.contiguous().reshape(batch_size, -1)
    pred_kernels = pred_kernels.contiguous().reshape(batch_size, -1)
    pred_sim_vectors = pred_sim_vectors.contiguous().view(batch_size, sim_vector_channel, -1)

    gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
    gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)

    loss_pulls = []
    loss_pushs = []
    for text_i, kernel_i, gt_text_i, gt_kernel_i, sim_vector_i \
            in zip(pred_texts, pred_kernels, gt_texts, gt_kernels, pred_sim_vectors):
        text_num = gt_text_i.max().item() + 1
        loss_pull_text_single = []
        G_kernel_list = []
        for text_idx in range(1, int(text_num)):
            current_kernel_mask = (gt_kernel_i == text_idx)
            if current_kernel_mask.sum() == 0 or torch.sum(gt_text_i == text_idx) == 0:
                continue
            G_kernel = sim_vector_i[:, current_kernel_mask].mean(1)
            G_kernel_list.append(G_kernel)

            text_sim_vector = sim_vector_i[:, gt_text_i == text_idx]

            # top_sim_vector = sim_vector_i[:, gt_top_i == text_idx]
            # bot_sim_vector = sim_vector_i[:, gt_bot_i == text_idx]

            text_distance = (text_sim_vector - G_kernel.reshape(sim_vector_channel, 1)).norm(2, dim=0) - 0.5
            text_distance = torch.max(text_distance,
                                      torch.tensor(0, device=text_distance.device, dtype=torch.float)).pow(2)
            loss_text_agg = torch.log(text_distance + 1).mean()
            loss_pull_text_single.append(loss_text_agg)
        if len(loss_pull_text_single) > 0:
            loss_pull_text_single = torch.stack(loss_pull_text_single).mean()
            # loss_pull_top_single = torch.stack(loss_pull_top_single).mean()
            # loss_pull_bot_single = torch.stack(loss_pull_bot_single).mean()
        else:
            loss_pull_text_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)
            # loss_pull_top_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)
            # loss_pull_bot_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)
        loss_pulls.append(loss_pull_text_single)

        loss_push_single = 0
        for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
            kernel_ij = 3 - (G_kernel_i - G_kernel_j).norm(2)
            D_kernel_ij = torch.max(kernel_ij, torch.tensor(0, device=kernel_ij.device, dtype=torch.float)).pow(2)
            loss_push_single += torch.log(D_kernel_ij + 1)

        if len(G_kernel_list) > 1:
            loss_push_single /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
        else:
            loss_push_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)
        loss_pushs.append(loss_push_single)

    return torch.stack(loss_pulls).mean(), torch.stack(loss_pushs).mean()




def get_pull_push_loss(outputs, gt_texts, gt_kernels, gt_tops, gt_bots):
    pred_texts = outputs[:, 0, :, :]
    pred_kernels = outputs[:, 1, :, :]
    pred_tops = outputs[:, 2, :, :]
    pred_bots = outputs[:, 3, :, :]
    pred_sim_vectors = outputs[:, 4:, :, :]

    batch_size = pred_texts.size(0)
    sim_vector_channel = pred_sim_vectors.size(1)
    pred_texts = pred_texts.contiguous().reshape(batch_size, -1)
    pred_kernels = pred_kernels.contiguous().reshape(batch_size, -1)
    pred_tops = pred_tops.contiguous().reshape(batch_size, -1)
    pred_bots = pred_bots.contiguous().reshape(batch_size, -1)
    pred_sim_vectors = pred_sim_vectors.contiguous().view(batch_size, sim_vector_channel, -1)

    gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
    gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)
    gt_tops = gt_tops.contiguous().reshape(batch_size, -1)
    gt_bots = gt_bots.contiguous().reshape(batch_size, -1)

    loss_pulls = []
    loss_pushs = []
    for text_i, kernel_i, top_i, bot_i, gt_text_i, gt_kernel_i, gt_top_i, gt_bot_i, sim_vector_i \
            in zip(pred_texts, pred_kernels, pred_tops, pred_bots, gt_texts, gt_kernels, gt_tops,
                   gt_bots, pred_sim_vectors):
        text_num = gt_text_i.max().item() + 1
        loss_pull_text_single = []
        loss_pull_top_single = []
        loss_pull_bot_single = []
        G_kernel_list = []
        # attention: calculate the pull loss in one image
        for text_idx in range(1, int(text_num)):
            current_kernel_mask = (gt_kernel_i == text_idx)
            if current_kernel_mask.sum() == 0 or torch.sum(gt_text_i == text_idx) == 0 or torch.sum(gt_top_i == text_idx) == 0 or torch.sum(gt_bot_i == text_idx) == 0:
                continue
            G_kernel = sim_vector_i[:, current_kernel_mask].mean(1)
            G_kernel_list.append(G_kernel)

            text_sim_vector = sim_vector_i[:, gt_text_i == text_idx]

            top_sim_vector = sim_vector_i[:, gt_top_i == text_idx]
            bot_sim_vector = sim_vector_i[:, gt_bot_i == text_idx]

            text_distance = (text_sim_vector - G_kernel.reshape(sim_vector_channel, 1)).norm(2, dim=0) - 0.5
            text_distance = torch.max(text_distance,
                                      torch.tensor(0, device=text_distance.device, dtype=torch.float)).pow(2)
            loss_text_agg = torch.log(text_distance + 1).mean()
            loss_pull_text_single.append(loss_text_agg)

            top_distance = (top_sim_vector - G_kernel.reshape(sim_vector_channel, 1)).norm(2, dim=0) - 0.5
            top_distance = torch.max(top_distance,
                                     torch.tensor(0, device=top_distance.device, dtype=torch.float)).pow(2)
            loss_top_agg = torch.log(top_distance + 1).mean()
            loss_pull_top_single.append(loss_top_agg)

            bot_distance = (bot_sim_vector - G_kernel.reshape(sim_vector_channel, 1)).norm(2, dim=0) - 0.5
            bot_distance = torch.max(bot_distance,
                                     torch.tensor(0, device=bot_distance.device, dtype=torch.float)).pow(2)
            loss_bot_agg = torch.log(bot_distance + 1).mean()
            loss_pull_bot_single.append(loss_bot_agg)

        if len(loss_pull_text_single) > 0:
            loss_pull_text_single = torch.stack(loss_pull_text_single).mean()
            loss_pull_top_single = torch.stack(loss_pull_top_single).mean()
            loss_pull_bot_single = torch.stack(loss_pull_bot_single).mean()
        else:
            loss_pull_text_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)
            loss_pull_top_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)
            loss_pull_bot_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)

        # loss_pulls.append(loss_pull_text_single + loss_pull_top_single + loss_pull_bot_single)
        loss_pulls.append(loss_pull_text_single)

        # attention: calculate the push loss in one image
        loss_push_single = 0
        for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
            kernel_ij = 3 - (G_kernel_i - G_kernel_j).norm(2)
            D_kernel_ij = torch.max(kernel_ij, torch.tensor(0, device=kernel_ij.device, dtype=torch.float)).pow(2)
            loss_push_single += torch.log(D_kernel_ij + 1)

        if len(G_kernel_list) > 1:
            loss_push_single /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
        else:
            loss_push_single = torch.tensor(0, device=pred_texts.device, dtype=torch.float)
        loss_pushs.append(loss_push_single)
    return torch.stack(loss_pulls).mean(), torch.stack(loss_pushs).mean()


class BoundLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, outputs, gts, training_masks):
        pred_texts = outputs[:, 0, :, :]
        pred_kernels = outputs[:, 1, :, :]
        pred_tops = outputs[:, 2, :, :]
        pred_bots = outputs[:, 3, :, :]
        pred_sim_vectors = outputs[:, 4:, :, :]

        # TODO: figure out how to implement this part
        # attention: gt_texts should label each text instance differently.
        gt_texts = None
        gt_kernels = None
        gt_tops = None
        gt_bots = None

        # text loss
        selected_masks_text = ohem_batch(pred_texts, gt_texts, training_masks)


        # kernel loss

        # top&bot loss

        # pull&push loss


    def pull_push_loss(self, outputs, gt_texts, gt_kernels, gt_tops, gt_bots):
        pred_texts = outputs[:, 0, :, :]
        pred_kernels = outputs[:, 1, :, :]
        pred_tops = outputs[:, 2, :, :]
        pred_bots = outputs[:, 3, :, :]
        pred_sim_vectors = outputs[:, 4:, :, :]

        batch_size = pred_texts.size(0)
        pred_texts = pred_texts.contiguous().reshape(batch_size, -1)
        pred_kernels = pred_kernels.contiguous().reshape(batch_size, -1)
        pred_tops = pred_tops.contiguous().reshape(batch_size, -1)
        pred_bots = pred_bots.contiguous().reshape(batch_size, -1)
        pred_sim_vectors = pred_sim_vectors.contiguous().view(batch_size, 4, -1)

        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)

        loss_pulls = []
        loss_pushs = []
        for text_i, kernel_i, top_i, bot_i, gt_text_i, gt_kernel_i, gt_top_i, gt_bot_i, sim_vector_i \
                in zip(pred_texts, pred_kernels, pred_tops, pred_bots, gt_texts, gt_kernels, gt_tops,
                       gt_bots, pred_sim_vectors):
            text_num = gt_text_i.max().item() + 1
            loss_pull_text_single = []
            loss_pull_top_single = []
            loss_pull_bot_single = []
            G_kernel_list = []
            # attention: calculate the pull loss in one image
            for text_idx in range(1, int(text_num)):
                current_kernel_mask = (gt_kernel_i == text_idx)
                if current_kernel_mask.sum() == 0:
                    continue
                G_kernel = sim_vector_i[:, current_kernel_mask].mean(1)
                G_kernel_list.append(G_kernel)

                text_sim_vector = sim_vector_i[:, gt_text_i == text_idx]

                top_sim_vector = sim_vector_i[:, gt_top_i == text_idx]
                bot_sim_vector = sim_vector_i[:, gt_bot_i == text_idx]

                text_distance = (text_sim_vector - G_kernel.reshape(4, 1)).norm(2, dim=0) - 0.5
                text_distance = torch.max(text_distance, torch.tensor(0, device=text_distance.device, dtype=torch.float)).pow(2)
                loss_text_agg = torch.log(text_distance + 1).mean()
                loss_pull_text_single.append(loss_text_agg)

                top_distance = (top_sim_vector - G_kernel.reshape(4, 1)).norm(2, dim=0) - 0.5
                top_distance = torch.max(top_distance, torch.Tensor(0, device=top_distance.device, dtype=torch.float)).pow(2)
                loss_top_agg = torch.log(top_distance + 1).mean()
                loss_pull_top_single.append(loss_top_agg)

                bot_distance = (bot_distance - G_kernel.reshape(4, 1)).norm(2, dim=0)
                bot_distance = torch.max(bot_distance, torch.Tensor(0, device=bot_distance.device, dtype=torch.float)).pow(2)
                loss_bot_agg = torch.log(bot_distance + 1).mean()
                loss_pull_bot_single.append(loss_bot_agg)

            if len(loss_pull_text_single) > 0:
                loss_pull_text_single = torch.stack(loss_pull_text_single).mean()
                loss_pull_top_single = torch.stack(loss_pull_top_single).mean()
                loss_pull_bot_single = torch.stack(loss_pull_bot_single).mean()
            else:
                raise NotImplementedError
            loss_pulls.append(loss_pull_text_single)

            # attention: calculate the push loss in one image
            loss_push_single = 0
            for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
                kernel_ij = 3 - (G_kernel_i - G_kernel_j).norm(2)
                D_kernel_ij = torch.max(kernel_ij, torch.tensor(0, device=kernel_ij.device, dtype=torch.float)).pow(2)
                loss_push_single += torch.log(D_kernel_ij + 1)


            if len(G_kernel_list) > 1:
                loss_push_single /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
            else:
                loss_push_single = torch.Tensor(0)
            loss_pushs.append(loss_push_single)
        return torch.stack(loss_pulls), torch.stack(loss_pushs)





class PANLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean'):
        """
        Implement PSE Loss.
        :param alpha: loss kernel 前面的系数
        :param beta: loss agg 和 loss dis 前面的系数
        :param delta_agg: 计算loss agg时的常量
        :param delta_dis: 计算loss dis时的常量
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.delta_agg = delta_agg
        self.delta_dis = delta_dis
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, outputs, gt_texts, gt_kernels, training_masks):
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1, :, :]
        # gt_texts = labels[:, 0, :, :]
        # gt_kernels = labels[:, 1, :, :]


        # 计算 agg loss 和 dis loss
        similarity_vectors = outputs[:, 2:, :, :]
        loss_aggs, loss_diss = self.agg_dis_loss(texts, kernels, gt_texts, gt_kernels, similarity_vectors)

        # 计算 text loss
        selected_masks = self.ohem_batch(texts, gt_texts, training_masks)
        selected_masks = selected_masks.to(outputs.device)

        loss_texts = self.dice_loss(texts, gt_texts, selected_masks)

        # 计算 kernel loss
        # selected_masks = ((gt_texts > 0.5) & (training_masks > 0.5)).float()
        mask0 = torch.sigmoid(texts).detach().cpu().numpy()
        mask1 = training_masks.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float().to(texts.device)
        loss_kernels = self.dice_loss(kernels, gt_kernels, selected_masks)

        # mean or sum
        if self.reduction == 'mean':
            loss_text = loss_texts.mean()
            loss_kernel = loss_kernels.mean()
            loss_agg = loss_aggs.mean()
            loss_dis = loss_diss.mean()
        elif self.reduction == 'sum':
            loss_text = loss_texts.sum()
            loss_kernel = loss_kernels.sum()
            loss_agg = loss_aggs.sum()
            loss_dis = loss_diss.sum()
        else:
            raise NotImplementedError

        loss_all = loss_text + self.alpha * loss_kernel + self.beta * (loss_agg + loss_dis)
        return loss_all, loss_text, loss_kernel, loss_agg, loss_dis

    def agg_dis_loss(self, texts, kernels, gt_texts, gt_kernels, similarity_vectors):
        """
        计算 loss agg
        :param texts: 文本实例的分割结果 batch_size * (w*h)
        :param kernels: 缩小的文本实例的分割结果 batch_size * (w*h)
        :param gt_texts: 文本实例的gt batch_size * (w*h)
        :param gt_kernels: 缩小的文本实例的gt batch_size*(w*h)
        :param similarity_vectors: 相似度向量的分割结果 batch_size * 4 *(w*h)
        :return:
        """
        batch_size = texts.size()[0]
        texts = texts.contiguous().reshape(batch_size, -1)
        kernels = kernels.contiguous().reshape(batch_size, -1)
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)
        similarity_vectors = similarity_vectors.contiguous().view(batch_size, 4, -1)
        loss_aggs = []
        loss_diss = []
        for text_i, kernel_i, gt_text_i, gt_kernel_i, similarity_vector in zip(texts, kernels, gt_texts, gt_kernels,
                                                                               similarity_vectors):
            text_num = gt_text_i.max().item() + 1
            loss_agg_single_sample = []
            G_kernel_list = []  # 存储计算好的G_Ki,用于计算loss dis
            # 求解每一个文本实例的loss agg
            for text_idx in range(1, int(text_num)):
                # 计算 D_p_Ki
                single_kernel_mask = gt_kernel_i == text_idx
                if single_kernel_mask.sum() == 0 or (gt_text_i == text_idx).sum() == 0:
                    # 这个文本被crop掉了
                    continue
                # G_Ki, shape: 4
                G_kernel = similarity_vector[:, single_kernel_mask].mean(1)  # 4
                G_kernel_list.append(G_kernel)
                # 文本像素的矩阵 F(p) shape: 4* nums (num of text pixel)
                text_similarity_vector = similarity_vector[:, gt_text_i == text_idx]
                # ||F(p) - G(K_i)|| - delta_agg, shape: nums
                text_G_ki = (text_similarity_vector - G_kernel.reshape(4, 1)).norm(2, dim=0) - self.delta_agg
                # D(p,K_i), shape: nums
                D_text_kernel = torch.max(text_G_ki, torch.tensor(0, device=text_G_ki.device, dtype=torch.float)).pow(2)
                # 计算单个文本实例的loss, shape: nums
                loss_agg_single_text = torch.log(D_text_kernel + 1).mean()
                loss_agg_single_sample.append(loss_agg_single_text)
            if len(loss_agg_single_sample) > 0:
                loss_agg_single_sample = torch.stack(loss_agg_single_sample).mean()
            else:
                loss_agg_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_aggs.append(loss_agg_single_sample)

            # 求解每一个文本实例的loss dis
            loss_dis_single_sample = 0
            for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
                # delta_dis - ||G(K_i) - G(K_j)||
                kernel_ij = self.delta_dis - (G_kernel_i - G_kernel_j).norm(2)
                # D(K_i,K_j)
                D_kernel_ij = torch.max(kernel_ij, torch.tensor(0, device=kernel_ij.device, dtype=torch.float)).pow(2)
                loss_dis_single_sample += torch.log(D_kernel_ij + 1)
            if len(G_kernel_list) > 1:
                loss_dis_single_sample /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
            else:
                loss_dis_single_sample = torch.tensor(0, device=texts.device, dtype=torch.float)
            loss_diss.append(loss_dis_single_sample)
        return torch.stack(loss_aggs), torch.stack(loss_diss)

    def dice_loss(self, input, target, mask):
        input = torch.sigmoid(input)
        target[target <= 0.5] = 0
        target[target > 0.5] = 1
        input = input.contiguous().view(input.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def ohem_single(self, score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

        if pos_num == 0:
            # selected_mask = gt_text.copy() * 0 # may be not good
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num * self.ohem_ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num - 1]
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    def ohem_batch(self, scores, gt_texts, training_masks):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_masks = training_masks.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()

        return selected_masks




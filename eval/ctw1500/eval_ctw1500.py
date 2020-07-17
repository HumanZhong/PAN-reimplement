import Polygon as plg
import numpy as np
import os

gt_root = '/home/data1/IC19/CTW1500/test_ctw/text_label_curve/'
pred_root = '/home/zhm/text-detection-baseline/outputs/result_ctw_txt_poly_0417_baseline_60_93/'



def read_pred(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        bboxes = []
        for line in lines:
            if line == '':
                continue
            bbox = line.split(',')
            if len(bbox) % 2 == 1:
                print('Error in', path)
            bbox = [int(bbox[i]) for i in range(len(bbox))]
            bboxes.append(bbox)
    return bboxes

def read_gt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        bboxes = []
        for line in lines:
            if line == '':
                continue
            bbox = line.split(',')
            bbox = [int(bbox[i]) for i in range(len(bbox))]
            x1 = np.int(bbox[0])
            y1 = np.int(bbox[1])
            bbox = bbox[4:]
            bbox = np.asarray(bbox) + ([x1, y1] * 14)
            bboxes.append(bbox)
    return bboxes



def get_intersection(pred_plg, gt_plg):
    inter_plg = pred_plg & gt_plg
    if len(inter_plg) == 0:
        return 0
    return inter_plg.area()


def get_union(pred_plg, gt_plg):
    return pred_plg.area() + gt_plg.area() - get_intersection(pred_plg, gt_plg)


if __name__ == '__main__':
    threshold = 0.5
    pred_list = os.listdir(pred_root)

    tp, fp, npos = 0, 0, 0
    for pred_path in pred_list:

        preds = read_pred(pred_root + pred_path)
        gt_path = gt_root + pred_path.split('/')[-1].split('_')[-1]
        gts = read_gt(gt_path)
        npos += len(gts)

        cover = set()
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred)
            pred = pred.reshape(int(pred.shape[0] / 2), 2)
            pred_plg = plg.Polygon(pred)

            flag = False
            for gt_id, gt in enumerate(gts):
                gt = np.array(gt)
                gt = gt.reshape(int(gt.shape[0] / 2), 2)
                gt_plg = plg.Polygon(gt)

                union = get_union(pred_plg, gt_plg)
                intersection = get_intersection(pred_plg, gt_plg)

                if intersection / union >= threshold:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
            if flag == True:
                tp += 1
            else:
                fp += 1
    print('True Positives', tp)
    print('False Positives', fp)
    print('Target Positives', npos)

    recall = tp / npos
    print('Recall', recall)
    precision = tp / (tp + fp)
    print('Precision', precision)
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    print('Hmean', hmean)


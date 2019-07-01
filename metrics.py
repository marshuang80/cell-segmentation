"""refence: https://github.com/neptune-ml/open-solution-data-science-bowl-2018/blob/master/src/metrics.py
"""

import numpy as np
from scipy.ndimage.measurements import label

def get_ious(pred, gt, threshold):
    """Caculate intersection over union between predcition and ground truth

    Parameters
    ----------
        pred: 
            predictions from the model
        gt: 
            ground truth labels
        threshold:
            threshold used to seperate binary labels
    """

    gt[gt > threshold] = 1.
    gt[gt <= threshold] = 0.

    pred[pred > threshold] = 1.
    pred[pred <= threshold] = 0.

    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)

    if union == 0:
        union = 1e-09

    return intersection / union


def compute_precision(pred, gt, threshold=0.5):
    """Compute the precision of IoU

    Parameters
    ----------
        pred:
            predictions from the model
        gt:
            ground truth labels
        threshold:
            threshold used to seperate binary labels
    """

    pred[pred > threshold] = 1.
    pred[pred <= threshold] = 0.

    structure = np.ones((3,3))

    labeled, ncomponents = label(pred, structure)

    pred_masks = []
    for l in range(1,ncomponents):
        pred_mask = np.zeros(labeled.shape)
        pred_mask[labeled == l] = 1
        pred_masks.append(pred_mask)

    iou_vol = np.zeros([10, len(pred_masks), len(gt)])

    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt):
            s = get_iou_vector(p, g)
            iou_vol[:,i,j] = s

    p = []
    for iou_mat in iou_vol:
        tp = np.sum(iou_mat.sum(axis=1) > 0)
        fp = np.sum(iou_mat.sum(axis=1) == 0)
        fn = np.sum(iou_mat.sum(axis=0) == 0)
        p.append(tp / (tp + fp + fn))

    return np.mean(p)


def get_iou_vector(pred, gt):
    """Compute the IoU hits with a range of thresholds

    Parameters
    ----------
        pred: 
            predictions from the model
        gt: 
            ground truth labels
    """
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)

    intersection = np.sum(intersection)
    union = np.sum(union)

    if union == 0:
        union = 1e-09

    iou = np.sum(intersection > 0) / np.sum(union > 0)
    s = []
    for thresh in np.arange(0.5,1,0.05):
        s.append(1 if iou > thresh else 0)

    return s



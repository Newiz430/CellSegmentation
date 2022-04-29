import numpy as np
import sklearn.metrics
from sklearn import metrics
import torch


def calc_err(pred, real):
    """计算 tile mode 的错误率、假阳性率、假阴性率。"""

    pred = np.asarray(pred)
    real = np.asarray(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0] # 错误率 = 预测错误的和 / 总和
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum() # 假阳性率 = 假阳性 / 所有的阴性
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum() # 假阴性率 = 假阴性 / 所有的阳性
    return err, fpr, fnr


def calc_map(pred, real):
    return metrics.average_precision_score(np.asarray(pred), np.asarray(real))


def weighted_mse(inputs, targets, reduction='mean'):
    assert reduction in ('mean', 'sum'), "\'reduction\' must be one of (\'mean\', \'sum\'). "

    # for count = 20 or more set weight = ln(count)
    weights = targets.clone()
    for i, w in enumerate(weights):
        if torch.ge(w, 20):
            weights[i] = torch.log(w) if torch.ge(w, 20) else 1

    tmp = weights * (inputs - targets) ** 2
    return torch.mean(tmp) if reduction == 'mean' else torch.sum(tmp)


def dice_coef(batch_inputs, batch_targets, epsilon=1e-6):

    assert batch_inputs.dtype == batch_targets.dtype, "Input & target vectors should have same dtype. "
    if batch_inputs.ndim == 2 and batch_targets.ndim == 2:
        batch_inputs = torch.flatten(batch_inputs)
        batch_targets = torch.flatten(batch_targets)
        a = torch.sum(batch_inputs * batch_targets)
        b = torch.sum(batch_inputs * batch_inputs)
        c = torch.sum(batch_targets * batch_targets)
    else:
        batch_inputs = batch_inputs.contiguous().view(batch_inputs.size()[0], -1)
        batch_targets = batch_targets.contiguous().view(batch_targets.size()[0], -1).float()
        a = torch.sum(batch_inputs * batch_targets, 1)
        b = torch.sum(batch_inputs * batch_inputs, 1)
        c = torch.sum(batch_targets * batch_targets, 1)

    d = (2 * a + epsilon) / (b + c + epsilon)
    return d


def euclid_dist(p1, p2):
    return np.sqrt(sum([(d1 - d2) * (d1 - d2) for (d1, d2) in zip(p1, p2)]))


def precision_recall(tp, fp, fn, return_f1=False):
    p = 1 if tp + fp == 0 else tp / (tp + fp)
    r = 1 if tp + fn == 0 else tp / (tp + fn)
    if return_f1:
        return p, r, 0 if p + r == 0 else (2 * p * r) / (p + r)
    else:
        return p, r

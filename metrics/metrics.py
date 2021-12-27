import numpy as np
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


def dice_coef(inputs, targets, epsilon=1e-6):

    assert inputs.dtype == targets.dtype, "Input & target vectors should have same dtype. "
    inter = torch.dot(inputs[-1], targets[-1])  # 最后一维作点积
    sets_sum = torch.sum(inputs) + torch.sum(targets)
    if sets_sum.item() == 0:
        sets_sum = 2 * inter

    return (2 * inter + epsilon) / (sets_sum + epsilon)

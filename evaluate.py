import numpy as np
import torch
import torch.nn.functional as F

from metrics import calc_err, calc_map, qwk
from train import weighted_mse

def evaluate_tile(valset, probs, tiles_per_pos, threshold):
    """tile mode 的验证"""

    val_groups = np.array(valset.tileIDX)

    order = np.lexsort((probs, val_groups)) # 对 tile 按预测概率排序
    val_groups = val_groups[order]
    val_probs = probs[order]

    val_index = np.array([prob > threshold for prob in val_probs])

    # 制作分类用的 label：根据计数标签 = n，前 n * tiles_per_pos 个 tile 为阳性
    labels = np.zeros(len(val_probs))
    for i in range(1, len(val_probs) + 1):
        if i == len(val_probs) or val_groups[i] != val_groups[i - 1]:
            labels[i - valset.labels[val_groups[i - 1]] * tiles_per_pos: i] = [1] * valset.labels[val_groups[i - 1]] * tiles_per_pos

    # 计算错误率、FPR、FNR
    err, fpr, fnr = calc_err(val_index, labels)
    return err, fpr, fnr


def evaluate_image(valset, categories, counts):
    """image mode 的验证，probs 应为二维数组，第二维的大小为 7。"""

    # map = calc_map(F.one_hot(torch.tensor(categories, dtype=torch.int64), num_classes=6).numpy(),
    #                F.one_hot(torch.tensor(valset.cls_labels, dtype=torch.int64), num_classes=6).numpy())
    mse = F.mse_loss(torch.from_numpy(counts), torch.tensor(valset.labels))
    # mse = weighted_mse_loss(torch.from_numpy(counts), torch.tensor(valset.labels))
    score = qwk(counts, valset.labels)

    # return map, mse.item(), score
    return 0, mse.item(), score

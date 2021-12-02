import torch
import torch.nn as nn


class MSELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        assert reduction in ('mean', 'sum'), "\'reduction\' must be one of (\'mean\', \'sum\'). "
        self.reduction = reduction

    def forward(self, inputs, targets):
        tmp = (inputs - targets) ** 2
        return torch.mean(tmp) if self.reduction == 'mean' else torch.sum(tmp)


class WeightedMSELoss(nn.Module):

    def __init__(self, reduction='mean'):
        """
        A weighted version of MSE. Weights are the classification labels from the image_cls branch.
        Set $ln(\text{count}), \text{count} \ge 20$ as a single weight factor.
        """
        super(WeightedMSELoss, self).__init__()
        assert reduction in ('mean', 'sum'), "\'reduction\' must be one of (\'mean\', \'sum\'). "
        self.reduction = reduction

    def forward(self, inputs, targets):
        return weighted_mse_loss(inputs, targets, reduction=self.reduction)


def weighted_mse_loss(inputs, targets, reduction='mean'):
    assert reduction in ('mean', 'sum'), "\'reduction\' must be one of (\'mean\', \'sum\'). "

    # for count = 20 or more set weight = ln(count)
    weights = targets.clone()
    for i, w in enumerate(weights):
        if torch.ge(w, 20):
            weights[i] = torch.log(w) if torch.ge(w, 20) else 1

    tmp = weights * (inputs - targets) ** 2
    return torch.mean(tmp) if reduction == 'mean' else torch.sum(tmp)


class DiceLoss(nn.Module):
    pass
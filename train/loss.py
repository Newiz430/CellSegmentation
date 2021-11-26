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
        Set $exp(w) / sum(w)$ as a single weight factor.
        """
        super(WeightedMSELoss, self).__init__()
        assert reduction in ('mean', 'sum'), "\'reduction\' must be one of (\'mean\', \'sum\'). "
        self.reduction = reduction

    def forward(self, inputs, targets, weights):
        return weighted_mse_loss(inputs, targets, weights=weights, reduction=self.reduction)


def weighted_mse_loss(inputs, targets, weights, reduction='mean'):
    assert reduction in ('mean', 'sum'), "\'reduction\' must be one of (\'mean\', \'sum\'). "

    # 2 ** (w - 1) to map categories-[0, 1, 2, 3, 4, 5, 6] to weights-[.5, 1, 2, 4, 8, 16, 32]
    tmp = 2 ** (weights - 1) * (inputs - targets) ** 2
    return torch.mean(tmp) if reduction == 'mean' else torch.sum(tmp)

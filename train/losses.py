import torch
import torch.nn as nn
from metrics import weighted_mse, dice_coef

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
        return weighted_mse(inputs, targets, reduction=self.reduction)


# def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, epsilon=1e-6):
#     # you need to extract dim [1] cuz input is like [n, 2, 300, 300] and target like [n, 300, 300]
#     # dice_coef takes [300, 300] as input
#     if inputs.ndim == 4:
#         inputs = inputs[:, 0]
#     dice = 0
#     for i in range(inputs.size(0)):
#         # dice += dice_coef(inputs[i, ...], targets[i, ...], self.epsilon)
#         dice += dice_coef(inputs[i], targets[i], epsilon)
#     return dice


class DiceLoss(nn.Module):

    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # you need to extract dim [1] cuz input is like [n, 2, 300, 300] and target like [n, 300, 300]
        # dice_coef takes [300, 300] as input
        if inputs.ndim == 4:
            inputs = inputs[:, 1]

        dice = 0
        for i in range(inputs.size(0)):
            # dice += dice_coef(inputs[i, ...], targets[i, ...], self.epsilon)
            dice += dice_coef(inputs[i], targets[i], self.epsilon)
        return 1 - dice

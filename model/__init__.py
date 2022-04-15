from .resnet import MILresnet18, MILresnet34, MILresnet50
from .efficientnet import MILefficientnetB0, MILefficientnetB2
from .resnext import MILresnext50_32x4d, MILresnext101_32x8d

nets = {
    'resnet18': MILresnet18(pretrained=True),
    'resnet34': MILresnet34(pretrained=True),
    'resnet50': MILresnet50(pretrained=True),
    'efficientnet_b0': MILefficientnetB0(pretrained=True),
    'efficientnet_b2': MILefficientnetB2(pretrained=True),
    'resnext50': MILresnext50_32x4d(pretrained=True),
    'resnext101': MILresnext101_32x8d(pretrained=True),
}

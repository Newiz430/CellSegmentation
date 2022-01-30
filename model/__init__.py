from .resnet import MILresnet18, MILresnet34, MILresnet50
from .efficientnet import MILefficientnetB0, MILefficientnetB2

nets = {
    'resnet18': MILresnet18(pretrained=True),
    'resnet34': MILresnet34(pretrained=True),
    'resnet50': MILresnet50(pretrained=True),
    'efficientnet_b0': MILefficientnetB0(pretrained=True),
    'efficientnet_b2': MILefficientnetB2(pretrained=True),
}

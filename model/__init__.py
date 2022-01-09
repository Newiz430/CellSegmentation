from .milnet import MILresnet, MILefficientnet

nets = {
    'resnet18': MILresnet('resnet18', pretrained=True),
    'resnet34': MILresnet('resnet34', pretrained=True),
    'resnet50': MILresnet('resnet50', pretrained=True),
    'efficientnet_b0': MILefficientnet('efficientnet_b0', pretrained=True),
    'efficientnet_b2': MILefficientnet('efficientnet_b2', pretrained=True),
}

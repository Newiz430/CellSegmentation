from .mil_resnet import MILresnet18, MILresnet34, MILresnet50

encoders = {
    'resnet18': MILresnet18(pretrained=True),
    'resnet34': MILresnet34(pretrained=True),
    'resnet50': MILresnet50(pretrained=True)
}
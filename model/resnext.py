import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}
__all__ = ['MILresnext50_32x4d', 'MILresnext101_32x8d']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MILResNeXt(nn.Module):

    def __init__(self, encoder, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        self.encoder_name = encoder
        self.mode = None

        self.encoder_prefix = (
            "conv1",
            "bn1",
            "relu",
            "layer1",
            "layer2",
            "layer3",
            "layer4"
        )
        self.image_module_prefix = (
            "fc_image_cls",
            "fc_image_reg"
        )
        self.tile_module_prefix = (
            "fc_tile",
        )
        self.seg_module_prefix = (
            "upconv",
            "seg_out_conv"
        )
        super(MILResNeXt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # encoder 以下部分
        def init_tile_modules():
            self.avgpool_tile = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool_tile = nn.AdaptiveMaxPool2d((1, 1))
            self.fc_tile = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * block.expansion, num_classes)
            )

        def init_image_modules(map_size):
            self.avgpool_image = nn.AdaptiveAvgPool2d((map_size, map_size))
            self.maxpool_image = nn.AdaptiveMaxPool2d((map_size, map_size))
            self.fc_image_cls = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(512 * map_size * map_size * block.expansion),
                nn.Dropout(p=0.25),
                nn.ReLU(inplace=True),
                nn.Linear(512 * map_size * map_size * block.expansion, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(),
                nn.Linear(64, 7)
            )
            self.fc_image_reg = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(512 * map_size * map_size * block.expansion),
                nn.Dropout(p=0.25),
                nn.ReLU(inplace=True),
                nn.Linear(512 * map_size * map_size * block.expansion, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(),
                nn.Linear(64, 1),
                nn.ReLU(inplace=True)
            )

        def init_seg_modules():
            # 图像上采样卷积层
            self.upconv1 = self.upsample_conv(512, 256)
            self.upconv2 = self.upsample_conv(512, 256)
            self.upconv3 = self.upsample_conv(256, 128)
            self.upconv4 = self.upsample_conv(256, 128)
            self.upconv5 = self.upsample_conv(128, 64)
            self.upconv6 = self.upsample_conv(128, 64)
            self.upconv7 = self.upsample_conv(64, 64)
            self.upconv8 = self.upsample_conv(64, 64)
            self.seg_out_conv = nn.Conv2d(64, 2, kernel_size=1)

        init_tile_modules()
        init_image_modules(map_size=1)
        init_seg_modules()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    @staticmethod
    def upsample_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def set_encoder_grads(self, requires_grad):

        self.conv1.requires_grad_(requires_grad)
        self.bn1.requires_grad_(requires_grad)
        self.relu.requires_grad_(requires_grad)
        self.maxpool.requires_grad_(requires_grad)
        self.layer1.requires_grad_(requires_grad)
        self.layer2.requires_grad_(requires_grad)
        self.layer3.requires_grad_(requires_grad)
        self.layer4.requires_grad_(requires_grad)

    def set_tile_module_grads(self, requires_grad):

        self.avgpool_tile.requires_grad_(requires_grad)
        self.maxpool_tile.requires_grad_(requires_grad)
        self.fc_tile.requires_grad_(requires_grad)

    def set_image_module_grads(self, requires_grad):

        self.avgpool_image.requires_grad_(requires_grad)
        self.maxpool_image.requires_grad_(requires_grad)
        self.fc_image_cls.requires_grad_(requires_grad)
        self.fc_image_reg.requires_grad_(requires_grad)

    def set_seg_module_grads(self, requires_grad):

        self.upconv1.requires_grad_(requires_grad)
        self.upconv2.requires_grad_(requires_grad)
        self.upconv3.requires_grad_(requires_grad)
        self.upconv4.requires_grad_(requires_grad)
        self.seg_out_conv.requires_grad_(requires_grad)

    def resnext_forward(self, x: torch.Tensor, return_intermediate: bool = False):

        x = self.conv1(x)       # x_tile: [nk,  64, 16, 16] x_image: [n,   64, 150, 150]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # x_tile: [nk,  64,  8,  8] x_image: [n,   64,  75,  75]
        x1 = self.layer1(x)     # x_tile: [nk,  64,  8,  8] x_image: [n,  256,  75,  75]
        x2 = self.layer2(x1)    # x_tile: [nk, 128,  4,  4] x_image: [n,  512,  38,  38]
        x3 = self.layer3(x2)    # x_tile: [nk, 256,  2,  2] x_image: [n, 1024,  19,  19]
        x4 = self.layer4(x3)    # x_tile: [nk, 512,  1,  1] x_image: [n, 2048,  10,  10]

        if return_intermediate:
            return x4, x3, x2, x1
        else:
            return x4

    def forward(self, x: torch.Tensor, freeze_bn=False):  # x_tile: [nk, 3, 32, 32] x_image: [n, 3, 299, 299]

        # Set freeze_bn=True in tile training mode to freeze E(x) & Var(x) in BatchNorm2D(x).
        # Otherwise, assessment results will decay as tile training propels.
        if self.mode == "tile" and freeze_bn:
            # TODO: iterate named_modules() and use bn.eval()
            self.eval()
            x4 = self.resnext_forward(x, False)
            self.train()
        elif self.mode == "segment":
            x4, x3, x2, x1 = self.resnext_forward(x, True)
        else:
            x4 = self.resnext_forward(x, False)

        if self.mode == "tile":

            x = self.avgpool_tile(x4) + self.maxpool_tile(x4) # x: [nk, 512, 1, 1]
            x = self.fc_tile(x)  # x: [nk, 512]

            return x

        elif self.mode == "image":

            # image_cls & image_reg
            out = self.avgpool_image(x4) + self.maxpool_image(x4)  # [n, 2048, ?, ?]
            out_cls = self.fc_image_cls(out)  # [n, 7]
            out_reg = self.fc_image_reg(out)  # [n, 1]

            return out_cls, out_reg

        elif self.mode == "segment":

            out_seg = F.interpolate(x4.clone(), size=19, mode="bilinear", align_corners=True)   # out_seg: [n, 2048, 19, 19]
            out_seg = self.upconv1(out_seg)                                                     # [n, 1024, 19, 19]
            out_seg = torch.cat([out_seg, x3], dim=1)                                           # 连接两层，输出 [n, 2048, 19, 19]
            out_seg = self.upconv2(out_seg)                                                     # [n, 1024, 19, 19]

            out_seg = F.interpolate(out_seg, size=38, mode="bilinear", align_corners=True)      # [n, 1024, 38, 38]
            out_seg = self.upconv3(out_seg)                                                     # [n, 512, 38, 38]
            out_seg = torch.cat([out_seg, x2], dim=1)                                           # 连接两层，输出 [n, 1024, 38, 38]
            out_seg = self.upconv4(out_seg)                                                     # [n, 512, 38, 38]

            out_seg = F.interpolate(out_seg, size=75, mode="bilinear", align_corners=True)      # [n, 512, 75, 75]
            out_seg = self.upconv5(out_seg)                                                     # [n, 256, 75, 75]
            out_seg = torch.cat([out_seg, x1], dim=1)                                           # 连接两层，输出 [n, 512, 75, 75]
            out_seg = self.upconv6(out_seg)                                                     # [n, 256, 75, 75]

            out_seg = F.interpolate(out_seg, size=150, mode="bilinear", align_corners=True)     # [n, 256, 150, 150]
            out_seg = self.upconv7(out_seg)                                                     # [n, 128, 150, 150]
            out_seg = self.upconv8(out_seg)                                                     # [n, 64, 150, 150]
            out_seg = F.interpolate(out_seg, size=299, mode="bilinear", align_corners=True)     # [n, 64, 299, 299]
            out_seg = self.seg_out_conv(out_seg)                                                # [n, 1, 299, 299]

            return out_seg

        else:
            raise Exception("Something wrong in setmode.")

    def setmode(self, mode):
        """
        mode "image":   pt.1 (whole image mode), pooled feature -> image classification & regression
        mode "tile":    pt.2 (instance mode), pooled feature -> tile classification
        mode "segment": pt.3 (segmentation mode), pooled feature -> expanding path -> output map
        """

        if mode == "tile":
            self.set_encoder_grads(False)
            self.set_tile_module_grads(True)
            self.set_image_module_grads(False)
            self.set_seg_module_grads(False)
        elif mode == "image":
            self.set_encoder_grads(True)
            self.set_tile_module_grads(False)
            self.set_image_module_grads(True)
            self.set_seg_module_grads(False)
        elif mode == "segment":
            self.set_encoder_grads(False)
            self.set_tile_module_grads(False)
            self.set_image_module_grads(False)
            self.set_seg_module_grads(True)
        else:
            raise Exception("Invalid mode: {}.".format(mode))

        self.mode = mode


def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = MILResNeXt(arch, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
        # change num of cell classes from 1000 to 2 here to make it compatible with pretrained files
        model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    return model


def MILresnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnext('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                    pretrained, progress, **kwargs)

def MILresnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                    pretrained, progress, **kwargs)

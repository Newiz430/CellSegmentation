import copy
import math
import torch

from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Sequence

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from torchvision.ops import StochasticDepth

model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}

__all__ = ["MILefficientnetB0", "MILefficientnetB2"]


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self,
                 expand_ratio: float, kernel: int, stride: int,
                 input_channels: int, out_channels: int, num_layers: int,
                 width_mult: float, depth_mult: float) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'expand_ratio={expand_ratio}'
        s += ', kernel={kernel}'
        s += ', stride={stride}'
        s += ', input_channels={input_channels}'
        s += ', out_channels={out_channels}'
        s += ', num_layers={num_layers}'
        s += ')'
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, expanded_channels, kernel_size=1,
                                             norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        layers.append(ConvNormActivation(expanded_channels, expanded_channels, kernel_size=cnf.kernel,
                                         stride=cnf.stride, groups=expanded_channels,
                                         norm_layer=norm_layer, activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(ConvNormActivation(expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                         activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class MILEfficientNet(nn.Module):
    def __init__(
            self,
            arch: str,
            inverted_residual_setting: List[MBConvConfig],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        self.encoder_name = arch
        self.encoder_prefix = (
            "features",
            # "avgpool",
            # "classifier"
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
        # encoder
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        self.firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3, self.firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                         activation_layer=nn.SiLU))

        # building inverted residual blocks
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        self.lastconv_input_channels = inverted_residual_setting[-1].out_channels
        self.lastconv_output_channels = 4 * self.lastconv_input_channels
        layers.append(ConvNormActivation(self.lastconv_input_channels, self.lastconv_output_channels, kernel_size=1,
                                         norm_layer=norm_layer, activation_layer=nn.SiLU))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(self.lastconv_output_channels, num_classes),
        )

        # heads beneath the encoder
        def init_tile_modules():
            self.avgpool_tile = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool_tile = nn.AdaptiveMaxPool2d((1, 1))
            self.fc_tile = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.lastconv_output_channels, num_classes)
            )

        def init_image_modules(map_size):
            self.avgpool_image = nn.AdaptiveAvgPool2d((map_size, map_size))
            self.maxpool_image = nn.AdaptiveMaxPool2d((map_size, map_size))
            self.fc_image_cls = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.3),
                nn.Linear(self.lastconv_output_channels, 7)
            )
            self.fc_image_reg = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.3),
                nn.Linear(self.lastconv_output_channels, 1),
                nn.ReLU(inplace=True)
            )

        def init_seg_modules():
            # upsample convolution layers
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
        # init_seg_modules()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_encoder_grads(self, requires_grad):

        self.features.requires_grad_(requires_grad)

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

    def efficientnet_forward(self, x: Tensor, return_intermediate: bool = False) -> Tensor:
        x = self.features(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return x

    def forward(self, x: torch.Tensor, freeze_bn=False):  # x_tile: [nk, 3, 32, 32] x_image: [n, 3, 299, 299]
        # Set freeze_bn=True in tile training mode to freeze E(x) & Var(x) in BatchNorm2D(x).
        # Otherwise, assessment results will decay as tile training propels.
        if self.mode == "tile" and freeze_bn:
            # TODO: iterate named_modules() and use bn.eval()
            # self.eval()
            x4 = self.efficientnet_forward(x, False)
            # self.train()
        elif self.mode == "segment":
            x4, x3, x2, x1 = self.efficientnet_forward(x, True)
        else:
            x4 = self.efficientnet_forward(x, False)

        if self.mode == "tile":

            x = self.avgpool_tile(x4) + self.maxpool_tile(x4)  # x: [nk, 512, 1, 1]
            x = self.fc_tile(x)  # x: [nk, 512]

            return x

        elif self.mode == "image":

            # image_cls & image_reg
            out = self.avgpool_image(x4) + self.maxpool_image(x4)  # [n, 2048, ?, ?]
            out_cls = self.fc_image_cls(out)  # [n, 7]
            out_reg = self.fc_image_reg(out)  # [n, 1]

            return out_cls, out_reg

        elif self.mode == "segment":

            pass
            # out_seg = F.interpolate(x4.clone(), size=19, mode="bilinear",
            #                         align_corners=True)  # out_seg: [n, 2048, 19, 19]
            # out_seg = self.upconv1(out_seg)  # [n, 1024, 19, 19]
            # out_seg = torch.cat([out_seg, x3], dim=1)  # concat: [n, 2048, 19, 19]
            # out_seg = self.upconv2(out_seg)  # [n, 1024, 19, 19]
            #
            # out_seg = F.interpolate(out_seg, size=38, mode="bilinear", align_corners=True)  # [n, 1024, 38, 38]
            # out_seg = self.upconv3(out_seg)  # [n, 512, 38, 38]
            # out_seg = torch.cat([out_seg, x2], dim=1)  # concat: [n, 1024, 38, 38]
            # out_seg = self.upconv4(out_seg)  # [n, 512, 38, 38]
            #
            # out_seg = F.interpolate(out_seg, size=75, mode="bilinear", align_corners=True)  # [n, 512, 75, 75]
            # out_seg = self.upconv5(out_seg)  # [n, 256, 75, 75]
            # out_seg = torch.cat([out_seg, x1], dim=1)  # concat: [n, 512, 75, 75]
            # out_seg = self.upconv6(out_seg)  # [n, 256, 75, 75]
            #
            # out_seg = F.interpolate(out_seg, size=150, mode="bilinear", align_corners=True)  # [n, 256, 150, 150]
            # out_seg = self.upconv7(out_seg)  # [n, 128, 150, 150]
            # out_seg = self.upconv8(out_seg)  # [n, 64, 150, 150]
            # out_seg = F.interpolate(out_seg, size=299, mode="bilinear", align_corners=True)  # [n, 64, 299, 299]
            # out_seg = self.seg_out_conv(out_seg)  # [n, 1, 299, 299]
            #
            # return out_seg

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
            # self.set_seg_module_grads(False)
        elif mode == "image":
            self.set_encoder_grads(True)
            self.set_tile_module_grads(False)
            self.set_image_module_grads(True)
            # self.set_seg_module_grads(False)
        elif mode == "segment":
            self.set_encoder_grads(False)
            self.set_tile_module_grads(False)
            self.set_image_module_grads(False)
            # self.set_seg_module_grads(True)
        else:
            raise Exception("Invalid mode: {}.".format(mode))

        self.mode = mode


def _efficientnet_conf(width_mult: float, depth_mult: float, **kwargs: Any) -> List[MBConvConfig]:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    return inverted_residual_setting


def _mileff_model(arch: str, inverted_residual_setting: List[MBConvConfig], dropout: float,
                        pretrained: bool, progress: bool, **kwargs: Any) -> MILEfficientNet:
    model = MILEfficientNet(arch, inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def MILefficientnetB0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MILEfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    return _mileff_model("efficientnet_b0", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)


def MILefficientnetB2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MILEfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.1, depth_mult=1.2, **kwargs)
    return _mileff_model("efficientnet_b2", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)

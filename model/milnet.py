import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from model.resnet import resnet18, resnet34, resnet50
from model.efficientnet import efficientnet_b0, efficientnet_b2

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

class MILNet(nn.Module):

    def __init__(self, encoder, num_classes=1000, **kwargs):

        super(MILNet, self).__init__()

        self.mode = None
        self.encoder = encoder

        self.encoder_prefix = self.encoder.module_prefix
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

        # encoder 以下部分
        def init_tile_modules():
            self.avgpool_tile = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool_tile = nn.AdaptiveMaxPool2d((1, 1))
            if self.encoder.encoder_name.startswith("efficientnet"):
                self.fc_tile = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.encoder.lastconv_output_channels, num_classes)
                )
            else:
                self.fc_tile = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * self.encoder.block.expansion, num_classes)
                )

        def init_image_modules(map_size):
            self.avgpool_image = nn.AdaptiveAvgPool2d((map_size, map_size))
            self.maxpool_image = nn.AdaptiveMaxPool2d((map_size, map_size))
            if self.encoder.encoder_name.startswith("efficientnet"):
                self.fc_image_cls = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.3),
                    nn.Linear(self.encoder.lastconv_output_channels, 7)
                )
                self.fc_image_reg = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.3),
                    nn.Linear(self.encoder.lastconv_output_channels, 1),
                    nn.ReLU(inplace=True)
                )
            else:
                self.fc_image_cls = nn.Sequential(
                    nn.Flatten(),
                    nn.BatchNorm1d(512 * map_size * map_size * self.encoder.block.expansion),
                    nn.Dropout(p=0.25),
                    nn.ReLU(inplace=True),
                    nn.Linear(512 * map_size * map_size * self.encoder.block.expansion, 64),
                    nn.BatchNorm1d(64),
                    nn.Dropout(),
                    nn.Linear(64, 7)
                )
                self.fc_image_reg = nn.Sequential(
                    nn.Flatten(),
                    nn.BatchNorm1d(512 * map_size * map_size * self.encoder.block.expansion),
                    nn.Dropout(p=0.25),
                    nn.ReLU(inplace=True),
                    nn.Linear(512 * map_size * map_size * self.encoder.block.expansion, 64),
                    nn.BatchNorm1d(64),
                    nn.Dropout(),
                    nn.Linear(64, 1),
                    nn.ReLU(inplace=True)
                )

        def init_seg_modules():
            # 图像上采样卷积层
            self.upconv1 = self.upsample_conv(512 * self.encoder.block.expansion, 256 * self.encoder.block.expansion)
            self.upconv2 = self.upsample_conv(512 * self.encoder.block.expansion, 256 * self.encoder.block.expansion)
            self.upconv3 = self.upsample_conv(256 * self.encoder.block.expansion, 128 * self.encoder.block.expansion)
            self.upconv4 = self.upsample_conv(256 * self.encoder.block.expansion, 128 * self.encoder.block.expansion)
            self.upconv5 = self.upsample_conv(128 * self.encoder.block.expansion, 64 * self.encoder.block.expansion)
            self.upconv6 = self.upsample_conv(128 * self.encoder.block.expansion, 64 * self.encoder.block.expansion)
            self.upconv7 = self.upsample_conv(64 * self.encoder.block.expansion,
                                              64 if self.encoder.block.expansion == 1 else 32 * self.encoder.block.expansion)
            self.upconv8 = self.upsample_conv(64 if self.encoder.block.expansion == 1 else 32 * self.encoder.block.expansion, 64)
            self.seg_out_conv = nn.Conv2d(64, 2, kernel_size=1)

        init_tile_modules()
        init_image_modules(map_size=1)
        # init_seg_modules()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def upsample_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def set_encoder_grads(self, requires_grad):

        self.encoder.requires_grad_(requires_grad)

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

    def forward(self, x: torch.Tensor, freeze_bn=False):  # x_tile: [nk, 3, 32, 32] x_image: [n, 3, 299, 299]

        # Set freeze_bn=True in tile training mode to freeze E(x) & Var(x) in BatchNorm2D(x).
        # Otherwise, assessment results will decay as tile training propels.
        if self.mode == "tile" and freeze_bn:
            self.eval()
            x4 = self.encoder(x, False)
            self.train()
        elif self.mode == "segment":
            x4, x3, x2, x1 = self.encoder(x, True)
        else:
            x4 = self.encoder(x, False)

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
            # self.set_seg_module_grads(False)
        elif mode == "segment":
            self.set_encoder_grads(False)
            self.set_tile_module_grads(False)
            self.set_image_module_grads(False)
            self.set_seg_module_grads(True)
        else:
            raise Exception("Invalid mode: {}.".format(mode))

        self.mode = mode


def MILresnet(encoder_name, pretrained=False, **kwargs):

    if encoder_name == 'resnet18':
        encoder = resnet18(pretrained, **kwargs)
    elif encoder_name == 'resnet34':
        encoder = resnet34(pretrained, **kwargs)
    else:
        encoder = resnet50(pretrained, **kwargs)
    model = MILNet(encoder, num_classes=2)
    # change num of cell classes from 1000 to 2 here to make it compatible with pretrained files
    model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    return model


def MILefficientnet(encoder_name, pretrained=False, **kwargs):

    if encoder_name == 'efficientnet_b2':
        encoder = efficientnet_b2(pretrained, **kwargs)
    else:
        encoder = efficientnet_b0(pretrained, **kwargs)
    model = MILNet(encoder, num_classes=2)
    model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    return model

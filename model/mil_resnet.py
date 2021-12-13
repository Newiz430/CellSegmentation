import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MILResNet(nn.Module):

    def __init__(self, encoder, block, layers, num_classes=1000, expansion=1):
        self.encoder_name = encoder

        self.inplanes = 64
        super(MILResNet, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # encoder 以下部分
        self.avgpool_tile = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_tile = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_tile = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )
        self.map_size = 1  # 1, 5?
        self.avgpool_image = nn.AdaptiveAvgPool2d((self.map_size, self.map_size))
        self.maxpool_image = nn.AdaptiveMaxPool2d((self.map_size, self.map_size))
        # self.fc_image_cls = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512 * self.map_size * self.map_size * block.expansion, 7)
        # )
        self.fc_image_cls = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512 * self.map_size * self.map_size * block.expansion),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True),
            nn.Linear(512 * self.map_size * self.map_size * block.expansion, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.Linear(64, 7)
        )
        # 回归层参考了 AlexNet 的结构（弃用）
        self.fc_image_reg_old1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512 * self.map_size * self.map_size * block.expansion, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.ReLU(inplace=True)
        )
        self.fc_image_reg_old2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * self.map_size * self.map_size * block.expansion, 1),
            nn.ReLU(inplace=True)
        )
        self.fc_image_reg = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512 * self.map_size * self.map_size * block.expansion),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True),
            nn.Linear(512 * self.map_size * self.map_size * block.expansion, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.ReLU(inplace=True)
        )

        self.image_channels = 32  # image mode 中金字塔卷积的输出通道数

        # # 金字塔层级
        # self.pyramid_10 = nn.Sequential(
        #     nn.Conv2d(512 * expansion, 128, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, self.image_channels, kernel_size=1, stride=1)
        # )
        # self.pyramid_19 = nn.Sequential(
        #     nn.Conv2d(256 * expansion, 128, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, self.image_channels, kernel_size=1, stride=1)
        # )
        # self.pyramid_38 = nn.Sequential(
        #     nn.Conv2d(128 * expansion, 128, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, self.image_channels, kernel_size=1, stride=1)
        # )

        # 图像上采样卷积层
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # self.upsample_conv1 = self.upsample_conv(2 * self.image_channels, self.image_channels)
        # self.upsample_conv2 = self.upsample_conv(2 * self.image_channels, self.image_channels)
        self.upconv1 = self.upsample_conv(512 * expansion, 128 * expansion)
        self.upconv2 = self.upsample_conv(256 * expansion, 64 * expansion)
        self.upconv3 = self.upsample_conv(128 * expansion, 32 * expansion)
        self.upconv4 = self.upsample_conv(64 * expansion, 64, mid_channels=64 if expansion == 1 else 32 * expansion)

        self.seg_out_conv = nn.Conv2d(self.image_channels, 2, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def upsample_conv(in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def set_resnet_module_grads(self, requires_grad):

        self.conv1.requires_grad_(requires_grad)
        self.bn1.requires_grad_(requires_grad)
        self.relu.requires_grad_(requires_grad)
        self.maxpool.requires_grad_(requires_grad)
        self.layer1.requires_grad_(requires_grad)
        self.layer2.requires_grad_(requires_grad)
        self.layer3.requires_grad_(requires_grad)
        self.layer4.requires_grad_(requires_grad)

    def resnet_forward(self, x: torch.Tensor):

        x = self.conv1(x)    # x_tile: [nk, 64, 16, 16] x_image: [n, 64, 150, 150]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # x_tile: [nk, 64, 8, 8] x_image: [n, 64, 75, 75]
        x1 = self.layer1(x)  # x_tile: [nk, 64, 8, 8] x_image: [n, 256, 75, 75]
        x2 = self.layer2(x1) # x_tile: [nk, 128, 4, 4] x_image: [n, 512, 38, 38]
        x3 = self.layer3(x2) # x_tile: [nk, 256, 2, 2] x_image: [n, 1024, 19, 19]
        x4 = self.layer4(x3) # x_tile: [nk, 512, 1, 1] x_image: [n, 2048, 10, 10]

        return x4, x3, x2, x1

    def forward(self, x: torch.Tensor):  # x_tile: [nk, 3, 32, 32] x_image: [n, 3, 299, 299]

        x4, x3, x2, x1 = self.resnet_forward(x)

        if self.mode in "tile":

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

            # image_seg
            # out_x4 = self.pyramid_10(x4)  # out_x4: [n, 32, 10, 10]
            # out_x3 = self.pyramid_19(x3)  # out_x3: [n, 32, 19, 19]
            # out_x2 = self.pyramid_38(x2)  # out_x2: [n, 32, 38, 38]

            out_seg = F.interpolate(x4.clone(), size=19, mode="bilinear", align_corners=True)  # out_seg: [n, 2048, 19, 19]
            # out_seg = self.upsample1(x4.clone())
            out_seg = self.upconv1(out_seg)  # out_seg: [n, 512, 19, 19]
            out_seg = torch.cat([out_seg, x3], dim=1)  # 连接两层，输出 [n, 1024, 19, 19]

            out_seg = F.interpolate(out_seg, size=38, mode="bilinear", align_corners=True)  # [n, 1024, 38, 38]
            # out_seg = self.upsample2(out_seg)
            out_seg = self.upconv2(out_seg)  # [n, 256, 38, 38]
            out_seg = torch.cat([out_seg, x2], dim=1)  # 连接两层，输出 [n, 512, 38, 38]

            out_seg = F.interpolate(out_seg, size=75, mode="bilinear", align_corners=True)  # [n, 512, 75, 75]
            # out_seg = self.upsample3(out_seg)
            out_seg = self.upconv3(out_seg)  # [n, 128, 75, 75]
            out_seg = torch.cat([out_seg, x1], dim=1)  # 连接两层，输出 [n, 256, 75, 75]

            out_seg = F.interpolate(out_seg, size=150, mode="bilinear", align_corners=True) # [n, 256, 150, 150]
            out_seg = self.upconv4(out_seg)  # [n, 64, 150, 150]
            out_seg = F.interpolate(out_seg, size=299, mode="bilinear", align_corners=True)  # [n, 64, 299, 299]
            out_seg = self.seg_out_conv(out_seg)  # [n, 1, 299, 299]

            return out_seg

        else:
            raise Exception("Something wrong in setmode.")

    def setmode(self, mode):
        """
        mode "image":   pt.1 (whole image mode), pooled feature -> image classification & regression
        mode "tile":    pt.2 (instance mode), pooled feature -> tile classification
        mode "segment": pt.3 (segmentation mode), pooled feature -> expanding path -> output map
        """
        self.mode = mode


def MILresnet18(pretrained=False, **kwargs):

    model = MILResNet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def MILresnet34(pretrained=False, **kwargs):

    model = MILResNet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def MILresnet50(pretrained=False, **kwargs):

    model = MILResNet('resnet50', Bottleneck, [3, 4, 6, 3], expansion=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

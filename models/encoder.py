from models.backbones.resnet_backbone import ResNetBackbone
from models.backbones.hrnet_backbone import HRNetBackbone
# from models.backbones.deeplabv3_plus_backbone import DeepLab_Backbone
from utils.helpers import initialize_weights
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os

pretrained_url = {
    "resnet50": "models/backbones/pretrained/3x3resnet50-imagenet.pth",
    "resnet101": "models/backbones/pretrained/3x3resnet101-imagenet.pth",
    "hrnet": "models/backbones/pretrained/hrnetv2_w48_imagenet_pretrained.pth"
}

class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes):
        super(_PSPModule, self).__init__()

        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), out_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=False) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class _ASPPModule(nn.Module):
    def __init__(self, in_channels=2048):
        super(_ASPPModule, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 512, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(512)


    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))

        return out




class Encoder_ResNet50(nn.Module):
    def __init__(self, pretrained):
        super(Encoder_ResNet50, self).__init__()

        if pretrained and not os.path.isfile(pretrained_url["resnet50"]):
            print("Downloading pretrained resnet (source : https://github.com/donnyyou/torchcv)")
            os.system('sh models/backbones/get_resnet50_pretrained_model.sh')

        model = ResNetBackbone(backbone='deepbase_resnet50_dilated8', pretrained=pretrained)
        self.base = nn.Sequential(
            nn.Sequential(model.prefix, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.decoder = 'ppm'

        if self.decoder == 'ppm':
            self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])
        if self.decoder == 'aspp':
            self.psp = _ASPPModule(2048)

    def forward(self, A, B):
        a = self.base(A)
        b = self.base(B)
        diff = torch.abs(a-b)
        x = self.psp(diff)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()


class Encoder_ResNet101(nn.Module):
    def __init__(self, pretrained):
        super(Encoder_ResNet101, self).__init__()

        model = ResNetBackbone(backbone='resnet101_dilated8', pretrained=pretrained)
        self.base = nn.Sequential(
            nn.Sequential(model.prefix, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.psp = _PSPModule(2048, bin_sizes=[1, 2, 3, 6])

    def forward(self, A, B):
        a = self.base(A)
        b = self.base(B)
        diff = torch.abs(a-b)
        x = self.psp(diff)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()
        

class Encoder_HRNet(nn.Module):
    def __init__(self, pretrained_path=None):
        super(Encoder_HRNet, self).__init__()
        if pretrained_path and not os.path.isfile(pretrained_url["hrnet"]):
            print("Downloading pretrained hrnet")
            os.system('sh models/backbones/get_hrnet_pretrained_model.sh')

        self.base = HRNetBackbone(pretrained_path=pretrained_path)
        self.psp = _PSPModule(720, bin_sizes=[1, 2, 3, 6])

    def forward(self, A, B):
        a = self.base(A)
        b = self.base(B)
        diff = torch.abs(a-b)
        x = self.psp(diff)
        return x

    def get_backbone_params(self):
        return self.base.parameters()

    def get_module_params(self):
        return self.psp.parameters()



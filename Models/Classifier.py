import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def basicConv(in_ch, out_ch):
    x = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
         nn.BatchNorm2d(out_ch),
         nn.ReLU(inplace=True)]
    return x
# Unet: Outmost Down
class UnetInDown(nn.Module):
    def __init__(self, out_ch, in_ch=2):
        super(UnetInDown, self).__init__()
        self.net = nn.Sequential(*basicConv(in_ch, out_ch))
    def forward(self, x):
        x = self.net(x)
        return x
# Unet: Down
class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetDown, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            *basicConv(in_ch, out_ch))
    def forward(self, x):
        x = self.net(x)
        return x
# Unet:Up
class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetUp, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d = nn.Sequential(*basicConv(in_ch, out_ch))
    def forward(self, down, res):
        down = self.upsample(down)

        # Padding, if needed
        diffY = res.shape[2] - down.shape[2]
        diffX = res.shape[3] - down.shape[3]
        down = F.pad(down, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([down, res], dim=1)
        x = self.conv2d(x)
        return x
class UnetOutUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetOutUp, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    def forward(self, down, res):
        down = self.upsample(down)

        # Padding, if needed
        diffY = res.shape[2] - down.shape[2]
        diffX = res.shape[3] - down.shape[3]
        down = F.pad(down, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([down, res], dim=1)
        x = self.conv2d(x)
        return torch.sigmoid(x)

# AudioNet
class Classifier(nn.Module):
    def __init__(self, in_ch=3, out_class = 98, image_size = 64, regression = False, vgg = None):
        """
        [input channel, output channel, scale to width and height]
        [2, 32, 1/1], [32, 64, 1/2], [64, 128, 1/4], [128, 256, 1/8], [256, 256, 1/16], [256, 256, 1/32]
        """
        super(Classifier, self).__init__()
        self.regression = regression
        self.image_size = image_size
        self.down1 = UnetInDown(in_ch=in_ch, out_ch=32)
        self.down2 = UnetDown(in_ch=32, out_ch=64)
        final_channels = 64
        if self.image_size >= 64:
            self.down3 = UnetDown(in_ch=64, out_ch=128)
            final_channels = 128
        if self.image_size >= 128:
            self.down4 = UnetDown(in_ch=128, out_ch=256)
            final_channels = 256
        if self.image_size >= 256:
            self.down5 = UnetDown(in_ch=256, out_ch=256)
            final_channels = 256
        self.pool = nn.AvgPool2d(kernel_size=[16, 16], stride=1)

        if self.regression:
            self.toClass = nn.Linear(final_channels, 1)
        else:
            self.toClass = nn.Linear(final_channels, out_class)

    def forward(self, spec):
        # Unet-down
        spec = self.down1(spec)
        spec = self.down2(spec)
        if self.image_size >= 64:
            spec = self.down3(spec)
        if self.image_size >= 128:
            spec = self.down4(spec)
        if self.image_size >= 256:
            spec = self.down5(spec)

        # Classifier
        out_class = self.pool(spec)
        out_class = out_class.view(out_class.size(0), -1)
        out_class = self.toClass(out_class)
        
        return out_class

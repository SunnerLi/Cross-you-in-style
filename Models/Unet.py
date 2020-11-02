import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def basicConv(in_ch, out_ch):
    x = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
         nn.InstanceNorm2d(out_ch),
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
        #print ("Down:", down.shape)
        #print ("res:",res.shape)
        x = torch.cat([down, res], dim=1)
        x = self.conv2d(x)
        return x
class UnetOutUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
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
        return x

# AudioNet
class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        """
        [input channel, output channel, scale to width and height]
        [2, 32, 1/1], [32, 64, 1/2], [64, 128, 1/4], [128, 256, 1/8], [256, 256, 1/16], [256, 256, 1/32]
        """
        super().__init__()
        self.down1 = UnetInDown(in_ch=in_ch, out_ch=32)
        self.down2 = UnetDown(in_ch=32, out_ch=64)
        self.down3 = UnetDown(in_ch=64, out_ch=128)
        self.down4 = UnetDown(in_ch=128, out_ch=256)
        self.down5 = UnetDown(in_ch=256, out_ch=256)
        self.down6 = nn.Conv2d(256, 1, kernel_size=1, padding=0)
        
        self.up5 = nn.Conv2d(1, 256, kernel_size=1, padding=0)
        self.up4 = UnetUp(in_ch=256*2, out_ch=128)
        self.up3 = UnetUp(in_ch=128*2, out_ch=64)
        self.up2 = UnetUp(in_ch=64*2, out_ch=32)
        self.up1 = UnetOutUp(in_ch=32*2, out_ch=out_ch)

    def forward(self, spec):
        # Unet-down
        spec1 = self.down1(spec)
        spec2 = self.down2(spec1)
        spec3 = self.down3(spec2)
        spec4 = self.down4(spec3)
        spec5 = self.down5(spec4) 
        spec6 = self.down6(spec5)   # Music latent

        vis_audio = self.up5(spec6)
        vis_audio = self.up4(vis_audio, spec4)
        vis_audio = self.up3(vis_audio, spec3)
        vis_audio = self.up2(vis_audio, spec2)
        vis_audio = self.up1(vis_audio, spec1) # Reconstructed mel spectrogram
        return vis_audio, spec6
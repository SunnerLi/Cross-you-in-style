import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from Models.layer import SpectralNorm2d

class Self_Attn(nn.Module):
    """ 
        Self attention Layer
    """
    def __init__(self, in_dim, activation):
        super().__init__()
        self.chanel_in = in_dim       
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim // 8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim // 8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)   # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)                      # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key)                                            # transpose check
        attention = self.softmax(energy)                                                    # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)                   # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma * out + x
        return self.activation(out), attention

class Generator(nn.Module):
    def __init__(self, out_channels = 3, image_size = 64, z_dim = 32, music_latent_dim = 1):
        super().__init__()
        self.imsize = image_size
        self.z_dim = z_dim
        self.music_latent_dim = music_latent_dim
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []

        # layer1
        layer1.append(SpectralNorm2d(nn.ConvTranspose2d(self.z_dim + self.music_latent_dim, 256, kernel_size = 4, stride = 2, padding = 1)))
        layer1.append(nn.InstanceNorm2d(256))
        layer1.append(nn.ReLU())
        curr_dim = 256

        # layer2
        if self.imsize >= 64:
            layer2.append(SpectralNorm2d(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size = 4, stride = 2, padding = 1)))
            layer2.append(nn.InstanceNorm2d(int(curr_dim / 2)))
            layer2.append(nn.ReLU())
            curr_dim = int(curr_dim / 2)

        # layer3 (include attention module)
        if self.imsize >= 128:
            layer3.append(SpectralNorm2d(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size = 4, stride = 2, padding = 1)))
            layer3.append(nn.InstanceNorm2d(int(curr_dim / 2)))
            layer3.append(nn.ReLU())
            curr_dim = int(curr_dim / 2)
        self.attn1 = Self_Attn(curr_dim, F.relu)

        # layer4 (include attention module)
        if self.imsize >= 256:
            layer4.append(SpectralNorm2d(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), kernel_size = 4, stride = 2, padding = 1)))
            layer4.append(nn.InstanceNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            curr_dim = int(curr_dim / 2)
        self.attn2 = Self_Attn(curr_dim, F.relu)

        # Last layer (include pooling operator)
        self.music_resize = torch.nn.FractionalMaxPool2d((1, 3), output_size = (8, 8))
        last.append(nn.ConvTranspose2d(curr_dim, out_channels, 4, 2, 1))
        last.append(nn.Tanh())

        # Form the module objects
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)

    def forward(self, music_latent, z = None):
        """
            Forward for SAGAN generator
            Arg:    music_latent    (torch.Tensor)  - The music latent representation, and the shape is (B, 256, 8, 16)
                    z               (torch.Tensor)  - The Gaussian noise, and the shape is (B, 32, 8, 8)
            Ret:    The music representation, and the shape is (B, 3, 256, 256)
        """
        # Prepaer the input
        B, C, H, W = music_latent.shape
        if z is None:
            z = torch.randn([B, self.z_dim, 8, 8])
        if not (music_latent.size(2) == 8 and music_latent.size(3) == 8):
            music_latent = self.music_resize(music_latent)
        z = torch.cat([music_latent, z], dim=1)

        # work
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out, p1 = self.attn1(out)
        out=self.l4(out)
        out, p2 = self.attn2(out)
        out=self.last(out)
        return out


class Discriminator(nn.Module):
    # def __init__(self, in_channels = 3, image_size = 64):
    def __init__(self, in_channels = 3, image_size = 64, vgg = None):
        super().__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        layer5 = []
        last = []

        # layer1
        conv_dim = 64
        layer1.append(SpectralNorm2d(nn.Conv2d(in_channels, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim

        # layer2
        layer2.append(SpectralNorm2d(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        # layer3
        if self.imsize >= 64:
            layer3.append(SpectralNorm2d(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer3.append(nn.LeakyReLU(0.1))
            curr_dim = curr_dim * 2

        # layer4 (include attention module)
        if self.imsize >= 128:
            layer4.append(SpectralNorm2d(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            curr_dim = curr_dim * 2
        self.attn1 = Self_Attn(curr_dim, F.relu)

        # layer5 (include attention module)
        if self.imsize >= 256:
            layer5.append(SpectralNorm2d(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer5.append(nn.LeakyReLU(0.1))
            curr_dim = curr_dim*2
        self.attn2 = Self_Attn(curr_dim, F.relu)

        # Form the module objects
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.l5 = nn.Sequential(*layer5)

        # Last layer
        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out,p1 = self.attn1(out)
        out=self.l5(out)
        out,p2 = self.attn2(out)
        out=self.last(out)
        return out.squeeze()

if __name__ == '__main__':
    """
    z = torch.randn([2, 512, 1, 1])
    music = torch.randn([2, 100, 1, 1])
    G = Generator(2, z_dim=612, image_size=128)
    D = Discriminator(2, image_size=128)
    out = G(music, z)
    out = D(out)
    print(out.size())
    """
    G = Generator(2, z_dim=32, image_size=128)
    z = torch.randn([2, 32, 8, 8])
    music_latent = torch.randn([2, 256, 8, 8])
    print ("Noise:", z.shape)
    print ("music_latent:", music_latent.shape)
    out = G(music_latent, z)
    print ("Generator output:", out.shape)



import torch.nn as nn
import torchvision.models as models
from torch import autograd
import torch.nn.functional as F
import torch

# vgg = models.vgg19(pretrained=True).features.eval()

def calc_gradient_penalty(netD, real_data, fake_data):
    assert real_data.size(0) == fake_data.size(0)
    alpha = torch.rand(fake_data.size(0), 1, 1, 1)
    # print(alpha.size(), real_data.size())
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

class StyleLoss(nn.Module):
    """
        Style loss module

        Ref:    https://www.pytorchtutorial.com/pytorch-style-transfer/
    """
    def __init__(self, vgg_module, layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        """
            The constructor of the style loss module

            Arg:    style_layers    (List)  - The list of the layer name where you want to consider the style loss toward
        """
        super().__init__()
        self.target = None
        self.layers = layers
        self.vgg = vgg_module

    def forward(self, predict, target):
        """
            Compute the style loss

            Arg:    predict (torch.Tensor)  - The predicted image tensor
                    target  (torch.Tensor)  - The target image tensor
            Ret:    The accumulated style loss
        """
        # Compute the gram matrix of predicted image for particular layers
        model = nn.Sequential()
        style_loss = 0.0
        i = 1
        for layer in self.vgg:
            if isinstance(layer, nn.Conv2d):
                name = 'conv_' + str(i)
                model.add_module(name, layer)

                # Accumulate the loss
                if name in self.layers:
                    model = model.cuda()
                    style_loss += F.l1_loss(self.get_gram(model(predict)), self.get_gram(model(target)))
                i += 1
            if isinstance(layer, nn.MaxPool2d):
                model.add_module('pool_' + str(i), layer)
            if isinstance(layer, nn.ReLU):
                model.add_module('relu' + str(i), layer)
        return style_loss

    def get_gram(self, input):
        """
            Return the normalized gram matrix for the given tensor

            Arg:    input   (torch.Tensor)  - The tensor you want to compute
        """
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_module, layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        super().__init__()
        self.target = None
        self.vgg = vgg_module
        self.layers = layers

    def forward(self, predict, target):
        """
            Compute the perceptual loss

            Arg:    predict (torch.Tensor)  - The predicted image tensor
                    target  (torch.Tensor)  - The target image tensor
            Ret:    The accumulated perceptual loss
        """
        # Compute the gram matrix of predicted image for particular layers
        model = nn.Sequential()
        perceptual_loss = 0.0
        i = 1
        for layer in self.vgg:
            if isinstance(layer, nn.Conv2d):
                name = 'conv_' + str(i)
                model.add_module(name, layer)

                # Accumulate the loss
                if name in self.layers:
                    predict = predict.to('cuda')
                    target = target.to('cuda')
                    perceptual_loss += F.l1_loss(model(predict), model(target))
                i += 1
            if isinstance(layer, nn.MaxPool2d):
                model.add_module('pool_' + str(i), layer)
            if isinstance(layer, nn.ReLU):
                model.add_module('relu' + str(i), layer)
        return perceptual_loss

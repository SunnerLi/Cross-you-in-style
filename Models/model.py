from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import numpy as np
import itertools
import torch
import os

from Models.Unet import Unet
from Models.Classifier import Classifier
from Models.loss import StyleLoss, PerceptualLoss, calc_gradient_penalty
from Models.generate import Generator, Discriminator
        
class SVN(nn.Module):
    # def __init__(self, z_dims=32, out_class=92):
    def __init__(self, z_dim=32, image_size = 128, out_class=92, regression = False, triplet = False):
        """
            The constructor of style visualization network
            There are 4 loss term:
                1. Style loss (SE, G)
                2. Perceptual loss (SE, G)
                3. Adversarial loss (G)
                4. Reconstruction loss (SE, SD)
        """
        super().__init__()
        self.z_dim = z_dim
        self.regression = regression
        self.triplet = triplet

        # Define the loss term weight
        # self.lambda_style = 1e-1
        # #self.lambda_percp = 1e-1
        # self.lambda_adver = 1.0
        # self.lambda_recon = 5.0
        # self.lambda_class = 1.0
        # self.lambda_tripe = 1.0
        self.lambda_style = 10.0
        self.lambda_adver = 1.0
        self.lambda_recon = 1.0
        self.lambda_class = 1e-1
        self.lambda_tripe = 1e-1

        # Define loss list
        self.loss_list_style = []
        self.loss_list_adver_g = []
        self.loss_list_adver_d = []
        # self.loss_list_recon = []
        self.loss_list_class = []
        #self.loss_list_acc = []
        if self.triplet:
            self.loss_list_triplet = []

        self.Loss_list_style = []
        self.Loss_list_adver_g = []
        self.Loss_list_adver_d = []
        # self.Loss_list_recon = []
        self.Loss_list_class = []
        #self.Loss_list_acc = []
        if self.triplet:
            self.Loss_list_triplet = []

        # Define network structure
        self.U = Unet(in_ch=3)
        self.G = Generator(image_size = image_size, z_dim = z_dim)
        self.D = Discriminator(image_size = image_size)
        self.C = Classifier(in_ch = 3, out_class = out_class, regression = regression, image_size = image_size)

        # Define optimizer
        self.optim_U = torch.optim.Adam(self.U.parameters(), lr=1e-4, weight_decay=0.0001)
        self.optim_G = torch.optim.Adam(self.G.parameters(), lr=1e-4, weight_decay=0.0001)
        self.optim_D = torch.optim.Adam(self.D.parameters(), lr=1e-4, weight_decay=0.0001)
        self.optim_C = torch.optim.Adam(self.C.parameters(), lr=1e-4, weight_decay=0.0001)


        # Define criterion
        self.vgg = models.vgg19(pretrained=True).features.eval()
        self.crit_style = StyleLoss(vgg_module = self.vgg) # Mysterious style loss
        self.crit_adver = nn.BCELoss()                     # Adversarila loss
        self.crit_recon = nn.L1Loss()                      # Mel-spectrogram reconstruction loss
        # self.crit_class = nn.CrossEntropyLoss()            # Classification loss
        if self.regression:
            self.crit_class = nn.MSELoss()
        else:
            self.crit_class = nn.CrossEntropyLoss()
        self.crit_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        """
        self.crit_percp = PerceptualLoss(vgg_module = self.vgg)
        """

        self.z = torch.randn([1, self.z_dim, 8, 8]).cuda()

    def resample(self):
        self.z = torch.randn([1, self.z_dim, 8, 8]).cuda()

    def forward(self, spec_in):
        # 1. Unet to extract music latent
        spec_out, music_latent = self.U(spec_in)

        # 2. Generate music representation
        b, c, h, w = music_latent.size()
        fake_style = self.G(music_latent, self.z)

        # 3. Year classification
        pred_year = self.C(fake_style)

        return spec_out, fake_style, pred_year, music_latent

    # ==============================================================================
    #   IO
    # ==============================================================================
    def load(self, path):
        if os.path.exists(path):
            print("Load the pre-trained model from {}".format(path))
            state = torch.load(path)
            for (key, obj) in state.items():
                if len(key) > 10:
                    if key[1:9] == 'oss_list':
                        setattr(self, key, obj)
            self.U.load_state_dict(state['U'])
            self.G.load_state_dict(state['G'])
            self.D.load_state_dict(state['D'])
            self.C.load_state_dict(state['C'])
            self.optim_U.load_state_dict(state['optim_U'])
            self.optim_G.load_state_dict(state['optim_G'])
            self.optim_D.load_state_dict(state['optim_D'])
            self.optim_C.load_state_dict(state['optim_C'])
        else:
            print("Pre-trained model {} is not exist...".format(path))

    def save(self, path):
        # Record the parameters
        state = {
            'U': self.U.state_dict(),
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict(),
        }

        # Record the optimizer and loss
        state['optim_U'] = self.optim_U.state_dict()
        state['optim_G'] = self.optim_G.state_dict()
        state['optim_D'] = self.optim_D.state_dict()
        state['optim_C'] = self.optim_C.state_dict()
        for key in self.__dict__:
            if len(key) > 10:
                if key[1:9] == 'oss_list':
                    state[key] = getattr(self, key)
        torch.save(state, path)

    # ==============================================================================
    #   Set & Get
    # ==============================================================================
    def getLoss(self, normalize = False):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'loss_list':
                if not normalize:
                    loss_dict[key] = round(getattr(self, key)[-1], 6)
                else:
                    loss_dict[key] = np.mean(getattr(self, key))
        return loss_dict

    def getLossList(self):
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'Loss_list':
                loss_dict[key] = getattr(self, key)
        return loss_dict

    # ==============================================================================
    #   Backward function
    # ==============================================================================
    def backward_U(self, out_spec, in_spec, pred_year, gt_year, true_style, fake_style, anchor = None, pos = None, neg = None):
        """
        Spec reconstruction loss + Year classification loss + Style loss (+ Triplet loss)
        """

        # # Reconstruction loss for spectrogram
        # loss_recon = self.crit_recon(out_spec, in_spec) * self.lambda_recon
        # self.loss_list_recon.append(loss_recon.item())

        # Year classification lss
        pred_year = pred_year.view(pred_year.size(0), -1)
        loss_class = self.crit_class(pred_year, gt_year) * self.lambda_class
        self.loss_list_class[-1] += loss_class.item()

        # Style loss
        loss_style = self.crit_style(fake_style, true_style) * self.lambda_style
        self.loss_list_style[-1] += loss_style.item()

        # Triplet loss
        if self.triplet:
            loss_triplet = self.crit_triplet(anchor = anchor, positive = pos, negative = neg) * self.lambda_tripe
            self.loss_list_triplet.append(loss_triplet.item())

        # Sum up the loss
        if self.triplet:
            # loss_u = loss_class + loss_recon + loss_style + loss_triplet
            loss_u = loss_class + loss_style + loss_triplet
        else:
            # loss_u = loss_class + loss_recon + loss_style
            loss_u = loss_class + loss_style
        loss_u.backward()

    def backward_G(self, true_style, fake_style, pred_year, gt_year):
        """
        Adversarial loss + Year classification loss + Style loss
        """
        # ---------------------------------------------------------------
        # Update for adversarial loss
        # ---------------------------------------------------------------
        ##### WGAN-GP
        # loss_adv = -self.D(fake_style).mean() * self.lambda_adver
        ##### Relativistic LSGAN
        r_logit = self.D(true_style)
        f_logit = self.D(fake_style)
        loss_adv = ((torch.mean((r_logit - torch.mean(f_logit) + 1) ** 2) + torch.mean((f_logit - torch.mean(r_logit) - 1) ** 2))/2) * self.lambda_adver
        self.loss_list_adver_g.append(loss_adv.item())

        # Year classification lss
        pred_year = pred_year.view(pred_year.size(0), -1)
        loss_class = self.crit_class(pred_year, gt_year) * self.lambda_class
        self.loss_list_class[-1] += loss_class.item()

        # Update for style loss
        loss_style = self.crit_style(fake_style, true_style) * self.lambda_style
        self.loss_list_style.append(loss_style.item())

        # Merge the several loss terms
        loss_g = loss_style + loss_class + loss_adv
        # loss_g = loss_percp + loss_adv
        loss_g.backward()
        
    def backward_D(self, true_style, fake_style):
        """
        Adversarial loss
        """
        loss_adv = 0.0
        ##### WGAN-GP
        # for iter_d in range(3):
        #     r_logit = self.D(true_style)
        #     f_logit = self.D(fake_style)
        #     gradient_penalty = calc_gradient_penalty(self.D, true_style, fake_style)
        #     loss_adv = loss_adv + (f_logit.mean() - r_logit.mean() + gradient_penalty) * self.lambda_adver
        
        ##### Relativistic LSGAN
        r_logit = self.D(true_style)
        f_logit = self.D(fake_style)
        loss_adv = loss_adv + ((torch.mean((r_logit - torch.mean(f_logit) - 1) ** 2) + torch.mean((f_logit - torch.mean(r_logit) + 1) ** 2))/2) * self.lambda_adver

        self.loss_list_adver_d.append(loss_adv.item())
        loss_adv.backward()

    def backward_C(self, pred_year, gt_year):
        """
            Update the year classifier via year classification loss
            Args:   pred_year   (torch.Tensor)  - The year prediction, and the shape is (B, #class)
                    gt_year     (torch.Tensor)  - The ground truth of year, and the shape is (B). dtype = torch.long
                    batch_size  (Int)
        """
        # Compute training acc
        # correct = 0
        # pred = pred_year.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        # correct += pred.eq(gt_year.view_as(pred)).sum().item()
        # self.loss_list_acc.append(100. * correct / gt_year.size(0))

        # Year classification lss
        pred_year = pred_year.view(pred_year.size(0), -1)
        loss_class = self.crit_class(pred_year, gt_year) * self.lambda_class
        self.loss_list_class.append(loss_class.item())
        loss_class.backward()

    def backward(self, in_spec, true_style, gt_year, pos = None, neg = None):
        """
            Update the parameters of whole model
            Args:   in_spec     (torch.Tensor)  - The input music mel-spectrogram. The shape is (B, 3, 128, 259)
                    true_style  (torch.Tensor)  - GT image. The shape is (B, 3, 256, 256)
                    gt_year     (torch.Tensor)  - Ground truth year, with each item ranging from 0 to (#class - 1)
                    pos         (torch.Tensor)  - The positive latent representation (to compute triplet loss)
                    neg         (torch.Tensor)  - The negative latent representation (to compute triplet loss)
                    batch_size  (Int)           - Need batch_size to reshape tensor when calculating year classification loss with batch_size=1.
        """
        # Update discriminator
        _, fake_style, _, _ = self.forward(in_spec)
        self.optim_D.zero_grad()
        self.backward_D(true_style, fake_style)
        self.optim_D.step()

        # Update classifier
        # _, _, pred_year, _ = self.forward(in_spec)
        pred_year = self.C(true_style)
        self.optim_C.zero_grad()
        self.backward_C(pred_year, gt_year)
        self.optim_C.step()

        # Update generator
        _, fake_style, pred_year, _ = self.forward(in_spec)
        self.optim_G.zero_grad()
        self.backward_G(true_style, fake_style, pred_year, gt_year)
        self.optim_G.step()

        # Update Unet
        out_spec, fake_style, pred_year, music_latent = self.forward(in_spec)
        self.optim_U.zero_grad()
        if self.triplet:
            _, pos_latent = self.U(pos)
            _, neg_latent = self.U(neg)
            self.backward_U(out_spec, in_spec, pred_year, gt_year, true_style, fake_style, music_latent, pos_latent, neg_latent)
        else:
            self.backward_U(out_spec, in_spec, pred_year, gt_year, true_style, fake_style)
        self.optim_U.step()

    # ====================================================================================================================================
    #   Finish epoch
    # ====================================================================================================================================
    def finish(self):
        for key in self.__dict__:
            if len(key) > 9 and key[0:9] == 'loss_list':
                sum_lost_list = getattr(self, 'L' + key[1:])
                sum_lost_list.append(np.mean(getattr(self, key)))       
                setattr(self, 'L' + key[1:], sum_lost_list)
                setattr(self, key, [])

# if __name__ == "__main__":

#    # Visible GPU
#    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(s) for s in [0, 1])

#    # Test one iteration
#    print ("Test one iteration")

#    model = SVN()
#    model = torch.nn.DataParallel(model).cuda()

#    B = 1

#    in_spec = torch.rand([B, 3, 128, 259]).cuda()
#    true_style = torch.rand([B, 3, 256, 256]).cuda()
#    gt_year = torch.empty(B, dtype=torch.long).random_(98).cuda()

#    # Start Testing
#    model.module.backward(in_spec, true_style, gt_year, batch_size=B)
#    print ("Pass")
# model = SVN(z_dim=256, image_size=64, out_class=54).cuda()
# model.load('experiment_result_history/20190729/Result_no_reg_with_triplet_ch256_1_8.91_exp1/Model/Result_no_reg_with_triplet_ch256_1_8.91_exp1.pth')

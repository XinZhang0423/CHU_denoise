from matplotlib.pyplot import xcorr
import torch
import torch.nn as nn
import pytorch_lightning as pl

import os

class VAE_DIP_2D(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        # Set random seed if asked (for NN weights here)
        if (os.path.isfile(os.getcwd() + "/seed.txt")):
            with open(os.getcwd() + "/seed.txt", 'r') as file:
                random_seed = file.read().rstrip()
            if (eval(random_seed)):
                pl.seed_everything(1)

        # Defining variables from config
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        if (config['mlem_sequence'] is None):
            self.post_reco_mode = True
        else:
            self.post_reco_mode = False

        # Defining CNN variables
        L_relu = 0.2
        num_channel = [16, 32, 64, 128]
        pad = [0, 0]
        
        # Dimensions before and after going to the latent space
        self.latent_dim = 2
        input_size = 128
        self.high_dim = int(num_channel[-1] * (input_size / 2**(len(num_channel) - 1)) **2)

        # Layers in CNN architecture
        self.deep1 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(1, num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.down1 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.deep2 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.down2 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.deep3 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.down3 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.deep4 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[3]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[3], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[3]),
                                   nn.LeakyReLU(L_relu))

        self.layer_mu = nn.Sequential(nn.Flatten(),
                                      nn.Linear(self.high_dim,self.latent_dim*4),
                                      nn.Linear(self.latent_dim*4,self.latent_dim))
        
        self.layer_logvar = nn.Sequential(nn.Flatten(),
                                          nn.Linear(self.high_dim,self.latent_dim*4),
                                          nn.Linear(self.latent_dim*4,self.latent_dim))

        self.up0 = nn.Sequential(nn.Linear(self.latent_dim,self.high_dim))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[3], num_channel[2], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[2]),
                                 nn.LeakyReLU(L_relu))

        self.deep5 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[2], num_channel[1], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[1]),
                                 nn.LeakyReLU(L_relu))

        self.deep6 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[1], num_channel[0], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[0]),
                                 nn.LeakyReLU(L_relu))

        self.deep7 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], 1, (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(1))

        self.positivity = nn.ReLU() # Final ReLU to enforce positivity of ouput image
        # self.positivity = nn.SiLU() # Final SiLU, smoother than ReLU but not positive
        # self.positivity = nn.Softplus() # Final SiLU to enforce positivity of ouput image, smoother than ReLU

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        # Encoder
        out1 = self.deep1(x)
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)

        # VAE part
        before_mu = out
        before_logvar = out
        # get the latent vector through reparameterization
        z, mu, logvar = self.reparameterize(before_mu, before_logvar, x) # 2D sample

        # Retrieve dimensions before vae part
        out = self.up0(z)
        out = out.view(-1,before_mu.shape[1],before_mu.shape[2],before_mu.shape[3])

        # Decoder
        out = self.up1(out)
        out_skip1 = out3 + out
        out = self.deep5(out_skip1)
        out = self.up2(out)
        out_skip2 = out2 + out
        out = self.deep6(out_skip2)
        out = self.up3(out)
        out_skip3 = out1 + out
        out = self.deep7(out_skip3)
        #out = self.positivity(out)

        return out, mu, logvar, z

    def reparameterize(self, mu, logvar, x):
        z_mu = self.layer_mu(mu).type_as(x)
        z_logvar = self.layer_logvar(logvar).type_as(x)
        z_std = torch.experiment(0.5*z_logvar) # standard deviation computed from log variance
        eps = torch.randn_like(z_mu) # N(0,I) computed with `randn_like` as we need the same size
        z = z_mu + (eps * z_std) # sampling as if coming from the input space
        return z, z_mu, z_std

    def gaussian_likelihood(self, x_hat, logscale, x): # return logarithm of gaussian likelihood (so with minus...)
        scale = torch.experiment(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale) # normal distribution object, with mean x_hat (output of DIP) and std scale (experiment(logscale) = experiment(0) = 1)
        # log prob = ||x_hat
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x) # x is x label = corrupted image = label of DIP

        return log_pxz.sum(dim=(1, 2, 3)) # compute norm

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def DIP_loss(self, out, image_corrupt_torch, mu, logvar, z):
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(out, self.log_scale, image_corrupt_torch) # corrupted image = label for DIP
        # kl divergence
        kl = self.kl_divergence(z, mu, torch.experiment(logvar / 2))
        # elbo
        elbo = (kl - recon_loss) # minus because recon_loss is minus norm
        loss = elbo.mean()
        self.log('loss_monitor', loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out, mu, logvar, z = self.forward(image_net_input_torch)
        # Save image over epochs
        if (self.post_reco_mode):
            self.post_reco(out)
        loss = self.DIP_loss(out, image_corrupt_torch, mu, logvar, z)
        # logging using tensorboard logger
        self.logger.experiment.add_scalar('loss', loss,self.current_epoch)        
        return loss

    def configure_optimizers(self):
        # Optimization algorithm according to command line
        if (self.opti_DIP == 'Adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
        elif (self.opti_DIP == 'LBFGS' or self.opti_DIP is None): # None means no argument was given in command line
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4) # Optimizing using L-BFGS
        return optimizer

    def post_reco(self,out):
        from utils.utils_func import save_img
        if ((self.current_epoch%(self.sub_iter_DIP // 10) == 0)):
            out_np = out.detach().numpy()[0,0,:,:]
            subroot = 'data/Algo/'
            experiment = 24
            save_img(out_np, subroot+'Block2/out_cnn/' + format(experiment) + '/out_' + 'DIP_VAE' + '_post_reco_epoch=' + format(self.current_epoch) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
        
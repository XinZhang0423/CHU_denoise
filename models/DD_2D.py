import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
        
class DD_2D(pl.LightningModule):

    def __init__(self, config, method):
        super().__init__()

        # Set random seed if asked (for NN weights here)
        if (os.path.isfile(os.getcwd() + "/seed.txt")):
            with open(os.getcwd() + "/seed.txt", 'r') as file:
                random_seed = file.read().rstrip()
            if (eval(random_seed)):
                pl.seed_everything(1)

        self.writer = SummaryWriter()
        # Defining variables from config
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.method = method
        if (config['mlem_sequence'] is None):
            self.post_reco_mode = True
            self.suffix = self.suffix_func(config)
        else:
            self.post_reco_mode = False
        d = config["d_DD"] # Number of layers
        k = config['k_DD'] # Number of channels, depending on how much noise we mant to remove. Small k = less noise, but less fit

        # Defining CNN variables
        self.num_channels_up = [k]*(d+1) + [1]
        self.decoder_layers = nn.ModuleList([])

        # Layers in CNN architecture
        for i in range(len(self.num_channels_up)-2):       
            self.decoder_layers.append(nn.Sequential(
                               nn.Conv2d(self.num_channels_up[i], self.num_channels_up[i+1], 1, stride=1),
                               nn.Upsample(scale_factor=2, mode='bilinear'),
                               nn.ReLU(),
                               nn.BatchNorm2d(self.num_channels_up[i+1]))) #,eps=1e-10))) # For uniform input image, default epsilon is too small which amplifies numerical instabilities 

        self.last_layers = nn.Sequential(nn.Conv2d(self.num_channels_up[-2], self.num_channels_up[-1], 1, stride=1))
        
        self.positivity = nn.ReLU() # Final ReLU to enforce positivity of ouput image
        # self.positivity = nn.SiLU() # Final SiLU, smoother than ReLU but not positive
        # self.positivity = nn.Softplus() # Final SiLU to enforce positivity of ouput image, smoother than ReLU


    def write_image_tensorboard(self,writer,image,name,suffix,i=0,full_contrast=False):
        # Creating matplotlib figure with colorbar
        plt.figure()
        if (len(image.shape) != 2):
            print('image is ' + str(len(image.shape)) + 'D, plotting only 2D slice')
            image = image.detach().numpy()[0,0,:,:]
        if (full_contrast):
            plt.imshow(image, cmap='gray_r',vmin=np.min(image),vmax=np.max(image)) # Showing each image with maximum contrast  
        else:
            plt.imshow(image, cmap='gray_r',vmin=0,vmax=500) # Showing all images with same contrast
        plt.colorbar()
        #plt.axis('off')
        # Adding this figure to tensorboard
        writer.add_figure(name,plt.gcf(),global_step=i,close=True)# for videos, using slider to change image with global_step

    def forward(self, x):
        out = x
        for i in range(len(self.num_channels_up)-2):
            out = self.decoder_layers[i](out)
        out = self.last_layers(out)
        #self.write_image_tensorboard(self.writer,out,"TEST (" + 'DD' + "output, FULL CONTRAST)","",0,full_contrast=True) # Showing each image with contrast = 1
        if (self.method == 'Gong'):
            out = self.positivity(out)
        return out

    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD

    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out = self.forward(image_net_input_torch)
        # Save image over epochs
        if (self.post_reco_mode):
            self.post_reco(out)
        loss = self.DIP_loss(out, image_corrupt_torch)
        # logging using tensorboard logger
        self.logger.experiment.add_scalar('loss', loss,self.current_epoch)        
        return loss

    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """

        if (self.opti_DIP == 'Adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
        elif (self.opti_DIP == 'LBFGS' or self.opti_DIP is None): # None means no argument was given in command line
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4) # Optimizing using L-BFGS
        return optimizer

    def post_reco(self,out):
        from utils.utils_func import save_img
        if ((self.current_epoch%(self.sub_iter_DIP // 10) == 0)):
            try:
                out_np = out.detach().numpy()[0,0,:,:]
            except:
                out_np = out.cpu().detach().numpy()[0,0,:,:]
            subroot = '/home/meraslia/sgld/hernan_folder/data/Algo/'
            experiment = 24
            save_img(out_np, subroot+'Block2/out_cnn/' + format(experiment) + '/out_' + 'DD' + '_post_reco_epoch=' + format(self.current_epoch) + self.suffix + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
                    
    def suffix_func(self,config):
        suffix = "config"
        for key, value in config.items():
            suffix +=  "_" + key + "=" + str(value)
        return suffix
# deep decoder by default
import torch
import torch.nn as nn
import pytorch_lightning as pl

import numpy as np
from utils.pre_utils import *
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import os


class DD_2D(pl.LightningModule):

    def __init__(self, config, param1_scale_im_corrupt, param2_scale_im_corrupt,ground_truth,suffix):
        #初始化内部参数
        super().__init__()

        # Set random seed if asked (for NN weights here) 这里是为了初始化权重的时候固定随机数，还原每次模型的结果
        self.random_seed = config['random_seed']
        if self.random_seed :
            pl.seed_everything(3407)
        # Defining variables from config
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        #我后加的
        self.gt = np.squeeze(ground_truth)
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.path="/home/xzhang/Documents/我的模型/data/results/images/"     
        self.suffix  = suffix
        self.repeat = config['repeat']
        self.mask = np.load(config['mask_path'])
        # 记录input位置
        self.input_path = config['input_path'].split('/')[-1].split('.')[0]

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
        
        self.positivity = nn.ReLU()


    def forward(self, x):
        out = x
        for i in range(len(self.num_channels_up)-2):
            out = self.decoder_layers[i](out)
        out = self.last_layers(out)
        out = self.positivity(out)
        
        return out

    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD


    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out = self.forward(image_net_input_torch)

            
        loss = self.DIP_loss(out, image_corrupt_torch)
        
        try:
            out_np = out.detach().numpy()
        except:
            out_np = out.cpu().detach().numpy()
        
        out_np = np.squeeze( out_np)
        out_np = descale_imag(out_np,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,scaling='normalization')
        
        os.makedirs(self.path + format(self.suffix) +'/' + self.input_path + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) +'/' +  self.input_path + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
    
        return loss

    def configure_optimizers(self):
        """
        Optimization of the DNN with SGLD
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
        return optimizer

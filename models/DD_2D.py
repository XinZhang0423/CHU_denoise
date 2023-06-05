# deep decoder by default
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
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
        
        self.writer = SummaryWriter()
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
        
        # # self.method = method
        # if (config['mlem_sequence'] is None):
        #     self.post_reco_mode = True
        #     self.suffix = self.suffix_func(config)
        # else:
        #     self.post_reco_mode = False
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

    def forward(self, x):
        out = x
        for i in range(len(self.num_channels_up)-2):
            out = self.decoder_layers[i](out)
        out = self.last_layers(out)
        #self.write_image_tensorboard(self.writer,out,"TEST (" + 'DD' + "output, FULL CONTRAST)","",0,full_contrast=True) # Showing each image with contrast = 1
        # if (self.method == 'Gong'):
        out = self.positivity(out)
        
        return out
    
   # 定义metric
    def DIP_metric(self, out_np):
        psnr_gt = peak_signal_noise_ratio(self.gt, out_np, data_range=np.amax(out_np)-np.amin(out_np))
        ssim = structural_similarity(self.gt, out_np, data_range=np.amax(out_np)-np.amin(out_np))
        
        return psnr_gt, ssim


    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD

    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out = self.forward(image_net_input_torch)
        # # Save image over epochs
        # if (self.post_reco_mode):
        #     self.post_reco(out)
            
        loss = self.DIP_loss(out, image_corrupt_torch)
        
        try:
            out_np = out.detach().numpy()
        except:
            out_np = out.cpu().detach().numpy()
        
        out_np = np.squeeze( out_np)
        out_np = descale_imag(out_np,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,scaling='normalization')
        
        # logging using tensorboard logger
        
        os.makedirs(self.path + format(self.suffix) +'/' + self.input_path + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) +'/' +  self.input_path + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
        
        mse = np.mean((self.gt*self.mask -out_np*self.mask)**2)
        # 256，256 原来的 和 out_np 256,256 原来的
        psnr, ssim = self.DIP_metric(out_np)
        self.logger.experiment.add_scalar('loss', loss,self.current_epoch)        
        self.logger.experiment.add_scalar('psnr',psnr,self.current_epoch)
        self.logger.experiment.add_scalar('ssim',ssim,self.current_epoch)
        self.logger.experiment.add_scalar('mse with ground truth',mse,self.current_epoch)
        
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
        return suffix
    

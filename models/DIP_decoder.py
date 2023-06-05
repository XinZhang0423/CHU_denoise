import torch
import torch.nn as nn
import pytorch_lightning as pl

from numpy import inf
import numpy as np

import matplotlib.pyplot as plt

import os

from skimage.metrics import peak_signal_noise_ratio,structural_similarity

from utils.pre_utils import *

# Local files to import
# from iWMV import iWMV

class DIP_decoder(pl.LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, 
                 config,suffix,ground_truth):
        #初始化内部参数
        super().__init__()
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        # self.initial_param = initial_param
        
        self.path="/home/xzhang/Documents/我的模型/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        self.mask = np.load(config['mask_path'])
        # 记录input位置
        self.input_path = config['input_path'].split('/')[-1].split('.')[0]   
        self.config = config
 
        
        L_relu = 0.2
        num_channel = [16, 32, 64,128]
        # num_channel = [8, 16, 32, 64, 64]
        # num_channel = [32, 64, 128, 256]
        pad = [0, 0]


        # self.up0 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
        #                          nn.ReplicationPad2d(1),
        #                          nn.Conv2d(num_channel[4], num_channel[3], 3, stride=(1, 1), padding=pad[0]),
        #                          nn.BatchNorm2d(num_channel[3]),
        #                          nn.LeakyReLU(L_relu))

        # self.deep4 = nn.Sequential(nn.ReplicationPad2d(1),
        #                            nn.Conv2d(num_channel[3], num_channel[3], (3, 3), stride=1, padding=pad[1]),
        #                            nn.BatchNorm2d(num_channel[3]),
        #                            nn.LeakyReLU(L_relu),
        #                            nn.ReplicationPad2d(1),
        #                            nn.Conv2d(num_channel[3], num_channel[3], (3, 3), stride=1, padding=pad[1]),
        #                            nn.BatchNorm2d(num_channel[3]),
        #                            nn.LeakyReLU(L_relu))

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

        self.positivity = nn.ReLU() 
        s  = sum([np.prod(list(p.size())) for p in self.parameters()]); 
        print ('Number of params: %d' % s)
        
    def forward(self, x):
        # Decoder
        # out = self.up0(x)
        # out = self.deep4(out)
        out = self.up1(x)
        out = self.deep5(out)
        out = self.up2(out)
        out = self.deep6(out)
        out = self.up3(out)
        out = self.deep7(out)
        out = self.positivity(out)
        return out 
    
    # 定义损失函数为输出和含噪图像的tensor的mse
    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) 
        
    # 定义训练流程，计算一次前向传播返回loss，（中间有logger记录tensorboard和early stopping）
    def training_step(self, train_batch, batch_idx):
        # train_batch 包含image_net_input_torch和image_corrupt_torch 分别为噪音图像和含噪图像 
        image_net_input_torch, image_corrupt_torch = train_batch
        # out 1,1,256,256 torch
        out = self.forward(image_net_input_torch)
        loss = self.DIP_loss(out, image_corrupt_torch)
        try:
            out_np = out.detach().numpy()
        except:
            out_np = out.cpu().detach().numpy()
        # 256,256 numpy
        out_np = np.squeeze(out_np)
        
        # 256，256 原来的 和 out_np 256,256 原来的
        out_np = destand_numpy_imag(out_np,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt)
        
        os.makedirs(self.path + format(self.suffix) +'/' + self.input_path + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) +'/' +  self.input_path + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
        
        return loss
    
    #配置优化器，可以选择各种优化器
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8)
        return optimizer


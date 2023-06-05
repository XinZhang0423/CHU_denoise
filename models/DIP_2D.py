import torch
import torch.nn as nn
import pytorch_lightning as pl

from numpy import inf
import numpy as np

import matplotlib.pyplot as plt

import os

from skimage.metrics import peak_signal_noise_ratio,structural_similarity

from utils.pre_utils import *
from models.iWMV import *
class DIP_2D(pl.LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, 
                 config, root,  method, all_images_DIP,
                 global_it, suffix, last_iter,ground_truth,initial_param,target=None):
        #初始化内部参数
        super().__init__()

        # Set random seed if asked (for NN weights here) 这里是为了初始化权重的时候固定随机数，还原每次模型的结果
        self.random_seed = config['random_seed']
        if self.random_seed :
            pl.seed_everything(3407)
        
        # Defining variables from config
        #依次为学习率，优化器种类（adam） ，迭代次数，skip 个数，方法，是否保存DIP， ？，？，？ ，保存路径，？，？，early stopping的算法相关     
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.skip = config['skip_connections']
        self.method = method
        self.all_images_DIP = True #all_images_DIP
        self.global_it = global_it
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.initial_param = config['initial_param']
        self.repeat = config['repeat']
        self.path="/home/xzhang/Documents/我的模型/data/results/images/"     
        self.config = config
        self.root = root
        #和tensorboard相关
        self.input_path = config['input_path'].split('/')[-1].split(".")[0] 
        # self.target_path = config['target_path'].split('/')[-2]
        # self.es_path = "/home/xzhang/Documents/我的模型/data/results/images/es"
        
        
        self.experiment = config["experiment"]
        self.info_list=list()
        # 输入target, ground truth 256,256,1
        self.mask = np.load(config['mask_path'])
        
        self.target = np.squeeze(target)
        self.gt = np.squeeze(ground_truth)
        # self.target,gt 256,256 self.gt_torch = 1,1,256,256
        self.gt_torch = torch.Tensor(ground_truth)#.view(1,1,112,112,1)[:,:,:,:,0]
        self.full_contrast = True
        self.psnr_list = []
        self.mse_gt_list = []
        # 以下early stopping相关的我暂时不关心
        # Defining variables from config    
        self.DIP_early_stopping = config["DIP_early_stopping"]
        self.classWMV = iWMV(config)
        self.scaling_input = 'normalization'
        if(self.DIP_early_stopping):
            
            self.initialize_WMV(config,param1_scale_im_corrupt,param2_scale_im_corrupt,self.scaling_input,suffix,global_it,root)

        self.write_current_img_mode = True
        self.suffix = suffix
        
        self.last_iter = last_iter + 1
        
        # Defining CNN variables
        L_relu = 0.2
        num_channel = [16, 32, 64, 128]
        pad = [0, 0]

        # Layers in CNN architecture,定义各个层
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


        # self.init_weights()
        s  = sum([np.prod(list(p.size())) for p in self.parameters()]); 
        print ('Number of params: %d' % s)
        
    def forward(self, x):
        # Encoder
        out1 = self.deep1(x)
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)
        encoder_output = out.clone()
        # Decoder
        out = self.up1(out)
        if (self.skip >= 1):
            out_skip1 = out3 + out
            out = self.deep5(out_skip1)
        else:
            out = self.deep5(out)
        out = self.up2(out)
        if (self.skip >= 2):
            out_skip2 = out2 + out
            out = self.deep6(out_skip2)
        else:
            out = self.deep6(out)
        out = self.up3(out)
        if (self.skip >= 3):
            out_skip3 = out1 + out
            out = self.deep7(out_skip3)
        else:
            out = self.deep7(out)

        out = self.positivity(out)

        return out

    def init_weights(self):
        for m in self.modules():
		# 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                if self.initial_param == 'xavier_norm':
                    torch.nn.init.xavier_normal_(m.weight.data)
                elif self.initial_param == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data)
                elif self.initial_param == 'kaiming_norm':
                    torch.nn.init.kaiming_normal_(m.weight.data)
                elif self.initial_param == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight.data)
                # # 判断是否有偏置
                # if m.bias is not None:
                #     torch.nn.init.constant_(m.bias.data,0.3)

            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1) 		 
            #     m.bias.data.fill_(0)	
        
    # 定义损失函数为输出和含噪图像的tensor的mse
    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD
    
    # 定义metric
    def DIP_metric(self, out_np):
        psnr_gt = peak_signal_noise_ratio(self.gt, out_np, data_range=np.amax(out_np)-np.amin(out_np))
        ssim = structural_similarity(self.gt, out_np, data_range=np.amax(out_np)-np.amin(out_np))
        return psnr_gt, ssim
        
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
            
        out_np = np.squeeze(out_np)
        out_np = descale_imag(out_np,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,scaling='normalization')
        
        # os.makedirs(self.path + format(self.suffix) +'/' + self.initial_param + '/train_' + str(self.repeat), exist_ok=True)
        # np.save(self.path + format(self.suffix) +'/' +  self.initial_param + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
        
        os.makedirs(self.path + format(self.suffix) +'/' + self.input_path + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) +'/' +  self.input_path + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
        
        # os.makedirs(self.path + format(self.suffix) +'/' + self.target_path + '/train_' + str(self.repeat), exist_ok=True)
        # np.save(self.path + format(self.suffix) +'/' +  self.target_path + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
        
        mse = np.mean((self.gt*self.mask -out_np*self.mask)**2)
        self.mse_gt_list.append(mse)
        # 256，256 原来的 和 out_np 256,256 原来的
        psnr_gt, ssim = self.DIP_metric(out_np)
        
        # self.psnr_list.append(psnr_gt)
        
        # #使用tensorboard logger记录loss


        self.logger.experiment.add_scalar('loss',loss,self.current_epoch)
        # # self.logger.experiment.add_scalar('loss_log',loss,np.log10(np.clip(self.current_epoch,a_min=1,a_max=10000)))
        self.logger.experiment.add_scalar('psnr_gt',psnr_gt,self.current_epoch)
        self.logger.experiment.add_scalar('ssim',ssim,self.current_epoch)
        self.logger.experiment.add_scalar('mse with ground truth',mse,self.current_epoch)

        # if (self.current_epoch%50 == 2):
        #     figure = self.plot_images(out_np)
        #     self.logger.experiment.add_figure('image',figure,self.current_epoch,close=True)
        # self.run_WMV(out,self.config,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt,self.scaling_input,self.suffix,self.global_it,self.root)   
        return loss
    
    def plot_images(self,out_np):
        
        plt.figure()
        if (self.full_contrast):
            plt.imshow(out_np, cmap='gray_r',vmin=np.min(out_np),vmax=np.max(out_np)) # Showing each out_np with maximum contrast and white is zero (gray_r) 
        else:
            plt.imshow(out_np, cmap='gray_r',vmin=np.min(out_np),vmax=1.25*np.max(out_np)) # Showing all images with same contrast and white is zero (gray_r)
        plt.colorbar()
        plt.title('train results',fontsize=16)

        return plt.gcf()
    
    #配置优化器，可以选择各种优化器
    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """

        if (self.opti_DIP == 'Adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam

        elif (self.opti_DIP == 'LBFGS' or self.opti_DIP is None): # None means no argument was given in command line
            # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4,line_search_fn=None) # Optimizing using L-BFGS
            # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4,line_search_fn="strong_wolfe") # Optimizing using L-BFGS 1
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=40,line_search_fn="strong_wolfe") # Optimizing using L-BFGS 3
        elif (self.opti_DIP == 'SGD'):
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) # Optimizing using SGD
        elif (self.opti_DIP == 'Adadelta'):
            optimizer = torch.optim.Adadelta(self.parameters()) # Optimizing using Adadelta
        return optimizer

    #保存照片
    def write_current_img(self,out):
        if (self.all_images_DIP == "False"):
            if ((self.current_epoch%(self.sub_iter_DIP // 10) == (self.sub_iter_DIP // 10) -1)):
                self.write_current_img_task(out)
        elif (self.all_images_DIP == "True"):
            self.write_current_img_task(out)
        elif (self.all_images_DIP == "Last"):
            if (self.current_epoch == self.sub_iter_DIP - 1):
                self.write_current_img_task(out)

    def write_current_img_task(self,out):
        try:
            out_np = out.detach().numpy()[0,0,:,:]
        except:
            out_np = out.cpu().detach().numpy()[0,0,:,:]
        
        out_np = descale_imag(out_np,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt)


        np.save(self.path + '/1111.npy',out_np)
        #self.save_img(out_np, self.path + '/output_' + format(self.suffix) + '_' + format(self.current_epoch) + '.img')#self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch + self.last_iter) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
                            
    def suffix_func(self,config,hyperparameters_list,NNEPPS=False):
        config_copy = dict(config)
        if (NNEPPS==False):
            config_copy.pop('NNEPPS',None)
        #config_copy.pop('nb_outer_iteration',None)
        suffix = "config"
        for key, value in config_copy.items():
            if key in hyperparameters_list:
                suffix +=  "_" + key[:min(len(key),5)] + "=" + str(value)
        return suffix

    def save_img(self,img,name):
        fp=open(name,'wb')
        img.tofile(fp)
        print('Succesfully save in:', name)
        
    def initialize_WMV(self,config,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root):
        self.classWMV = iWMV(config)            
        self.classWMV.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.classWMV.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.classWMV.scaling_input = scaling_input
        self.classWMV.suffix = suffix
        self.classWMV.global_it = global_it
        # Initialize variables
        self.classWMV.initializeSpecific(config,root)
        
    def run_WMV(self,out,config,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root):
        if (self.DIP_early_stopping):
            self.SUCCESS = self.classWMV.SUCCESS
            self.log("SUCCESS", int(self.classWMV.SUCCESS))

            self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate = self.classWMV.WMV(out.detach().numpy(),self.current_epoch,self.classWMV.queueQ,self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate)
            self.VAR_recon = self.classWMV.VAR_recon
            self.MSE_WMV = self.classWMV.MSE_WMV
            self.PSNR_WMV = self.classWMV.PSNR_WMV
            self.SSIM_WMV = self.classWMV.SSIM_WMV
            self.epochStar = self.classWMV.epochStar
            '''
            if self.EMV_or_WMV == "EMV":
                self.alpha_EMV = self.classWMV.alpha_EMV
            else:
                self.windowSize = self.classWMV.windowSize
            '''
            self.patienceNumber = self.classWMV.patienceNumber

            if self.SUCCESS:
                print("SUCCESS WMVVVVVVVVVVVVVVVVVV")
                self.initialize_WMV(config,param1_scale_im_corrupt,param2_scale_im_corrupt,scaling_input,suffix,global_it,root)
        
        else:
            self.log("SUCCESS", int(False))
    

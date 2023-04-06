import torch
import torch.nn as nn
import pytorch_lightning as pl

from numpy import inf

import os

# Local files to import
from iWMV import iWMV

class DIP_3D(pl.LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, scaling_input, config, root, subroot, method, all_images_DIP, global_it, fixed_hyperparameters_list, hyperparameters_list, debug, suffix):
        super().__init__()

        #'''
        # Set random seed if asked (for NN weights here)
        if (os.path.isfile(root + "/seed.txt")): # Put root for path because raytune path !!!
            with open(root + "/seed.txt", 'r') as file:
                random_seed = file.read().rstrip()
            if (eval(random_seed)):
                pl.seed_everything(1)

        #'''
        
        # Defining variables from config        
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.skip = config['skip_connections']
        self.method = method
        self.all_images_DIP = all_images_DIP
        self.global_it = global_it
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.subroot = subroot
        self.config = config
        self.experiment = config["experiment"]
        '''
        ## Variables for WMV ##
        self.queueQ = []
        self.VAR_min = inf
        self.SUCCESS = False
        self.stagnate = 0
        '''
        self.DIP_early_stopping = config["DIP_early_stopping"]
        self.classWMV = iWMV(config)
        if(self.DIP_early_stopping):
            
            self.classWMV.fixed_hyperparameters_list = fixed_hyperparameters_list
            self.classWMV.hyperparameters_list = hyperparameters_list
            self.classWMV.debug = debug
            self.classWMV.param1_scale_im_corrupt = param1_scale_im_corrupt
            self.classWMV.param2_scale_im_corrupt = param2_scale_im_corrupt
            self.classWMV.scaling_input = scaling_input
            self.classWMV.suffix = suffix
            self.classWMV.global_it = global_it
            # Initialize variables
            self.classWMV.do_everything(config,root)

        self.write_current_img_mode = True
        #self.suffix = self.suffix_func(config,hyperparameters_list)
        #if (config["task"] == "post_reco"):
        #    self.suffix = config["task"] + ' ' + self.suffix
        self.suffix = suffix
        
        '''
        if (config['mlem_sequence'] is None):
            self.write_current_img_mode = True
            self.suffix = self.suffix_func(config)
        else:
            self.write_current_img_mode = False
        '''
        # Defining CNN variables
        L_relu = 0.2
        num_channel = [16, 32, 64, 128]
        pad = [0, 0]

        # Layers in CNN architecture
        self.deep1 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(1, num_channel[0], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[0], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.down1 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[0], 3, stride=(2, 2, 2), padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.deep2 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[1], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.down2 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], 3, stride=(2, 2, 2), padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.deep3 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.down3 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], 3, stride=(2, 2, 2), padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.deep4 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[3], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[3]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[3], num_channel[3], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[3]),
                                   nn.LeakyReLU(L_relu))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                                 nn.ReplicationPad3d(1),
                                 nn.Conv3d(num_channel[3], num_channel[2], 3, stride=1, padding=pad[0]),
                                 nn.BatchNorm3d(num_channel[2]),
                                 nn.LeakyReLU(L_relu))

        self.deep5 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                                 nn.ReplicationPad3d(1),
                                 nn.Conv3d(num_channel[2], num_channel[1], 3, stride=1, padding=pad[0]),
                                 nn.BatchNorm3d(num_channel[1]),
                                 nn.LeakyReLU(L_relu))

        self.deep6 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], (3, 3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                                 nn.ReplicationPad3d(1),
                                 nn.Conv3d(num_channel[1], num_channel[0], 3, stride=1, padding=pad[0]),
                                 nn.BatchNorm3d(num_channel[0]),
                                 nn.LeakyReLU(L_relu))

        self.deep7 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[0], (3, 3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], 1, (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(1))

        self.positivity = nn.ReLU() # Final ReLU to enforce positivity of ouput image
        # self.positivity = nn.SiLU() # Final SiLU, smoother than ReLU but not positive
        # self.positivity = nn.Softplus() # Final SiLU to enforce positivity of ouput image, smoother than ReLU

    def forward(self, x):

        # Pad if 3D dimension not divisible by 2^3
        original_x_dim = x.shape[2]
        if (x.shape[2] % 8 > 0):
            unpad_half_size = int((8 - original_x_dim % 8) / 2)
            x = nn.ReplicationPad3d((0,0,0,0,unpad_half_size,8 - original_x_dim % 8 - unpad_half_size))(x)
        # Encoder
        out1 = self.deep1(x)
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)

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

        # Unpad if original 3D dimension was not divisible by 2^3
        if (original_x_dim % 8 > 0):
            out = out[:,:,unpad_half_size:original_x_dim + unpad_half_size,:,:]

        if (self.method == 'Gong'):
            out = self.positivity(out)

        return out

    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD

    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        '''
        import matplotlib.pyplot as plt
        plt.imshow(image_corrupt_torch.cpu().detach().numpy()[0,0,:,:,:][30,:,:],cmap='gray')
        plt.colorbar()
        plt.show()
        '''
        out = self.forward(image_net_input_torch)
        # Save image over epochs
        if (self.write_current_img_mode):
            self.write_current_img(out)

        loss = self.DIP_loss(out, image_corrupt_torch)
        # logging using tensorboard logger
        self.logger.experiment.add_scalar('loss', loss,self.current_epoch)        

        # WMV
        self.log("SUCCESS", int(self.classWMV.SUCCESS))
        if (self.DIP_early_stopping):
            self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate = self.classWMV.WMV(out.cpu().detach().numpy(),self.current_epoch,self.classWMV.queueQ,self.classWMV.SUCCESS,self.classWMV.VAR_min,self.classWMV.stagnate)
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
            self.SUCCESS = self.classWMV.SUCCESS

            if self.SUCCESS:
                print("SUCCESS WMVVVVVVVVVVVVVVVVVV")
        
        return loss

    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """

        if (self.opti_DIP == 'Adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
            #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) # Optimizing using SGD
        elif (self.opti_DIP == 'LBFGS' or self.opti_DIP is None): # None means no argument was given in command line
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4) # Optimizing using L-BFGS
        elif (self.opti_DIP == 'SGD'):
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) # Optimizing using SGD
        return optimizer

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
            out_np = out.detach().numpy()[0,0,:,:,:]
        except:
            out_np = out.cpu().detach().numpy()[0,0,:,:,:]

        '''
        import matplotlib.pyplot as plt
        plt.imshow(out.cpu().detach().numpy()[0,0,:,:,:][30,:,:],cmap='gray')
        plt.colorbar()
        plt.show()
        '''
        self.save_img(out_np, self.subroot+'Block2/' + self.suffix + '/out_cnn/' + format(self.experiment) + '/out_' + 'DIP' + format(self.global_it) + '_epoch=' + format(self.current_epoch) + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
                            
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

import torch
import torch.nn as nn
import pytorch_lightning as pl

from numpy import inf
import numpy as np

import matplotlib.pyplot as plt

import os

from skimage.metrics import peak_signal_noise_ratio,structural_similarity

from utils.pre_utils import *


# 论文附赠代码部分：
from torch.nn import Parameter

# 通用性
class MeanOnlyBatchNorm(nn.Module):
    
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.Tensor(num_features))
        self.bias.data.zero_()

    def forward(self, inp):
        size = list(inp.size())
        beta = self.bias.view(1, self.num_features, 1, 1)
        avg = torch.mean(inp.view(size[0], self.num_features, -1), dim=2)

        output = inp - avg.view(size[0], size[1], 1, 1)
        output = output + beta

        return output


def bn(num_features,mean_only=False):
    if mean_only:
        return MeanOnlyBatchNorm(num_features)
    else:
        return nn.BatchNorm2d(num_features)
    #return nn.BatchNorm2d(num_features)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, ln_lambda=2.0, name='weight'):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.ln_lambda = torch.tensor(ln_lambda)
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]

        _,w_svd,_ = torch.svd(w.view(height,-1).data, some=False, compute_uv=False)
        sigma = w_svd[0]
        sigma = torch.max(torch.ones_like(sigma),sigma/self.ln_lambda)
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



def get_kernel(kernel_width=5, sigma=0.5):

    kernel = np.zeros([kernel_width, kernel_width])
    center = (kernel_width + 1.)/2.
    sigma_sq =  sigma * sigma

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            di = (i - center)/2.
            dj = (j - center)/2.
            kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
            kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)

    kernel /= kernel.sum()

    return kernel

class gaussian(nn.Module):
    def __init__(self, n_planes,  kernel_width=5, sigma=0.5):
        super().__init__()
        self.n_planes = n_planes
        self.kernel = get_kernel(kernel_width=kernel_width,sigma=sigma)

        convolver = nn.ConvTranspose2d(n_planes, n_planes, kernel_size=5, stride=2, padding=2, output_padding=1, groups=n_planes)
        convolver.weight.data[:] = 0
        convolver.bias.data[:] = 0
        convolver.weight.requires_grad = False
        convolver.bias.requires_grad = False

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            convolver.weight.data[i, 0] = kernel_torch
        
        self.upsampler_ = convolver

    def forward(self, x):
        x = self.upsampler_(x)
        return x
   
# 模型
import torch.nn.functional as F

# 仿照上面的自定义一套conv， bn， up
def Conv(input_channels, output_channels,kernel_size=3, ln_lambda=2, stride=1, bias=True, pad='Replication'):
    """ 
        定义两种类型:
        1. ln_lamdba>0, lipschiz-controlled conv
        2. ln_lamdba=0, normale conv
    """
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'Replication':
        padder = nn.ReplicationPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=to_pad, bias=bias)

    if ln_lambda>0:
        convolver = SpectralNorm(convolver, ln_lambda)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)

def Up(upsample_mode,input_channels,sigma):
    """
        定义三种upsample：
        1. 一般deconv 一般转置卷积
        2. gaussian-controlled deconv 
        3. 非转置卷积 bilinear nearest
    """
    if upsample_mode == 'deconv':
        up= nn.ConvTranspose2d(input_channels, input_channels, 4, stride=2, padding=1)
    elif upsample_mode=='bilinear' or upsample_mode=='nearest':
        up = nn.Upsample(scale_factor=2, mode=upsample_mode,align_corners=False)
    elif upsample_mode == 'gaussian':
        up = gaussian(input_channels, kernel_width=5, sigma=sigma)
    else:
        assert False
        
    return up
# 测试不同的models
# 1. DIP_LG
# 2. DD_LG
# 3. Decoder_LG
class UnetUp(nn.Module):
    
    def __init__(self,model_name, input_channel,output_channel,upsample_mode='bilinear', ln_lambda = 0, sigma=None):
        super().__init__()
        
        if model_name == 'DIP':
            L_relu = 0.2
            self.up = nn.Sequential(Up(upsample_mode,input_channel,sigma),
                                    Conv(input_channel, output_channel, kernel_size = 3, stride= 1, ln_lambda=ln_lambda, bias=True, pad='Replication'),
                                    bn(output_channel,mean_only=(ln_lambda>0)),
                                    nn.LeakyReLU(L_relu))
        elif model_name == 'DD':
            self.up = nn.Sequential(Up(upsample_mode,input_channel,sigma),
                                    nn.ReLU(),
                                    bn(output_channel,mean_only=(ln_lambda>0)))
        else:
            assert False
    def forward(self,x):
        return self.up(x)
class UnetConv(nn.Module):
    def __init__(self,model_name,input_channel,output_channel,ln_lambda=0):
        super().__init__()
        
        if model_name == 'DIP':
            L_relu = 0.2
            self.conv=nn.Sequential(Conv(input_channel, output_channel, kernel_size = 3, stride= 1, ln_lambda=ln_lambda, bias=True, pad='Replication'),
                                    bn(output_channel,mean_only=(ln_lambda>0)),
                                    nn.LeakyReLU(L_relu),
                                    Conv(input_channel, output_channel, kernel_size = 3, stride= 1, ln_lambda=ln_lambda, bias=True, pad='Replication'),
                                    bn(output_channel,mean_only=(ln_lambda>0)),
                                    nn.LeakyReLU(L_relu))
        elif model_name == 'DD':
            self.conv =nn.Conv2d(input_channel, input_channel, 1, stride=1)
        
        else:
            assert False
    def forward(self,x):
        return self.conv(x)

class module_test(pl.LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, 
                 config, suffix):
        super().__init__()
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.path="/home/xzhang/Documents/我的模型/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.num_layers = config['num_layers']
        self.num_channels = config['num_channels']
        
        self.initialize_network()
        
        
    def initialize_network(self):
        if self.model_name == 'DIP' :
            L_relu = 0.2
            sigmas = [0.1,0.1,0.5]

            # Layers in CNN architecture,定义各个层
            self.decoder_layers = nn.ModuleList([])
            if self.model_name == 'DIP' :
                for i in range(self.num_layers):
                    print(self.num_channels[self.num_layers-i])
                    self.decoder_layers.append(UnetUp(self.model_name,input_channel = self.num_channels[self.num_layers-i],output_channel =self.num_channels[self.num_layers-i-1],upsample_mode=self.config['upsampling_mode'], ln_lambda = self.config['ln_lambda'], sigma=self.config['sigma']))
                    self.decoder_layers.append(UnetConv(self.model_name,input_channel=self.num_channels[self.num_layers-i-1],output_channel=self.num_channels[self.num_layers-i-1],ln_lambda= self.config['ln_lambda']))
                self.decoder_layers.append(UnetUp(self.model_name,input_channel = self.num_channels[self.num_layers-i],output_channel =self.num_channels[self.num_layers-i-1],upsample_mode=self.config['upsampling_mode'], ln_lambda = self.config['ln_lambda'], sigma=self.config['sigma']))
                self.decoder_layers.append(nn.Sequential(
                                                        Conv(self.num_channels[0], self.num_channels[0], kernel_size = 3, stride= 1, ln_lambda= self.config['ln_lambda'] ,bias=True, pad='Replication'),
                                                        bn(self.num_channels[0],mean_only=(self.config['ln_lambda']>0)),
                                                        nn.LeakyReLU(L_relu),
                                                        Conv(self.num_channels[0], 1, kernel_size = 3, stride= 1, ln_lambda= self.config['ln_lambda'], bias=True, pad='Replication'),
                                                        bn(self.num_channels[0],mean_only=(self.config['ln_lambda']>0)),
                                                        nn.LeakyReLU(L_relu)))
                
                
                
                (UnetConv(self.model_name,input_channel=self.num_channels[self.num_layers-i-1],output_channel=self.num_channels[self.num_layers-i-1],ln_lambda= self.config['ln_lambda']))
            elif self.model_name == 'DD':
                for i in range(self.num_layers):
                    self.decoder_layers.append(UnetConv(self.model_name,input_channel=self.num_channels[self.num_layers-i],output_channel=self.num_channels[self.num_layers-i-1],ln_lambda= self.config['ln_lambda']))
                    self.decoder_layers.append(UnetUp(self.model_name,input_channel= self.num_channels[self.num_layers-i],output_channel=self.num_channels[self.num_layers-i-1],upsample_mode=self.config['upsampling_mode'], ln_lambda = self.config['ln_lambda'], sigma=self.config['sigma']))
                self.decoder_layers.append(nn.Conv2d(self.num_channels[0],1, 1, stride=1))
            self.positivity = nn.ReLU()
                
            
        s  = sum([np.prod(list(p.size())) for p in self.parameters()]); 
        print ('Number of params: %d' % s)
        
    def forward(self, x):
        out = x
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out

    # 定义损失函数为输出和含噪图像的tensor的mse
    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD
        
    # 定义训练流程，计算一次前向传播返回loss，（中间有logger记录tensorboard和early stopping）
    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
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

        
        os.makedirs(self.path + format(self.suffix)  + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
                
        return loss
    
    #配置优化器，可以选择各种优化器
    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) 
        return optimizer



class DIP_LG(pl.LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, 
                 config, suffix):
        super().__init__()
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.path="/home/xzhang/Documents/我的模型/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.num_layers = config['num_layers']
        self.num_channels = config['num_channels']

        self.initialize_network()
        
        
    def initialize_network(self):

        L_relu = 0.2
        num_channel = [16, 32, 64,128]
        sigmas = [0.1,0.1,0.5]
            
        self.up1 = nn.Sequential(   gaussian(num_channel[3],kernel_width=5,sigma=sigmas[0]),
                                    Conv(num_channel[3], num_channel[2], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication'),
                                    bn(num_channel[2]),
                                    nn.LeakyReLU(L_relu))

        self.deep5 = nn.Sequential( Conv(num_channel[2], num_channel[2], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication')   ,                          
                                    bn(num_channel[2]),
                                    nn.LeakyReLU(L_relu),
                                    Conv(num_channel[2], num_channel[2], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication')      ,                       
                                    bn(num_channel[2]),
                                    nn.LeakyReLU(L_relu))

        self.up2 = nn.Sequential(   gaussian(num_channel[2],kernel_width=5,sigma=sigmas[1]),
                                    Conv(num_channel[2], num_channel[1], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication'),
                                    bn(num_channel[1]),
                                    nn.LeakyReLU(L_relu))

        self.deep6 = nn.Sequential( Conv(num_channel[1], num_channel[1], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication')     ,                        
                                    bn(num_channel[1]),
                                    nn.LeakyReLU(L_relu),
                                    Conv(num_channel[1], num_channel[1], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication')    ,                         
                                    bn(num_channel[1]),
                                    nn.LeakyReLU(L_relu))

        self.up3 = nn.Sequential(   gaussian(num_channel[1],kernel_width=5,sigma=sigmas[2]),
                                    Conv(num_channel[1], num_channel[0], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication'),
                                    bn(num_channel[0]),
                                    nn.LeakyReLU(L_relu))

        self.deep7 = nn.Sequential( Conv(num_channel[0], num_channel[0], kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication')   ,                          
                                    bn(num_channel[0]),
                                    nn.LeakyReLU(L_relu),
                                    Conv(num_channel[0], 1, kernel_size = 3, stride= 1, ln_lambda=2, bias=True, pad='Replication'),
                                    bn(1))
        self.positivity = nn.ReLU() 

        
    def forward(self, x):
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
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD
        
    # 定义训练流程，计算一次前向传播返回loss，（中间有logger记录tensorboard和early stopping）
    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
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

        print(self.path + format(self.suffix)  + '/train_' + str(self.repeat))
        os.makedirs(self.path + format(self.suffix)  + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
                
        return loss
    
    #配置优化器，可以选择各种优化器
    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) 
        return optimizer
    
    
class DD_LG(pl.LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, 
                 config, suffix):
        super().__init__()
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.path="/home/xzhang/Documents/我的模型/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.num_layers = config['num_layers']
        self.num_channels = config['num_channels']

        self.initialize_network()
        
        
    def initialize_network(self):
        d = self.config["d_DD"] # Number of layers
        k = self.config['k_DD'] # Number of channels, depending on how much noise we mant to remove. Small k = less noise, but less fit

        # Defining CNN variables
        self.num_channels_up = [k]*(d+1) + [1]
        self.decoder_layers = nn.ModuleList([])
        for i in range(len(self.num_channels_up)-2):       
            self.decoder_layers.append(nn.Sequential(
                               Conv(self.num_channels_up[i],self.num_channels_up[i+1],1,ln_lambda=2,stride=1,pad = 'zero'),
                               nn.Conv2d(self.num_channels_up[i], self.num_channels_up[i+1], 1, stride=1),
                               gaussian(self.num_channels_up[i+1],5,0.5),
                               nn.ReLU(),
                               bn(self.num_channels_up[i+1]))) 

        self.last_layers = nn.Sequential(Conv(self.num_channels_up[-2], self.num_channels_up[-1], 1, ln_lambda=2,stride=1,pad='zero'))
        
        self.positivity = nn.ReLU() 

    def forward(self, x):
        out = x
        for i in range(len(self.num_channels_up)-2):
            out = self.decoder_layers[i](out)
        out = self.last_layers(out)
        out = self.positivity(out)
        return out

    # 定义损失函数为输出和含噪图像的tensor的mse
    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD
        
    # 定义训练流程，计算一次前向传播返回loss，（中间有logger记录tensorboard和early stopping）
    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
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

        print(self.path + format(self.suffix)  + '/train_' + str(self.repeat))
        os.makedirs(self.path + format(self.suffix)  + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
                
        return loss
    
    #配置优化器，可以选择各种优化器
    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) 
        return optimizer
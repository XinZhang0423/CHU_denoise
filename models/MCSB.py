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


def bn(num_features):
    return MeanOnlyBatchNorm(num_features)
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

def conv(in_f, out_f, kernel_size=3, ln_lambda=2, stride=1, bias=True, pad='zero'):
    downsampler = None
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    nn.init.kaiming_uniform_(convolver.weight, a=0, mode='fan_in')
    if ln_lambda>0:
        convolver = SpectralNorm(convolver, ln_lambda)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

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
        super(gaussian, self).__init__()
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


class decoder(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear', 'gaussian']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, ln_lambda=2, 
                       upsample_mode='gaussian', pad='zero', need_sigmoid=True, need_bias=True):
        super(decoder, self).__init__()


        filters = [128, 128, 128, 128, 128]
        sigmas = [0.1,0.1,0.1,0.5,0.5]

        layers = []
        layers.append(unetConv2(num_input_channels, filters[0], ln_lambda, need_bias, pad))
        for i in range(len(filters)):
            layers.append(unetUp(filters[i], upsample_mode, ln_lambda, need_bias, pad, sigmas[i]))

        layers.append(conv(filters[-1], num_output_channels, 1, 0, bias=need_bias, pad=pad))
        if need_sigmoid: 
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, ln_lambda, need_bias, pad):
        super(unetConv2, self).__init__()

        self.conv1= nn.Sequential(conv(in_size, out_size, 3, ln_lambda, bias=need_bias, pad=pad),
                                   bn(out_size),
                                   nn.LeakyReLU(),)
        self.conv2= nn.Sequential(conv(out_size, out_size, 3, ln_lambda, bias=need_bias, pad=pad),
                                   bn(out_size),
                                   nn.LeakyReLU(),)
    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        return x


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, ln_lambda, need_bias, pad, sigma=None):
        super(unetUp, self).__init__()

        num_filt = out_size
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= conv(out_size, out_size, 3, ln_lambda, bias=need_bias, pad=pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.conv= unetConv2(out_size, out_size, ln_lambda, need_bias, pad)
        elif upsample_mode == 'gaussian':
            self.up = gaussian(out_size, kernel_width=5, sigma=sigma)
            self.conv= unetConv2(out_size, out_size,ln_lambda, need_bias, pad)
        else:
            assert False

    def forward(self, x):
        x= self.up(x)
        x = self.conv(x)

        return x


class MCSB(pl.LightningModule):

    def __init__(self, param1_scale_im_corrupt, param2_scale_im_corrupt, 
                 config, suffix):
        super().__init__()
        self.lr = config['lr']
        self.opti_DIP = config['opti_DIP']
        self.sub_iter_DIP = config['sub_iter_DIP']
        self.param1_scale_im_corrupt = param1_scale_im_corrupt
        self.param2_scale_im_corrupt = param2_scale_im_corrupt
        self.path="/home/xzhang/Documents/我的模型/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        self.mask = np.load(config['mask_path'])
        # 记录input位置
        self.input_path = config['input_path'].split('/')[-1].split('.')[0]   
        self.config = config
        self.skip = 0
        L_relu = 0.2
        num_channel = [16, 32, 64, 128]
        pad = [0, 0]

        # Layers in CNN architecture,定义各个层
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

        return out,encoder_output	

    # 定义损失函数为输出和含噪图像的tensor的mse
    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD
        
    # 定义训练流程，计算一次前向传播返回loss，（中间有logger记录tensorboard和early stopping）
    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out,encoder_output = self.forward(image_net_input_torch)

        loss = self.DIP_loss(out, image_corrupt_torch)
        
        try:
            out_np = out.detach().numpy()
            encoder_output_np = encoder_output.detach().numpy()
        except:
            out_np = out.cpu().detach().numpy()
            encoder_output_np = encoder_output.cpu().detach().numpy()
        # 256,256 numpy
        out_np = np.squeeze(out_np)
        # 256，256 原来的 和 out_np 256,256 原来的
        out_np = destand_numpy_imag(out_np,self.param1_scale_im_corrupt,self.param2_scale_im_corrupt)

        
        os.makedirs(self.path + format(self.suffix) +'/' + self.input_path + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) +'/' +  self.input_path + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)
        
        os.makedirs(self.path + format(self.suffix) +'/encoder_output/' + self.input_path + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) +'/encoder_output/' +  self.input_path + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',encoder_output_np)
        
        return loss
    
    #配置优化器，可以选择各种优化器
    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) 
        return optimizer


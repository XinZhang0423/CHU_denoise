# 我在做各种实验的时候发现，每次基本上都在重复写一样的内容，所以为了简化这个做实验的过程
# 我决定将重复的内容放到一起，然后再每次实验调用一下，改个参数就好了
import os
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from functools import partial
import datetime
# 将当前路径给解释器，这样解释器在搜索module的时候就会从当前路径上找
import sys
sys.path.append('/home/xzhang/Documents/我的模型/')

from models.DD_2D import DD_2D
from models.DIP_2D import DIP_2D
from models.DIP_decoder import DIP_decoder
from models.DIP_2D_modified import DIP_2D_modified
from config.config_tune import *
from config.config import config
from config.config_input import config_input
from config.config_target import config_target

from utils.pre_utils import *

def main_dd_grid_search(config = config,
         path_ground_truth="data/ground_truth/ground_truth_padded.npy",
         path_target="data/corrupted_images/target_padded.npy",
         log_dir = '/logs/test_DD',
         suffix = 'test_DD'):
    # 调整工作路经
    current_path = os.getcwd()
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    
    layers = config['d_DD']
    channels = config['k_DD']
    repeat = config['repeat']
    size = (int(128/(2**layers)),int(128/(2**layers)),channels)
    print(layers,channels,repeat)

    image_net_input = np.random.uniform(low=0, high=1, size=size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,channels,size[0],size[1],1)
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    # 标准化
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,128,128,1)
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    
    model = DD_2D(config, param1_scale_im_corrupt, param2_scale_im_corrupt,ground_truth,suffix=f'{suffix}/{layers}_{channels}_{repeat}')
    
    #定义tensorboard
    checkpoint_simple_path = os.getcwd() + log_dir
    name=str(datetime.datetime.now())
    # print(checkpoint_simple_path)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)

def main_dip_decoder(config = config,
         path_ground_truth="data/ground_truth/ground_truth_padded.npy",
         path_target="data/corrupted_images/target_padded.npy",
         log_dir = '/logs/test_DD',
         suffix = 'test_DD'):
    # 调整工作路经
    current_path = os.getcwd()
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    
    layers = 3
    channels = 128
    repeat = config['repeat']
    size = (int(128/(2**layers)),int(128/(2**layers)),channels)
    print(layers,channels,repeat)

    image_net_input = np.random.uniform(low=0, high=1, size=size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,channels,size[0],size[1],1)
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    # 标准化
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,128,128,1)
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    
    model = DIP_decoder(param1_scale_im_corrupt, param2_scale_im_corrupt,config,f'{suffix}/{layers}_{channels}_{repeat}',ground_truth)

    checkpoint_simple_path = os.getcwd() + log_dir
    name=str(datetime.datetime.now())
    # print(checkpoint_simple_path)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)
    
def main_encoder_decoder(config = config,
         path_ground_truth="data/ground_truth/ground_truth_padded.npy",
         path_target="data/corrupted_images/target_padded.npy",
         log_dir = '/logs/test_DD',
         suffix = 'test_DD'):
    # 调整工作路经
    current_path = os.getcwd()
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    
    layers = 3
    channels = 128
    repeat = config['repeat']
    size = (int(128/(2**layers)),int(128/(2**layers)),channels)
    print(layers,channels,repeat)

    image_net_input = np.random.uniform(low=0, high=1, size=(128,128,1))# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,1,128,128,1)
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    # 标准化
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,128,128,1)
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    
    model = DIP_2D_modified(param1_scale_im_corrupt, param2_scale_im_corrupt,config,f'{suffix}/{layers}_{channels}_{repeat}')
    checkpoint_simple_path = os.getcwd() + log_dir
    name=str(datetime.datetime.now())
    # print(checkpoint_simple_path)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)
    
# 用于平行跑DIP
def main(config = config,
         path_input="data/noisy_images/uniform_noise3.npy",
         path_ground_truth="data/ground_truth/ground_truth.npy",
         path_target="data/corrupted_images/BSREM_it30.npy",
         log_dir = '/logs/test2_logs',
         suffix = 'aaa',
         init_weights = False
         ):
    # 调整工作路经
    current_path = os.getcwd()
    print(current_path)
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    print(f'current working directory switched to {current_path}')
    
    image_net_input = np.load(path_input) # 112*112*1
    ground_truth = np.load(path_ground_truth) # 112*112*1
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, config,'data/Algo/',
                "nested",all_images_DIP="False",global_it=-100, suffix=suffix,last_iter=-1,ground_truth=ground_truth,target=image_corrupt,initial_param='kaiming_norm')
    
    if init_weights :
        model.init_weights()

    #定义tensorboard
    checkpoint_simple_path = os.getcwd() + log_dir
    # experiment = 24
    name=str(datetime.datetime.now())
    # print(checkpoint_simple_path)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)
    
    # return (model.psnr_list, model.mse_gt_list)

#用于平行测试不同的输入DIP
def main_input(config = config_input,
         path_input="data/noisy_images/uniform_noise3.npy",
         path_ground_truth="data/ground_truth/ground_truth.npy",
         path_target="data/corrupted_images/BSREM_it30.npy",
         log_dir = '/logs/test2_logs',
         suffix = 'test8',
         init_weights = False
         ):
    # 调整工作路经
    current_path = os.getcwd()
    print(current_path)
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    print(f'current working directory switched to {current_path}')
    
    image_net_input = np.load(config['input_path']) # 112*112*1
    ground_truth = np.load(path_ground_truth) # 112*112*1
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
   
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, config,'data/Algo/',
                "nested",all_images_DIP="False",global_it=-100, suffix=suffix,last_iter=-1,ground_truth=ground_truth,target=image_corrupt,initial_param='kaiming_norm')
    
    if init_weights :
        model.init_weights()

    #定义tensorboard
    checkpoint_simple_path = os.getcwd() + log_dir
    # experiment = 24
    name=str(datetime.datetime.now())

    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)
    
    #return (model.psnr_list, model.mse_gt_list)

#用于并行测试不同的target
def main_target(config = config_target,
         path_input="data/noisy_images/uniform_noise3.npy",
         path_ground_truth="data/ground_truth/ground_truth.npy",
         path_target="data/corrupted_images/BSREM_it30.npy",
         log_dir ='/logs/test2_logs',
         suffix = 'aaa',
         init_weights = False
         ):
    # 调整工作路经
    current_path = os.getcwd()
    print(current_path)
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    print(f'current working directory switched to {current_path}')
    
    image_net_input = np.load(path_input) # 112*112*1
    ground_truth = np.load(path_ground_truth) # 112*112*1
    image_corrupt = fijii_np(config['target_path'],(112,112,1)) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, config,'data/Algo/',
                "nested",all_images_DIP="False",global_it=-100, suffix=suffix,last_iter=-1,ground_truth=ground_truth,target=image_corrupt,initial_param='kaiming_norm')
    
    if init_weights :
        model.init_weights()

    #定义tensorboard
    checkpoint_simple_path = os.getcwd() + log_dir
    # experiment = 24
    name=str(datetime.datetime.now())
    # print(checkpoint_simple_path)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)
    
    # return (model.psnr_list, model.mse_gt_list)

def save_csv(filename,psnr_list,mse_gt_list):
    with open(filename,'w',newline='') as csvfile:
        writer =csv.writer(csvfile)
        
        header = ['iters','psnr','mse_gt']
        writer.writerow(header)
        
        for i in range(len(psnr_list)):
            row = [i ] + [psnr_list[i]] + [mse_gt_list[i]]
            writer.writerow(row)
                    
def read_csv(filename):
    tb=pd.read_csv(filename)    
    print(tb.head())
    print(tb.describe())
    return tb

def write_csv(my_dict, file_name):
    nb_cols = len(list(my_dict.keys()))
    nb_rows = len(my_dict[f"{1}"])
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        header = ['iters'] + list(range(nb_cols))
        writer.writerow(header)
        
        # 写入数据
        for i in range(nb_rows):
            row = [i] + [my_dict[f"{j}"][i] for j in range(nb_cols)]
            writer.writerow(row)
 
#用于平行测试早停           
def main_es(config = config,
         path_input="data/noisy_images/uniform_noise3.npy",
         path_ground_truth="data/ground_truth/ground_truth.npy",
         path_target="data/corrupted_images/BSREM_it30.npy",
         log_dir = '/logs/test2_logs',
         suffix = 'aaa',
         init_weights = False
         ):
    # 调整工作路经
    current_path = os.getcwd()
    print(current_path)
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    print(f'current working directory switched to {current_path}')
    
    image_net_input = np.load(path_input) # 112*112*1
    ground_truth = np.load(path_ground_truth) # 112*112*1
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, config,'data/Algo/',
                "nested",all_images_DIP="False",global_it=-100, suffix=suffix,last_iter=-1,ground_truth=ground_truth,target=image_corrupt,initial_param='kaiming_norm')

    #定义tensorboard
    checkpoint_simple_path = os.getcwd() + log_dir
    # experiment = 24
    name=str(datetime.datetime.now())
    # print(checkpoint_simple_path)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stopping_callback = EarlyStopping(monitor="SUCCESS", mode="max",stopping_threshold=0.9,patience=np.inf) # SUCCESS will be 1 when ES if found, which is greater than stopping_threshold = 0.9
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger,callbacks=[early_stopping_callback])
    # 训练模型
    trainer.fit(model, train_dataloader)


#用于测试Deep decoder (从现在起以后都是不固定随机种子，且不固定输入)
def main_dd(config = config,
         path_input="/home/xzhang/Documents/我的模型/data/noisy_images/uniform_noise_7*7_0.npy",
         path_ground_truth="data/ground_truth/ground_truth.npy",
         path_target="data/corrupted_images/BSREM_it30.npy",
         log_dir = '/logs/test_DD',
         suffix = 'test_DD',
         ):
    # 调整工作路经
    current_path = os.getcwd()
    print(current_path)
    os.chdir("/home/xzhang/Documents/我的模型/")
    current_path = os.getcwd()
    print(f'current working directory switched to {current_path}')
    
    image_net_input = np.load(config['input_path']) # 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,128,7,7,1)
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    # 标准化
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"normalization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    model = DD_2D(config, param1_scale_im_corrupt, param2_scale_im_corrupt,ground_truth,suffix)
    
    #定义tensorboard
    checkpoint_simple_path = os.getcwd() + log_dir
    name=str(datetime.datetime.now())
    # print(checkpoint_simple_path)
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)



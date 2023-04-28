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


from models.DIP_2D import DIP_2D
from config.config_tune import *
from config.config import config
from utils.pre_utils import *



def main(path_input="data/noisy_images/uniform_noise.npy",
         path_ground_truth="data/ground_truth/ground_truth.npy",
         path_target="data/corrupted_images/BSREM_it30.npy",
         log_dir = '/test_log'):
    # 调整工作路经
    current_path = os.getcwd()
    print(current_path)
    os.chdir("..")
    current_path = os.getcwd()
    print(f'current working directory switched to {current_path}')
    
    
    image_net_input = np.load(path_input) # 112*112*1
    ground_truth = np.load(path_ground_truth) # 112*112*1
    image_corrupt = np.load(path_target) # 112*112*1
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization")
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
    
    ground_truth = np.load(path_ground_truth)    
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, config,'data/Algo/',
                "nested",all_images_DIP="False",global_it=-100, suffix='aaa',last_iter=-1,ground_truth=ground_truth,target=image_corrupt,initial_param='kaiming_norm')
    # model.init_weights()

    #定义tensorboard
    checkpoint_simple_path = os.getcwd() + log_dir
    
    # experiment = 24
    name=str(datetime.datetime.now())

    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path,name=name)
    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)
    # 训练模型
    trainer.fit(model, train_dataloader)
    
    return (model.psnr_list, model.mse_gt_list)

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
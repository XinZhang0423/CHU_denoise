
# 模型相关
import torch
from models.DIP_2D import DIP_2D
import pytorch_lightning as pl
import numpy as np

# 画图相关
import matplotlib.pyplot as plt

# 文件读写相关
import csv
from config import config
import os
import pandas as pd
import csv

# 自定义函数
from utils.pre_utils import *


# 定义各种文件的路径 
path_noisy="/home/xzhang/Documents/我的模型/src/BSREM_it30.img" # 含噪图片位置
path_output = "/home/xzhang/Documents/我的模型/output"            # 输出图片位置
PETImage_shape=(112,112,1)  # 输入图片的大小
path_input = "/home/xzhang/Documents/我的模型/images/noise_images/" # 输入图片位置

# 读取不同的含噪声文件，目前只有一个，将其做rescale和格式转换tensor 1,1,112,112
image_corrupt=fijii_np(path_noisy,PETImage_shape) # 读取图片并将图片转换成numpy array
image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization") # 标准化图片, 减去平均值，除以标准差，参数1是mean，参数2是std
image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]

# 定义一个字典，用于保存每个npy文件的损失值、PSNR和SSIM值
result=dict()

ground_truth=fijii_np("/home/xzhang/Documents/我的模型/output/image4_0.raw",PETImage_shape)

# 读取不同的噪声文件，用于输入。并且同样经过rescale处理
for npy in os.listdir("images/noise_images/"):
    
    # 读取对应文件
    image_net_input = np.load(os.path.join("images/noise_images/",npy))
    image_net_input_torch = torch.Tensor(image_net_input)

    image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]

    image_corrupt_input_scale,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization") 

    # 用dataset和dataloader 读取训练数据 input和目标
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 读取并导入模型
    model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt,config,'data/Algo/',
                "nested",all_images_DIP="False",global_it=-100, suffix="suffix", last_iter=-1,ground_truth=ground_truth)
    model_class = DIP_2D

    # 设置log路径保存在runs目录
    checkpoint_simple_path = 'runs/'
    experiment=24
    name='my_model'

    # 创建TensorBoardLogger，注意savedir和后面打开tensorboard时的路径要一致，versions是啥？
    logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name)

    trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")
    trainer.fit(model, train_dataloader)
    out = model(image_net_input_torch)

    # 在训练完成后，使用logger读取日志文件并获取记录的值
    loss= logger.experiment.get_scalar('loss')
    psnr = logger.experiment.get_scalar('psnr')
    ssim = logger.experiment.get_scalar('ssim')
    
    name = "DIP_2D_" + npy.split(".")[0] 
    
    result[name] = {'loss': loss, 'psnr': psnr, 'ssim': ssim}
    
# 写到.csv文件中
csv_dir = '/home/xzhang/Documents/我的模型/csvfiles'
 
with open(os.path.join(csv_dir,'results.csv'),'w',newline='') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow('input',result[name].keys())
    for name,res in result.items(): 
        for epoch,(loss,psnr,ssim) in enumerate(zip(result[name]['loss'],result[name]['psnr'],result[name]['ssim'])):
            writer.writerow([name,epoch,loss,psnr,ssim])
        

# 读取.csv文件，并画图 
import pandas as pd 
import matplotlib.pyplot as plt

# 画出dip论文中的那种曲线图
def plot_results():
    df = pd.read_csv(os.path.join(path_output,"results.csv"))
    
    fig,axes=plt.subplots(nrows=3,ncols=1,figsize=(10,10))
    for name in df['input'].unique():
        data = df[df['input']==name]
        
        axes[0].plot(range(config["sub_iter_DIP"]),data['loss'],label='loss')
        axes[0].set_xscale('log')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Sub-Iters')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(range(config["sub_iter_DIP"]),data['psnr'],label='psnr')
        axes[1].set_xscale('log')
        axes[1].set_title('PSNR')
        axes[1].set_xlabel('Sub-Iters')
        axes[1].set_ylabel('PSNR')
        axes[1].legend()

        axes[2].plot(range(config["sub_iter_DIP"]),data['ssim'],lable='ssim')
        axes[2].set_xscale('log')
        axes[2].set_title('SSIM')
        axes[2].set_xlabel('Sub-Iters')
        axes[2].set_ylabel('SSIM')
        axes[2].legend()

    fig.tight_layout()
 
    plt.savefig(os.path.join(path_output,"results.png"))
    plt.show()
    
    
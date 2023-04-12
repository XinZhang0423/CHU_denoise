

# 模型相关
import torch
from models.DIP_2D import DIP_2D
import pytorch_lightning as pl
from config import *

# 数据处理相关
import numpy as np
import matplotlib.pyplot as plt

# 自定义函数
from utils.pre_utils import *

# 定义各种文件的路径 
path_noisy="/home/xzhang/Documents/我的模型/src/BSREM_it30.img" # 含噪图片位置
path_output = "/home/xzhang/Documents/我的模型/output_images"            # 输出图片位置
PETImage_shape=(112,112,1)  # 输入图片的大小
path_input = "/home/xzhang/Documents/我的模型/images/noise_images/" # 输入图片位置


# 读取不同的含噪声文件，目前只有一个，将其做rescale和格式转换tensor 1,1,112,112
image_corrupt=fijii_np(path_noisy,PETImage_shape) # 读取图片并将图片转换成numpy array
image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization") # 标准化图片, 减去平均值，除以标准差，参数1是mean，参数2是std
image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]

# 读取不同的噪声文件，用于输入。并且同样经过rescale处理
image_net_input =np.random.uniform(PETImage_shape)
image_net_input_torch = torch.rand(*PETImage_shape)

image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
image_net_input_torch = image_net_input_torch[:,:,:,:,0]

image_corrupt_input_scale,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization") 

# ground_truth = np.zeros(PETImage_shape)
ground_truth = np.load("/home/xzhang/Documents/我的模型/images/noise_images/image4_0.npy")

# 加载数据
train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

# 加载模型
model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, config,'data/Algo/',
               "nested",all_images_DIP="False",global_it=-100, suffix='aaa',last_iter=-1,ground_truth=ground_truth)
model_class = DIP_2D

# 定义tensorboard
checkpoint_simple_path = '/home/xzhang/Documents/我的模型/lightning_logs'
# experiment = 24
# name = 'my_model'

logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, )#version=format(experiment), name=name)
trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")

# 训练模型
trainer.fit(model, train_dataloader)
out = model(image_net_input_torch)


print(out.shape)
image_out = out.view(PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]).detach().numpy()
image_concat = np.concatenate((image_corrupt, destand_numpy_imag(image_out,param1_scale_im_corrupt,param2_scale_im_corrupt)), axis=1)
image_reversed =np.max(image_concat)-image_concat

plt.imshow(image_reversed, cmap='gray')
plt.show()  




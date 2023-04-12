


import torch
from models.DIP_2D import DIP_2D
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from config import config

def fijii_np(path,shape,type_im=None):
    """"Transforming raw data to numpy array"""
    if (type_im is None):
        if (config["FLTNB"]  == 'float'):
            type_im = '<f'
        elif (config["FLTNB"] == 'double'):
            type_im = '<d'

    try:
        file_path=(path)
        dtype = np.dtype(type_im)
        fid = open(file_path, 'rb')
        data = np.fromfile(fid,dtype)
        #'''
        if (1 in shape): # 2D
            image = data.reshape(shape)
        else: # 3D
            image = data.reshape(shape[::-1])
    except:
        type_im = ('<f')*(type_im=='<d') + ('<d')*(type_im=='<f')
        file_path=(path)
        dtype = np.dtype(type_im)
        fid = open(file_path, 'rb')
        data = np.fromfile(fid,dtype)
        #'''
        if (1 in shape): # 2D
            image = data.reshape(shape)
        else: # 3D
            image = data.reshape(shape[::-1])
    return image
    
def stand_imag(image_corrupt):
    print("staaaaaaaaaaand")
    """ Standardization of input - output with mean 0 and std 1 for each slide"""
    mean=np.mean(image_corrupt)
    std=np.std(image_corrupt)
    image_center = image_corrupt - mean
    if (std == 0.):
        raise ValueError("std 0")
    image_corrupt_std = image_center / std
    return image_corrupt_std,mean,std

def destand_numpy_imag(image, mean, std):
    """ Destandardization of input - output with mean 0 and std 1 for each slide"""
    return image * std + mean

def destand_imag(image, mean, std):
    image_np = image.detach().numpy()
    return destand_numpy_imag(image_np, mean, std)

def rescale_imag(image_corrupt, scaling):
    """ Scaling of input """
    if (scaling == 'standardization'):
        return stand_imag(image_corrupt)
    else: # No scaling required
        return image_corrupt, 0, 0

def descale_imag(self,image, param_scale1, param_scale2, scaling='standardization'):
    """ Descaling of input """
    try:
        image_np = image.detach().numpy()
    except:
        image_np = image
    if (scaling == 'standardization'):
        return destand_numpy_imag(image_np, param_scale1, param_scale2)
    else: # No scaling required
        return image_np
    
path_noisy="/home/xzhang/Documents/我的模型/src/BSREM_it30.img" # 含噪图片位置
path_output = "/home/xzhang/Documents/我的模型/output"            # 输出图片位置
PETImage_shape=(112,112,1)  # 输入图片的大小
path_input = "/home/xzhang/Documents/我的模型/images/noise_images/"
image_corrupt=fijii_np(path_noisy,PETImage_shape) # 读取图片并将图片转换成numpy array
image_corrupt_input_scaled,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization") # 标准化图片, 减去平均值，除以标准差，参数1是mean，参数2是std
image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]

PETImage_shape=(112,112,1)
image_net_input =np.random.uniform(PETImage_shape)
image_net_input_torch = torch.rand(*PETImage_shape)

image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
image_net_input_torch = image_net_input_torch[:,:,:,:,0]

image_corrupt_input_scale,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization") 

train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, 'standardization', config,'data/Algo/',
               path_output,"nested",all_images_DIP="False",global_it=-100, fixed_hyperparameters_list="for early stopping",
               hyperparameters_list="for early stopping", debug=config["debug"], suffix="suffix", last_iter=-1)
model_class = DIP_2D

checkpoint_simple_path = 'runs/'
experiment=24
name='my_model'

logger = pl.loggers.TensorBoardLogger(save_dir=checkpoint_simple_path, version=format(experiment), name=name)
trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1,logger=logger)#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")

trainer.fit(model, train_dataloader)
out = model(image_net_input_torch)

print(out.shape)
image_out = out.view(PETImage_shape[0],PETImage_shape[1],PETImage_shape[2]).detach().numpy()
image_concat = np.concatenate((image_corrupt, destand_numpy_imag(image_out,param1_scale_im_corrupt,param2_scale_im_corrupt)), axis=1)
image_reversed =np.max(image_concat)-image_concat

plt.imshow(image_reversed, cmap='gray')
plt.show()  




from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt

PETImage_shape=(112,112,1)

def fijii_np(path,shape,type_im=None):
    """"Transforming raw data to numpy array"""
    type_im = '<f'
    try:
        file_path=(path)
        dtype = np.dtype(type_im)
        fid = open(file_path, 'rb')
        data = np.fromfile(fid,dtype)
        
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
        if (1 in shape): 
            image = data.reshape(shape)
        else: 
            image = data.reshape(shape[::-1])
    return image

#下面是归一化和去归一化过程，这些都有公式的，只需要记住 norm是numpy， denorm是tensor就好，
#归一化norm_img将numpy array处理，返回处理好的图片和min，max
#去归一化demorm_img将torch tensor detach().numpy()转化为numpy array然后再去标准化
def norm_imag(img):
    print("nooooooooorm")
    """ Normalization of input - output [0..1] and the normalization value for each slide"""
    if (np.max(img) - np.min(img)) != 0:
        return (img - np.min(img)) / (np.max(img) - np.min(img)), np.min(img), np.max(img)
    else:
        return img, np.min(img), np.max(img)

def denorm_imag(image, mini, maxi):
    """ Denormalization of input - output [0..1] and the normalization value for each slide"""
    image_np = image.detach().numpy()
    return denorm_numpy_imag(image_np, mini, maxi)

def denorm_numpy_imag(img, mini, maxi):
    if (maxi - mini) != 0:
        return img * (maxi - mini) + mini
    else:
        return img
    
def norm_positive_imag(img):
    """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
    if (np.max(img) - np.min(img)) != 0:
        print(np.max(img))
        print(np.min(img))
        return img / np.max(img), 0, np.max(img)
    else:
        return img, 0, np.max(img)

def denorm_positive_imag(image, mini, maxi):
    """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
    image_np = image.detach().numpy()
    return denorm_numpy_imag(image_np, mini, maxi)

def denorm_numpy_positive_imag( img, mini, maxi):
    if (maxi - mini) != 0:
        return img * maxi 
    else:
        return img

#下面一样，是标准化过程和上面类似，只不过区别是需要mean和std
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

#rescale一下图片，也就是对图片进行一下预处理，包括标准化，归一化和positive归一化，我觉得这里可以改进的点在于引用sklearn或者opencv里面的函数
#根据scaling的模式选择处理方式，返回处理好的图片和 反向处理所需要的参数
def rescale_imag(image_corrupt, scaling):
    """ Scaling of input """
    if (scaling == 'standardization'):
        return stand_imag(image_corrupt)
    elif (scaling == 'normalization'):
        return norm_imag(image_corrupt)
    elif (scaling == 'positive_normalization'):
        return norm_positive_imag(image_corrupt)
    else: # No scaling required
        return image_corrupt, 0, 0
    
#直接对tensor进行反向处理
def descale_imag(image, param_scale1, param_scale2, scaling='standardization'):
    """ Descaling of input """
    try:
        image_np = image.detach().numpy()
    except:
        image_np = image
    if (scaling == 'standardization'):
        return destand_numpy_imag(image_np, param_scale1, param_scale2)
    elif (scaling == 'normalization'):
        return denorm_numpy_imag(image_np, param_scale1, param_scale2)
    elif (scaling == 'positive_normalization'):
        return denorm_numpy_positive_imag(image_np, param_scale1, param_scale2)
    else: # No scaling required
        return image_np



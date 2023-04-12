from config import config
import numpy as np



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

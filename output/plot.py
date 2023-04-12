

import matplotlib.pyplot as plt
import numpy as np


# 本文件只是用来看图片的，读取一个.img .raw 文件，并将其转化为numpy图片然后plot出来
def fijii_np(path,shape,type_im=None):
    """"Transforming raw data to numpy array"""
    type_im = '<f'

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

PETImage_shape=(112,112,1)  
image_corrupt=fijii_np("/home/xzhang/Documents/我的模型/output/image4_0.raw",PETImage_shape)


plt.imshow(image_corrupt, cmap='gray')
plt.show()  
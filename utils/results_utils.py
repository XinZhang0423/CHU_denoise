# Math
import numpy as np
import matplotlib.pyplot as plt

# Useful
import os
import csv
import pandas as pd
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def save_csv(filename,psnr_list,mse_gt_list):
    """_
        单次train,将psnr_list,mse_gt_list保存到csv文件中
    """
    with open(filename,'w',newline='') as csvfile:
        writer =csv.writer(csvfile)
        header = ['iters','psnr','mse_gt']
        writer.writerow(header)
        
        for i in range(len(psnr_list)):
            row = [i ] + [psnr_list[i]] + [mse_gt_list[i]]
            writer.writerow(row)
                     
def read_csv(filename):
    """_
        同样是单次train,将psnr_list,mse_gt_list从csv文件中读取出来
    """
    tb=pd.read_csv(filename)    
    print(tb.head())
    print(tb.describe())
    return tb


def images_to_csv(from_dir,to_file,corrupted_image,ground_truth,mask,metric='mse_gt'):
    trains = os.listdir(from_dir)
    res_list = list()
    for train in trains:
        iters = os.listdir(trains)
        for iter in train:
            image_np = np.load(os.path.join(from_dir,image))
            if metric == 'loss':
                res= np.mean((corrupted_image - image_np)**2)
            elif metric == 'mse_gt':
                res= np.mean((ground_truth*mask - image_np*mask)**2)
            elif metric == 'psnr':
                res = peak_signal_noise_ratio(ground_truth, image_np, data_range=np.amax(image_np)-np.amin(image_np))
            elif metric == 'ssim':
                res = structural_similarity(ground_truth, image_np, data_range=np.amax(image_np)-np.amin(image_np))
            res_list.append(res) 
        
        with open(to_file,'w',newline='') as csvfile:
            writer =csv.writer(csvfile)
            header = ['iters',metric]
            writer.writerow(header)
            
            for i in range(len(res_list)):
                row = [i ] + [res_list[i]]
                writer.writerow(row)
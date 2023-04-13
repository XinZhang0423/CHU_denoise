## Python libraries

# Useful
# ?
from pathlib import Path
import os
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import re
# 用于abc抽象类，不知道是干什么用的，就这样抄吧，先默认这样，在研究为什么
import abc

# vGeneral 主要有图像的预处理，以及写到tensorboard中的函数
class vGeneral(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self,config, *args, **kwargs):
        self.experiment = "not updated"
        self.config = config

    #将.img文件读取成numpy array,需要1.读取路径，2.图片shape，3.存储种类
    def fijii_np(self,path,shape,type_im=None):
        """"Transforming raw data to numpy array"""
        if (type_im is None):
            if (self.config["FLTNB"]  == 'float'):
                type_im = '<f'
            elif (self.config["FLTNB"] == 'double'):
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

    #下面是归一化和去归一化过程，这些都有公式的，只需要记住 norm是numpy， denorm是tensor就好，
    #归一化norm_img将numpy array处理，返回处理好的图片和min，max
    #去归一化demorm_img将torch tensor detach().numpy()转化为numpy array然后再去标准化
    def norm_imag(self,img):
        print("nooooooooorm")
        """ Normalization of input - output [0..1] and the normalization value for each slide"""
        if (np.max(img) - np.min(img)) != 0:
            return (img - np.min(img)) / (np.max(img) - np.min(img)), np.min(img), np.max(img)
        else:
            return img, np.min(img), np.max(img)

    def denorm_imag(self,image, mini, maxi):
        """ Denormalization of input - output [0..1] and the normalization value for each slide"""
        image_np = image.detach().numpy()
        return self.denorm_numpy_imag(image_np, mini, maxi)

    def denorm_numpy_imag(self,img, mini, maxi):
        if (maxi - mini) != 0:
            return img * (maxi - mini) + mini
        else:
            return img
        
    def norm_positive_imag(self,img):
        """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
        if (np.max(img) - np.min(img)) != 0:
            print(np.max(img))
            print(np.min(img))
            return img / np.max(img), 0, np.max(img)
        else:
            return img, 0, np.max(img)

    def denorm_positive_imag(self,image, mini, maxi):
        """ Positive normalization of input - output [0..1] and the normalization value for each slide"""
        image_np = image.detach().numpy()
        return self.denorm_numpy_imag(image_np, mini, maxi)

    def denorm_numpy_positive_imag(self, img, mini, maxi):
        if (maxi - mini) != 0:
            return img * maxi 
        else:
            return img

    #下面一样，是标准化过程和上面类似，只不过区别是需要mean和std
    def stand_imag(self,image_corrupt):
        print("staaaaaaaaaaand")
        """ Standardization of input - output with mean 0 and std 1 for each slide"""
        mean=np.mean(image_corrupt)
        std=np.std(image_corrupt)
        image_center = image_corrupt - mean
        if (std == 0.):
            raise ValueError("std 0")
        image_corrupt_std = image_center / std
        return image_corrupt_std,mean,std

    def destand_numpy_imag(self,image, mean, std):
        """ Destandardization of input - output with mean 0 and std 1 for each slide"""
        return image * std + mean

    def destand_imag(self,image, mean, std):
        image_np = image.detach().numpy()
        return self.destand_numpy_imag(image_np, mean, std)

    #rescale一下图片，也就是对图片进行一下预处理，包括标准化，归一化和positive归一化，我觉得这里可以改进的点在于引用sklearn或者opencv里面的函数
    #根据scaling的模式选择处理方式，返回处理好的图片和 反向处理所需要的参数
    def rescale_imag(self,image_corrupt, scaling):
        """ Scaling of input """
        if (scaling == 'standardization'):
            return self.stand_imag(image_corrupt)
        elif (scaling == 'normalization'):
            return self.norm_imag(image_corrupt)
        elif (scaling == 'positive_normalization'):
            return self.norm_positive_imag(image_corrupt)
        else: # No scaling required
            return image_corrupt, 0, 0
        
    #直接对tensor进行反向处理
    def descale_imag(self,image, param_scale1, param_scale2, scaling='standardization'):
        """ Descaling of input """
        try:
            image_np = image.detach().numpy()
        except:
            image_np = image
        if (scaling == 'standardization'):
            return self.destand_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'normalization'):
            return self.denorm_numpy_imag(image_np, param_scale1, param_scale2)
        elif (scaling == 'positive_normalization'):
            return self.denorm_numpy_positive_imag(image_np, param_scale1, param_scale2)
        else: # No scaling required
            return image_np

    #使用将每个iter的图片写到tensorboard的日志中
    def write_image_tensorboard(self,writer,image,name,suffix,image_gt,i=0,full_contrast=False):
        # Creating matplotlib figure with colorbar
        plt.figure()
        if (len(np.squeeze(image).shape) != 2):
            print('image is ' + str(len(image.shape)) + 'D, plotting only 2D slice')
            image = image[int(image.shape[0] / 2.),:,:]
        if (full_contrast):
            plt.imshow(image, cmap='gray_r',vmin=np.min(image),vmax=np.max(image)) # Showing each image with maximum contrast and white is zero (gray_r) 
        else:
            plt.imshow(image, cmap='gray_r',vmin=np.min(image_gt),vmax=1.25*np.max(image_gt)) # Showing all images with same contrast and white is zero (gray_r)
        plt.colorbar()
        
        #存照片的位置
        dir_path = os.path.join('images', 'tmp')
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(os.path.join(dir_path,name + '_' + str(i) + '.png'))
        writer.add_figure(name,plt.gcf(),global_step=i,close=True)

    def castor_common_command_line(self, subroot, PETImage_shape_str, phantom, replicates, post_smoothing=0):
        executable = 'castor-recon'
        if (self.nb_replicates == 1):
            header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[5:] + '/data' + phantom[5:] + '.cdh' # PET data path
        else:
            header_file = ' -df ' + subroot + 'Data/database_v2/' + phantom + '/data' + phantom[5:] + '_' + str(replicates) + '/data' + phantom[5:] + '_' + str(replicates) + '.cdh' # PET data path
        dim = ' -dim ' + PETImage_shape_str
        vox = ' -vox 4,4,4'
        vb = ' -vb 3'
        th = ' -th ' + str(self.nb_threads) # must be set to 1 for ADMMLim, as multithreading does not work for now with ADMMLim optimizer
        proj = ' -proj incrementalSiddon'
        if ("1" in PETImage_shape_str.split(',')): # 2D
            psf = ' -conv gaussian,4,1,3.5::psf'
        else: # 3D
            psf = ' -conv gaussian,4,4,3.5::psf' # isotropic psf in simulated phantoms

        if (post_smoothing != 0):
            if ("1" in PETImage_shape_str.split(',')): # 2D
                conv = ' -conv gaussian,' + str(post_smoothing) + ',1,3.5::post'
            else: # 3D
                conv = ' -conv gaussian,' + str(post_smoothing) + ',' + str(post_smoothing) + ',3.5::post' # isotropic post smoothing
        else:
            conv = ''
        # Computing likelihood
        if (self.castor_foms):
            opti_like = ' -opti-fom'
        else:
            opti_like = ''

        return executable + dim + vox + header_file + vb + th + proj + opti_like + psf + conv

    def castor_opti_and_penalty(self, method, penalty, rho, i=None, unnested_1st_global_iter=None):
        if (method == 'MLEM'):
            opti = ' -opti ' + method
            pnlt = ''
            penaltyStrength = ''
        if (method == 'OPTITR'):
            opti = ' -opti ' + method
            pnlt = ''
            penaltyStrength = ''
        elif (method == 'OSEM'):
            opti = ' -opti ' + 'MLEM'
            pnlt = ''
            penaltyStrength = ''
        elif (method == 'AML'):
            opti = ' -opti ' + method + ',1,1e-10,' + str(self.A_AML)
            pnlt = ''
            penaltyStrength = ''
        elif (method == 'APGMAP'):
            #opti = ' -opti ' + "APPGML" + ',1,1e-10,0.01,-1,' + str(self.A_AML) + ',0' # Multimodal image is only used by APPGML
            opti = ' -opti ' + "APPGML" + ':' + self.subroot + '/' + self.suffix  + '/' + 'APPGML.conf'
            pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF.conf'
            penaltyStrength = ' -pnlt-beta ' + str(self.beta)
        elif (method == 'BSREM'):
            opti = ' -opti ' + method + ':' + self.subroot_data + method + '.conf'
            pnlt = ' -pnlt ' + penalty + ':' + self.subroot_data + method + '_MRF.conf'
            penaltyStrength = ' -pnlt-beta ' + str(self.beta)
        elif ('nested' in method or 'ADMMLim' in method):
            if (self.recoInNested == "ADMMLim"):
                opti = ' -opti ' + 'ADMMLim' + ',' + str(self.alpha) + ',' + str(self.castor_adaptive_to_int(self.adaptive_parameters)) + ',' + str(self.mu_adaptive) + ',' + str(self.tau) + ',' + str(self.xi) + ',' + str(self.tau_max) + ',' + str(self.stoppingCriterionValue) + ',' + str(self.saveSinogramsUAndV)
                if ('nested' in method):
                    if ((i==0 and unnested_1st_global_iter) or (i==-1 and not unnested_1st_global_iter)): # For first iteration, put rho to zero
                        rho = 0
                        #self.rho = 0
                    method = 'ADMMLim' + method[6:]
                    #pnlt = ' -pnlt QUAD' # Multimodal image is only used by quadratic penalty
                    pnlt = ' -pnlt ' + "QUAD" + ':' + self.subroot + 'Block1/' + self.suffix  + '/' + 'QUAD.conf'
                elif ('ADMMLim' in method):
                    pnlt = ' -pnlt ' + penalty
                    if penalty == "MRF":
                        pnlt += ':' + self.subroot_data + method + '_MRF.conf'
                penaltyStrength = ' -pnlt-beta ' + str(rho)
            elif (self.recoInNested == "APGMAP"):
                if ((i==0 and unnested_1st_global_iter) or (i==-1 and not unnested_1st_global_iter)): # For first iteration, put rho to zero
                    rho = 0
                    #self.rho = 0
                #opti = ' -opti APPGML' + ',1,1e-10,0.01,-1,' + str(self.A_AML) + ',-1' # Do not use a multimodal image for APPGML, so let default multimodal index (-1)
                opti = ' -opti ' + "APPGML" + ':' + self.subroot + 'Block1/' + self.suffix  + '/' + 'APPGML.conf'
                #pnlt = ' -pnlt QUAD,0' # Multimodal image is used only for quadratic penalty, so put multimodal index to 0
                pnlt = ' -pnlt ' + "QUAD" + ':' + self.subroot + 'Block1/' + self.suffix  + '/' + 'QUAD.conf'
                penaltyStrength = ' -pnlt-beta ' + str(rho)
            
            # For all optimizers, remove penalty if rho == 0
            if (rho == 0):
                pnlt = ''
                penaltyStrength = ''
        elif (method == 'Gong'):
            if ((i==0 and unnested_1st_global_iter) or (i==-1 and not unnested_1st_global_iter)): # For first iteration, put rho to zero
                rho = 0
                #self.rho = 0
            opti = ' -opti OPTITR'
            pnlt = ' -pnlt OPTITR'
            penaltyStrength = ' -pnlt-beta ' + str(rho)
        
        # For all optimizers, remove penalty if rho == 0
        if (rho == 0):
            pnlt = ''
            penaltyStrength = ''
        
        return opti + pnlt + penaltyStrength

    def castor_adaptive_to_int(self,adaptive_parameters):
        if (adaptive_parameters == "nothing"): # not adaptive
            return 0
        if (adaptive_parameters == "alpha"): # only adative alpha
            return 1
        if (adaptive_parameters == "both"): # both adaptive alpha and tau
            return 2

    def get_phantom_ROI(self,image='image0'):
        # Select only phantom ROI, not whole reconstructed image
        path_phantom_ROI = self.subroot_data+'Data/database_v2/' + image + '/' + "phantom_mask" + str(image[5:]) + '.raw'
        my_file = Path(path_phantom_ROI)
        if (my_file.is_file()):
            phantom_ROI = self.fijii_np(path_phantom_ROI, shape=(self.PETImage_shape),type_im='<f')
        else:
            print("No phantom file for this phantom")
            phantom_ROI = np.ones_like(self.image_gt)
            #raise ValueError("No phantom file for this phantom")
            #phantom_ROI = self.fijii_np(self.subroot_data+'Data/database_v2/' + image + '/' + "background_mask" + image[5:] + '.raw', shape=(self.PETImage_shape),type_im='<f')
            
        return phantom_ROI
    
    def mkdir(self,path):
        # check path exists or no before saving files
        folder = os.path.exists(path)

        if not folder:
            os.makedirs(path)

        return path


    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        print(re.split(r'(\d+)', text))
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ] # APGMAP final curves + resume computation
        #return [ self.atoi(c) for c in re.split(r'(\+|-)\d+(\.\d+)?', text) ] # ADMMLim final curves
    
    def natural_keys_ADMMLim(self,text): # Sort by scientific or float numbers
        #return [ self.atoi(c) for c in re.split(r'(\d+)', text) ] # APGMAP final curves + resume computation
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
        final_list = [float(x) for x in re.findall(match_number, text)] # Extract scientific of float numbers in string
        return final_list # ADMMLim final curves
        
    def has_numbers(self,inputString):
        return any(char.isdigit() for char in inputString)

    def ImageAndItToResumeComputation(self,sorted_files, it, folder_sub_path):
        sorted_files.sort(key=self.natural_keys)
        last_file = sorted_files[-1]
        if ("=" in last_file): # post reco mode
            last_file = last_file[-10:]
            last_file = "it_" + last_file.split("=",1)[1]
        last_iter = int(re.findall(r'(\w+?)(\d+)', last_file.split('.')[0])[0][-1])
        initialimage = ' -img ' + folder_sub_path + '/' + last_file
        it += ' -skip-it ' + str(last_iter)
        
        return initialimage, it, last_iter

    def linear_regression(self, x, y):
        x_mean = x.mean()
        y_mean = y.mean()
        
        B1_num = ((x - x_mean) * (y - y_mean)).sum()
        B1_den = ((x - x_mean)**2).sum()
        B1 = B1_num / B1_den
                        
        return B1

    def defineTotalNbIter_beta_rho(self,method,config,task):
        if (method == 'ADMMLim'):
            try:
                self.path_stopping_criterion = self.subroot + self.suffix + '/' + format(0) + '_adaptive_stopping_criteria.log'
                with open(self.path_stopping_criterion) as f:
                    first_line = f.readline() # Read first line to get second one
                    self.total_nb_iter = min(int(f.readline().rstrip()) - self.i_init, config["nb_outer_iteration"] - self.i_init + 1)
                    #self.total_nb_iter = int(self.total_nb_iter / self.i_init) # if 1 out of i_init iterations was saved
                    #self.total_nb_iter = config["nb_outer_iteration"] - self.i_init + 1 # Override value
            except:
                self.total_nb_iter = config["nb_outer_iteration"] - self.i_init + 1
                #self.total_nb_iter = int(self.total_nb_iter / self.i_init) # if 1 out of i_init iterations was saved
            self.beta = config["alpha"]
        elif ('nested' in method or 'Gong' in method or 'DIPRecon' in method):
            if ('post_reco' in task):
                self.total_nb_iter = config["sub_iter_DIP"]
            else:
                self.total_nb_iter = config["max_iter"]
        else:
            self.total_nb_iter = self.max_iter

            if (config["method"] == 'AML'):
                self.beta = config["A_AML"]
            if (config["method"] == 'BSREM' or 'nested' in config["method"] or 'Gong' in config["method"] or 'DIPRecon' in config["method"] or 'APGMAP' in config["method"]):
                self.rho = config["rho"]
                self.beta = self.rho
from ray import tune
import torch
from models.DIP_2D import DIP_2D
import pytorch_lightning as pl
import numpy as np

settings_config = {
    # "image" : (['image4_0']), # Image from database
    "random_seed" : True, # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    # "method" : (['nested']), # Reconstruction algorithm (nested, Gong, or algorithms from CASToR (MLEM, BSREM, AML, etc.))'ADMMLim'
    "processing_unit" : 'CPU', # CPU or GPU
    # "nb_threads" : ([1]), # Number of desired threads. 0 means all the available threads
    "FLTNB" : 'float', # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
    "debug" : False, # Debug mode = run without raytune and with one iteration
    "ray" : True, # Ray mode = run with raytune if True, to run several settings in parallel
    "tensorboard" : True, # Tensorboard mode = show results in tensorboard
    # "all_images_DIP" : (['True']), # Option to store only 10 images like in tensorboard (quicker, for visualization, set it to "True" by default). Can be set to "True", "False", "Last" (store only last image)
    # "experiment" : ([24]),
    # "replicates" : (list(range(1,1+1))), # List of desired replicates. list(range(1,n+1)) means n replicates
    # "average_replicates" : ([False]), # List of desired replicates. list(range(1,n+1)) means n replicates
    # "castor_foms" : ([True]), # Set to True to compute CASToR Figure Of Merits (likelihood, residuals for ADMMLim)
}
# Configuration dictionnary for previous hyperparameters, but fixed to simplify
fixed_config = {
    # "max_iter" : ([15]), # Number of global iterations for usual optimizers (MLEM, BSREM, AML etc.) and for nested and Gong
    # "nb_subsets" : ([28]), # Number of subsets in chosen reconstruction algorithm (automatically set to 1 for ADMMLim)
    # "finetuning" : (['last']),
    # "penalty" : (['MRF']), # Penalty used in CASToR for PLL algorithms
    # "unnested_1st_global_iter" : ([False]), # If True, unnested are computed after 1st global iteration (because rho is set to 0). If False, needs to set f_init to initialize the network, as in Gong paper, and rho is not changed.
    # "sub_iter_DIP_initial_and_final" : ([1000]), # Number of epochs in first global iteration (pre iteraiton) in network optimization (only for Gong for now)
    # "nb_inner_iteration" : ([1]), # Number of inner iterations in ADMMLim (if mlem_sequence is False). (3 sub iterations are done within 1 inner iteration in CASToR)
    # "xi" : ([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in ADMMLim
    # "xi_DIP" : ([1]), # Factor to balance primal and dual residual convergence speed in adaptive tau computation in Gong and nested
    # "net" : (['DIP']), # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "DIP_early_stopping" : False, # Use DIP early stopping with WMV strategy
    "windowSize" : 50, # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "patienceNumber" : 100, # Network to use (DIP,DD,DD_AE,DIP_VAE)
}
# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    # "image_init_path_without_extension" : (['1_im_value_cropped']), # Initial image of the reconstruction algorithm (taken from data/algo/Data/initialization)
    # "rho" : ([0.003,8e-4,0.008,0.03]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    # "rho" : ([0.0002]), # Penalty strength (beta) in PLL algorithms, ADMM penalty parameter (nested and Gong)
    # "adaptive_parameters_DIP" : (["nothing"]), # which parameters are adaptive ? Must be set to nothing, alpha, or tau (which means alpha and tau)
    # "mu_DIP" : ([10]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMLim
    # "tau_DIP" : ([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMLim. If adaptive tau, it corresponds to tau max
    ## network hyperparameters
    "lr" : 1e-4, # Learning rate in network optimization
    # "lr" : ([0.01]), # Learning rate in network optimization
    "sub_iter_DIP" : 20, # Number of epochs in network optimization
    "opti_DIP" : 'Adam', # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : 3, # Number of skip connections in DIP architecture (0, 1, 2, 3)
    # "scaling" : (['standardization']), # Pre processing of neural network input (nothing, uniform, normalization, standardization)
    # "input" : (['random']), # Neural network input (random or CT)
    #"input" : (['CT','random']), # Neural network input (random or CT)
    # "d_DD" : ([4]), # d for Deep Decoder, number of upsampling layers. Not above 4, otherwise 112 is too little as output size / not above 6, otherwise 128 is too little as output size
    # "k_DD" : ([32]), # k for Deep Decoder
    ## ADMMLim - OPTITR hyperparameters
    #"nb_outer_iteration": ([30]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
    #"nb_outer_iteration": ([3]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
    # "nb_outer_iteration": ([30]), # Number of outer iterations in ADMMLim (and nested) and OPTITR (for Gong)
    # "alpha" : ([1]), # alpha (penalty parameter) in ADMMLim
    # "adaptive_parameters" : (["both"]), # which parameters are adaptive ? Must be set to nothing, alpha, or both (which means alpha and tau)
    # "mu_adaptive" : ([2]), # Factor to balance primal and dual residual in adaptive alpha computation in ADMMLim
    # "tau" : ([2]), # Factor to multiply alpha in adaptive alpha computation in ADMMLim
    # "tau_max" : ([100]), # Maximum value for tau in adaptive tau in ADMMLim
    # "stoppingCriterionValue" : ([0.01]), # Value of the stopping criterion in ADMMLim
    # "saveSinogramsUAndV" : ([1]), # 1 means save sinograms u and v from CASToR, otherwise it means do not save them
    # ## hyperparameters from CASToR algorithms 
    # # Optimization transfer (OPTITR) hyperparameters
    # "mlem_sequence" : ([False]), # Given sequence (with decreasing number of subsets) to quickly converge. True or False
    # # AML/APGMAP hyperparameters
    # "A_AML" : ([-100,-500,-10000]), # AML lower bound A
    # "A_AML" : ([-100]), # AML lower bound A
    # # Post smoothing by CASToR after reconstruction
    # "post_smoothing" : ([0]), # Post smoothing by CASToR after reconstruction
    # #"post_smoothing" : ([6,9,12,15]), # Post smoothing by CASToR after reconstruction
    # # NNEPPS post processing
    # "NNEPPS" : ([False]), # NNEPPS post-processing. True or False
}
split_config = {
    "fixed_hyperparameters" : list(fixed_config.keys()),
    "hyperparameters" : list(hyperparameters_config.keys())
}
config = {**settings_config, **fixed_config, **hyperparameters_config, **split_config}
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
    
# Merge 3 dictionaries
split_config = {
    "fixed_hyperparameters" : list(fixed_config.keys()),
    "hyperparameters" : list(hyperparameters_config.keys())
}

#todo:
path_noisy="/home/xzhang/Documents/DIP/image/BSREM_it30.img"
path_output = "/home/xzhang/Documents/DIP/image/"
PETImage_shape=(112,112,1)
image_corrupt=fijii_np(path_noisy,PETImage_shape)
image_corrupt_input_scale,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization")
image_corrupt_torch = torch.Tensor(image_corrupt_input_scale)
image_corrupt_torch = image_corrupt_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]

PETImage_shape=(112,112,1)
image_net_input =np.random.uniform(PETImage_shape)
image_net_input_torch = torch.Tensor(image_net_input)
image_net_input_torch = image_net_input_torch.view(1,1,PETImage_shape[0],PETImage_shape[1],PETImage_shape[2])
image_net_input_torch = image_net_input_torch[:,:,:,:,0]

config = {**settings_config, **fixed_config, **hyperparameters_config, **split_config}

image_corrupt_input_scale,param1_scale_im_corrupt,param2_scale_im_corrupt = rescale_imag(image_corrupt,"standardization") 


train_dataset = torch.utils.data.TensorDataset(image_net_input_torch, image_corrupt_torch)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

model = DIP_2D(param1_scale_im_corrupt, param2_scale_im_corrupt, 'standardization', config,'data/Algo/',path_output,"nested",all_images_DIP="True",global_it=-100, fixed_hyperparameters_list="for early stopping", hyperparameters_list="for early stopping", debug=config["debug"], suffix="suffix", last_iter=-1)
model_class = DIP_2D
trainer = pl.Trainer(max_epochs=config["sub_iter_DIP"],log_every_n_steps=1)#, callbacks=[checkpoint_callback, tuning_callback, early_stopping_callback], logger=logger,gpus=gpus, accelerator=accelerator, profiler="simple")

trainer.fit(model, train_dataloader)
out = model(image_net_input_torch)
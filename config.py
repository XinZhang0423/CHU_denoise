#from ray import tune


PETImage_shape=(112,112,1)  # 输入图片的大小

settings_config = {
   
    "random_seed" : True, # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
    "processing_unit" : 'CPU', # CPU or GPU
    "FLTNB" : 'float', # FLTNB precision must be set as in CASToR (double necessary for ADMMLim and nested)
    "debug" : False, # Debug mode = run without raytune and with one iteration
    "ray" : True, # Ray mode = run with raytune if True, to run several settings in parallel
    "tensorboard" : False, # Tensorboard mode = show results in tensorboard
    "experiment" : 24,
}

# Configuration dictionnary for previous hyperparameters, but fixed to simplify
fixed_config = {
    "DIP_early_stopping" : False, # Use DIP early stopping with WMV strategy
    "windowSize" : 50, # Network to use (DIP,DD,DD_AE,DIP_VAE)
    "patienceNumber" : 100, # Network to use (DIP,DD,DD_AE,DIP_VAE)
}

# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "lr" : 1e-4, # Learning rate in network optimization
    # "lr" : ([0.01]), # Learning rate in network optimization
    "sub_iter_DIP" : 10, # Number of epochs in network optimization
    "opti_DIP" : 'Adam', # Optimization algorithm in neural network training (Adam, LBFGS)
    "skip_connections" : 3, # Number of skip connections in DIP architecture (0, 1, 2, 3)
}

# Merge 3 dictionaries
split_config = {
    "fixed_hyperparameters" : list(fixed_config.keys()),
    "hyperparameters" : list(hyperparameters_config.keys())
}

config = {**settings_config, **fixed_config, **hyperparameters_config, **split_config}
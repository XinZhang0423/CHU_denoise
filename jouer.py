import torch
import os
import numpy as np
for npy in os.listdir("images/noise_images/"):
    print(npy)
    image = np.load(npy)
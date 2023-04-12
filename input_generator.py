

import numpy as np
import os

# 本文件用于测试各种input图片对结果的影响，手动生成各种各样的随机图片，不过我感觉结果区别不大
# 分别有高斯，均匀，

# 生成高斯噪声图像
def generate_gaussian_noise_image(shape, mean, std_dev):
    noise = np.random.normal(mean, std_dev, shape)
    return noise

# 生成均匀分布噪声图像
def generate_uniform_noise_image(shape, low, high):
    noise = np.random.uniform(low, high, shape)
    return noise

# 本质上是rand的二值化，即均匀分布的二值化
def generate_uniform_binaire_image(shape, salt_prob, pepper_prob):
    noise = np.zeros(shape)
    salt_mask = np.random.rand(*shape) < salt_prob
    pepper_mask = np.random.rand(*shape) < pepper_prob
    noise[salt_mask] = 255
    noise[pepper_mask] = 0
    return noise

# 生成泊松噪音
def generate_poisson_noise_image(shape, lam):
    noise = np.random.poisson(lam, shape)
    return noise

# 生成指数噪声
def generate_exponential_noise_image(shape, scale):
    noise = np.random.exponential(scale, shape)
    return noise

#生成gamma噪音
def generate_gamma_noise_image(shape, shape_param, scale):
    noise = np.random.gamma(shape_param, scale, shape)
    return noise

# 设置保存路径
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images',"noise_images")
os.makedirs(save_path, exist_ok=True)

# 生成并保存噪声图像
shape = (112, 112,1)
# 高斯
mean = 0
std_dev = 1

# 均匀
low = 0
high = 0.1



gaussian_noise = generate_gaussian_noise_image(shape, mean, std_dev)
uniform_noise = generate_uniform_noise_image(shape, low, high)
poisson_noise = generate_poisson_noise_image(shape, 1)
uniform_noise_binaire = generate_uniform_binaire_image(shape, 0.5,0.8)
exponential_noise = generate_exponential_noise_image(shape, 1)
gamma_noise = generate_gamma_noise_image(shape, 0.5, 0.8)

# 保存高斯噪声图像
filename = os.path.join(save_path, "gaussian_noise.npy")
np.save(filename, gaussian_noise)

# 保存均匀噪声图像
filename = os.path.join(save_path, "uniform_noise.npy")
np.save(filename, uniform_noise)

filename = os.path.join(save_path, "poisson_noise.npy")
np.save(filename, poisson_noise)

filename = os.path.join(save_path, "uniform_noise_binaire.npy")
np.save(filename, uniform_noise_binaire)

filename = os.path.join(save_path, "exponential_noise.npy")
np.save(filename, exponential_noise)

filename = os.path.join(save_path, "gamma_noise.npy")
np.save(filename, gamma_noise)

print("生成完毕")


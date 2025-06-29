import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
import model
import data as da
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import binarize_image
import matplotlib.pyplot as plt
import file_class
import random


def generate_newbatch(batch_size, height=1024, width=2048, mean=0, var=0.5):
    noise = np.random.randint(0, 256, (batch_size, height, width), dtype=np.uint8)
    noise = noise.astype('float32') / 127.5 - 1.
    return noise


def read_batch(batch_size, height=1024, width=2048, mean=0, var=0.5, paths='work/noise_images'):
    path_list = os.listdir(paths)
    idx = random.randint(0, len(path_list) - 1)
    noise_path = os.path.join(paths, path_list[idx])
    noise = cv2.imread(noise_path)
    noise = cv2.resize(noise, (width, height))
    noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    # noise = np.expand_dims(noise, axis=2)
    # noise = noise.transpose(2, 0, 1)  # HWC -> CHW
    noise = noise.astype('float32') / 127.5 - 1.  # 归一化
    noise = np.tile(noise, (batch_size, 1, 1))
    return noise


def generate_batch_gaussian_noise_image(batch_size, height=1024, width=2048, mean=0, var=0.5):
    """
    生成指定大小的高斯噪声图像。

    :param height: 图像的高度
    :param width: 图像的宽度
    :param mean: 高斯噪声的均值
    :param var: 高斯噪声的方差
    :return: 生成的高斯噪声图像
    """
    gauss = np.random.normal(mean, var ** 0.5, (batch_size, height, width))
    gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min()) * 255  # 归一化到0-255
    gauss = gauss.astype('float32') / 127.5 - 1.
    return gauss


if __name__ == '__main__':
    # 先读取模型，然后生成高斯噪声，然后生成背景图片 读取原始图片，然后添加噪声并保存
    # 这句话用来添加噪声 real_A = np.where(real_B == 255, noise_image, real_B)
    model_path = 'work/GANs/generate_noise.pdparams'
    generator = model.UnetGenerator()
    generator.set_dict(paddle.load(model_path))
    height = 256
    weight = 512

    all_path_f = 'work/pro_ecg_fenge_version'
    all_path_d = 'work/randominput_generate_noise'
    src_pathlist = os.listdir(all_path_f)
    for src_path in src_pathlist:
        path = os.path.join(all_path_f, src_path)
        print(path)
        path_to = os.path.join(all_path_d, src_path)
        pro_paths = file_class.ergodic(path, arrangement=1)
        pro_paths = list(pro_paths)
        for k, pro_path in tqdm(enumerate(pro_paths)):
            path_to2 = os.path.join(path_to, os.path.split(pro_path)[1][:-4] + '_' + str(k) + '.png')
            # 在这里开始循环加噪
            img = cv2.imread(pro_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (2048, 1024))
            # 获取高斯噪声
            # back_input = generate_newbatch(16, height, weight)
            # 随机高斯和训练集高斯各一半
            if random.randint(1,2) == 1:
                back_input = generate_newbatch(16, height, weight)
            else:
                back_input = read_batch(16, height, weight)

            # 用生成器生成背景
            back_gan = generator(paddle.to_tensor(back_input).unsqueeze(1))
            back_gan = (back_gan * 127.5 + 127.5).astype(np.uint8)
            back_gan = back_gan.numpy()
            back_gan = back_gan.reshape(1024, 2048)
            # print(img.shape, back_gan.shape)
            img = np.where(img == 255, back_gan, img)
            cv2.imwrite(path_to2, img)

    # cv2.imshow('img',img)
    # cv2.waitKey(0)




















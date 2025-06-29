import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
import model
import data as da
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


# 用于生成高斯噪声图像作为输入
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


def center_crop(image, target_height=1024, target_width=2048):
    """
    从中心裁剪图像为指定尺寸
    :param image: 输入图像
    :param target_width: 目标宽度
    :param target_height: 目标高度
    :return: 裁剪后的图像
    """
    height, width = image.shape[:2]
    if width < target_width or height < target_height:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # 计算中心位置
    center_x, center_y = width // 2, height // 2

    # 计算裁剪区域
    crop_x1 = max(0, center_x - target_width // 2)
    crop_x2 = min(width, center_x + target_width // 2)
    crop_y1 = max(0, center_y - target_height // 2)
    crop_y2 = min(height, center_y + target_height // 2)

    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_image


class Back_Dataset(Dataset):
    def __init__(self, noise_dir, back_dir, height=256, weight=512, transform=None):
        self.height = height
        self.weight = weight
        # 读取噪声图片路径
        self.noise_dir = noise_dir
        self.transform = transform
        self.noise_pths = []  # 存储图像路径和标签
        for cls_dir in os.listdir(noise_dir):
            cls_path = os.path.join(noise_dir, cls_dir)
            self.noise_pths.append(cls_path)
        # 读取的是背景图片路径
        self.back_dir = back_dir
        self.transform = transform
        self.img_pths = []  # 存储图像路径和标签

        # 遍历数据目录，构建图像路径和标签的列表
        for cls_dir in os.listdir(back_dir):
            cls_path = os.path.join(back_dir, cls_dir)
            self.img_pths.append(cls_path)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        :param idx: 索引
        :return: 图像数据，标签
        """
        noise_path = self.noise_pths[idx]
        noise = cv2.imread(noise_path)
        noise = cv2.resize(noise, (self.weight, self.height))
        noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
        noise = np.expand_dims(noise, axis=2)
        noise = noise.transpose(2, 0, 1)  # HWC -> CHW
        noise = noise.astype('float32') / 127.5 - 1.  # 归一化

        img_path = self.img_pths[idx % len(self.img_pths)]
        img_A2B = cv2.imread(img_path)  # 读取数据
        # img_A2B = cv2.resize(img_A2B, (2048, 1024))
        img_A2B = center_crop(img_A2B, self.height, self.weight)
        # 转为单通道：
        img_A2B = cv2.cvtColor(img_A2B, cv2.COLOR_BGR2GRAY)
        img_A2B = np.expand_dims(img_A2B, axis=2)
        img_A2B = img_A2B.transpose(2, 0, 1)  # HWC -> CHW
        img_A2B = img_A2B.astype('float32') / 127.5 - 1.  # 归一化
        return noise, img_A2B

    def __len__(self):
        """
        返回数据集的总样本数
        """
        return len(self.noise_pths)


if __name__ == '__main__':
    height = 256
    weight = 512
    back_dataset = Back_Dataset('work/noise_images', 'work/back', height, weight)
    # for i in range(len(back_dataset)):
    #     back_img = back_dataset.__getitem__(i)
    #     print(back_img.shape)
    test_dataset = Back_Dataset('work/noise_images', 'work/back', height, weight)
    generator = model.UnetGenerator()
    discriminator = model.NLayerDiscriminator(1)
    # 超参数
    LR = 1e-4
    BATCH_SIZE = 4
    EPOCHS = 500
    paddle.set_device('gpu:0')

    # 优化器
    optimizerG = paddle.optimizer.Adam(
        learning_rate=LR,
        parameters=generator.parameters(),
        beta1=0.5,
        beta2=0.999)

    optimizerD = paddle.optimizer.Adam(
        learning_rate=LR,
        parameters=discriminator.parameters(),
        beta1=0.5,
        beta2=0.999)

    # 损失函数
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()

    # dataloader
    data_loader_train = DataLoader(
        back_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    data_loader_test = DataLoader(
        back_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    results_save_path = 'work/results/fig'
    os.makedirs(results_save_path, exist_ok=True)  # 保存每个epoch的测试结果

    weights_save_path = 'work/results/weights'
    os.makedirs(weights_save_path, exist_ok=True)  # 保存模型

    for epoch in range(EPOCHS):
        for real_A, real_B in tqdm(data_loader_train):
            # print('#########################')
            # print(real_B.shape,real_A.shape)
            # print(real_A, real_B)
            optimizerD.clear_grad()
            # D(real)
            # real_B = real_B.unsqueeze(1)
            d_real_predict = discriminator(real_B)
            d_real_loss = bce_loss(d_real_predict, paddle.ones_like(d_real_predict))  # 判别器损失，结果为真

            # D(fake)
            fake_B = generator(real_A).detach()
            # fake_B = fake_B.unsqueeze(1)
            d_fake_predict = discriminator(fake_B)
            d_fake_loss = bce_loss(d_fake_predict, paddle.zeros_like(d_fake_predict))  # 判别器损失，结果为假

            # print(fake_B.shape, d_fake_predict.shape)

            # train D
            d_loss = (d_real_loss + d_fake_loss) / 2.  # 判别器两对样本损失相加
            d_loss.backward()
            optimizerD.step()  # 更新判别器

            optimizerG.clear_grad()
            # D(fake)
            fake_B = generator(real_A)
            # fake_B = fake_B.unsqueeze(1)
            g_fake_predict = discriminator(fake_B)
            g_bce_loss = bce_loss(g_fake_predict, paddle.ones_like(g_fake_predict))
            g_l1_loss = l1_loss(fake_B, real_B) * 100.
            g_loss = g_bce_loss + g_l1_loss

            # train G
            g_loss.backward()
            optimizerG.step()

        print(f'Epoch [{epoch + 1}/{EPOCHS}] Loss D: {d_loss.numpy()}, Loss G: {g_loss.numpy()}')

        if (epoch + 1) % 10 == 0:
            paddle.save(generator.state_dict(),
                        os.path.join(weights_save_path, 'epoch' + str(epoch + 1).zfill(3) + '.pdparams'))

            # test
            generator.eval()
            with paddle.no_grad():
                for real_A, real_B in data_loader_test:
                    break
                fake_B = generator(real_A)
                result = paddle.concat([real_A[:3], real_B[:3], fake_B[:3]], 3)
                print(result.shape)
                result = result.detach().numpy().transpose(0, 2, 3, 1)
                print(result.shape)
                result = np.vstack(result)
                print(result.shape)
                # print(result)
                result = (result * 127.5 + 127.5).astype(np.uint8)
            # print(result)
            cv2.imwrite(os.path.join(results_save_path, 'epoch' + str(epoch + 1).zfill(3) + '.png'), result)

            generator.train()

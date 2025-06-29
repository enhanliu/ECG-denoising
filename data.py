from paddle.io import Dataset, DataLoader
import os
import cv2
from PIL import Image
import numpy as np
from paddle.vision import transforms


class PairedData(Dataset):
    def __init__(self, phase):
        super(PairedData, self).__init__()
        self.img_path_list = self.load_A2B_data(phase)  # 获取数据列表
        self.num_samples = len(self.img_path_list)  # 数据量
        self.transform = transforms.Compose([
            transforms.Resize((1024, 2048)),  # 替换new_height和new_width为你需要的大小
            transforms.ToTensor()  # 如果需要转换为tensor的话
        ])

    def __getitem__(self, idx):
        img_A2B = cv2.imread(self.img_path_list[idx])  # 读取数据
        img_A2B = cv2.resize(img_A2B, (2048, 1024))
        # 转为单通道：
        img_A2B = cv2.cvtColor(img_A2B, cv2.COLOR_BGR2GRAY)
        img_A2B = np.expand_dims(img_A2B, axis=2)
        img_A2B = img_A2B.transpose(2, 0, 1)  # HWC -> CHW
        img_A2B = img_A2B.astype('float32') / 127.5 - 1.  # 归一化

        img_A = img_A2B

        path_b = os.path.join('work/ecg_pro', os.path.split(self.img_path_list[idx])[1].split('_')[0] + '.png')
        # print(path_b)
        img_B = cv2.imread(path_b)
        img_B = cv2.resize(img_B, (2048, 1024))

        # 转为单通道
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        img_B = np.expand_dims(img_B, axis=2)
        img_B = img_B.transpose(2, 0, 1)
        img_B = img_B.astype('float32') / 127.5 - 1.



        return img_A, img_B # A为有噪声，B为无噪声
        # return img_A[:, :255, :255], img_B[:, :255, :255]

    def __len__(self):
        return self.num_samples

    @staticmethod
    def load_A2B_data(phase):
        # assert phase in ['train1', 'train2'],
        data_path = phase
        return [os.path.join(data_path, x) for x in os.listdir(data_path)]

# data1 = PairedData('compose_ecg_0')
# a, b = data1[1]
# print(a.shape,b.shape)

import csv
import os
import cv2
import numpy
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# from config import *
from utils.func import csv2lst, list_to_float, delete_first_element, flatten_list

'''
path: input image 文件夹
flag: True(original 数据输出 O) / False(original 数据输出 X)
'''


class CoarseDataset(Dataset):
    def __init__(self, path, flag=False):
        self.img_dir = path
        self.image_lst = glob(self.img_dir + '*.jpg')
        self.img_size = 160
        self.mode = flag

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, index):
        img_path = self.image_lst[index]
        img_title = img_path.split('/')[-1]
        title = ''.join(img_title.split('.jpg')[0])
        # print(title)
        row = title[-3:]
        # print(row)
        # 训练/测试
        csv_path = f"../Graduation_Project/trainSet/cephalometric-landmarks/cepha400/trainDataset/train_senior.csv"
        # csv_path = f"../Graduation_Project/trainSet/cephalometric-landmarks/cepha400/testDataset/test.csv"
        #测试
        # landmark = csv2lst(csv_path,int(row)-351)
        #训练
        landmark = csv2lst(csv_path, int(row))

        landmark = flatten_list(landmark)  #去掉列表的一层嵌套
        landmark = delete_first_element(landmark)

        landmark = list_to_float(landmark)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # print(img)

        # CLAHE 增强图像对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(img)

        org_size = image.shape
        w_ratio, h_ratio = org_size[0] / self.img_size, org_size[1] / self.img_size
        transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensorV2()
        ])
        data = transform(image=image)
        res = []
        for idx in range(0, len(landmark), 2):
            res.append(int(landmark[idx] / h_ratio))
            res.append(int(landmark[idx + 1] / w_ratio))
            # res.append(int(landmark[idx] ))
            # res.append(int(landmark[idx + 1]))

        if self.mode:
            return title, data['image'], res, [img, landmark]
        else:
            return data['image'], res, [img, landmark]
            # return title, data['image'], res


class FineDataset(Dataset):
    def __init__(self, num, anno_path):
        self.root_dir = '../Graduation_Project/trainSet/cephalometric-landmarks/cepha400/trainDataset/'
        self.img_dir = self.root_dir + num + '/'
        self.image_lst = os.listdir(self.img_dir)
        self.landmarks = []
        self.img_size = 224

        df = pd.read_csv(anno_path, header=None)

        for i in range(df.shape[0]):
            sr = df.iloc[i].tolist()
            self.landmarks.append(sr)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, index):
        landmark = self.landmarks[index]

        img_path = ''
        for path in self.image_lst:
            if path == landmark[0]:
                img_path = self.img_dir + path

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(img)

        org_size = image.shape
        w_ratio, h_ratio = org_size[0] / self.img_size, org_size[1] / self.img_size
        transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensorV2()
        ])
        data = transform(image=image)
        res = []
        for idx in range(1, len(landmark), 2):
            res.append(int(landmark[idx] / h_ratio))
            res.append(int(landmark[idx + 1] / w_ratio))

        return data['image'], res, [img, landmark]
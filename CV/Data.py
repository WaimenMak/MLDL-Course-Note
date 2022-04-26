# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 17:20
# @Author  : Weiming Mai
# @FileName: Data.py
# @Software: PyCharm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import pandas as pd
# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )
preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normalize
])

augprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
    transforms.RandomVerticalFlip(p=0.5),  # 随机水平翻转 选择一个概率
    transforms.ToTensor()
])

augprocess_2 = transforms.Compose([
    transforms.ToTensor(),
    # transforms.ToPILImage(),
    transforms.CenterCrop(196),
    transforms.RandomRotation((90), expand=True),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
    transforms.RandomVerticalFlip(p=0.5),  # 随机水平翻转 选择一个概率
    # transforms.RandomErasing(),
])


def default_loader(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((224, 224))
    img_tensor = preprocess(img_pil)
    return img_tensor


def augment_loader(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((224, 224))
    img_tensor = augprocess(img_pil)
    return img_tensor

def augment_loader2(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((224, 224))
    img_tensor = augprocess_2(img_pil)
    return img_tensor

class trainset(Dataset):
    def __init__(self, loader=default_loader, val_ratio=0.3, train=True, val=False, test=False):
        # 定义好 image 的路径
        data = pd.read_csv("./classify-leaves/processed_data.csv")
        idx = int(len(data) * (1 - val_ratio))
        self.train, self.val, self.test = train, val, test
        if train:
            self.images_path = data["image"][0:idx].values
            self.label = data["label"][0:idx].values
        elif val:
            self.images_path = data["image"][idx:].values
            self.label = data["label"][idx:].values
        elif test:
            data = pd.read_csv("./classify-leaves/test.csv")
            self.images_path = data["image"].values
        self.loader = loader

    def __getitem__(self, index):
        path = self.images_path[index]
        img = self.loader("./classify-leaves/" + path)
        if not self.test:
            label = self.label[index]
            return img, label
        return img

    def __len__(self):

        return len(self.images_path)
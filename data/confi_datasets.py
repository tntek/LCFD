import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
import re

class BaseJsonDataset(Dataset):
    def __init__(self, confi_imag, confi_dis,mode='train', n_shot=None, transform=None):
        self.transform = transform
        # self.image_path = image_path
        # self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        self.shot_predict_list = []
        # txt_tar = open(json_path).readlines()
        # samples = []
        samples = confi_imag
        shot_predict = confi_dis
        # cls_val, shot_predict = torch.max(confi_dis, 1)
        self.shot_predict_list = shot_predict.cpu().numpy().tolist()
        # for line in txt_tar:
        #     # line=line.rstrip("\n")
        #     line_split = re.split(' ',line)
        #     samples.append(line_split)
        for s in samples: #s:['Faces/image_0353.jpg', 0, 'face']
            self.image_list.append(s[0])
            # s[1] = s[1])
            self.label_list.append(s[1])
            # self.shot_predict_list.append(s[1])
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        pesu_label = self.shot_predict_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long(),torch.tensor(pesu_label),idx

# fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397', 
#                     'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']

domain_datasets = ['office','office-home','VISDA-C','Domain-Net']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "data/data_splits/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "data/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "data/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["", "data/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["", "data/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["", "data/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["", "data/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "data/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["", "data/data_splits/split_zhou_EuroSAT.json"]
}

def build_confi_dataset(confi_imag, confi_dis,transform, mode='train', n_shot=None):
    return BaseJsonDataset(confi_imag,confi_dis,mode, n_shot, transform)

def build_cifar_dataset(confi_imag, confi_dis,transform):
    return CustomCifarDataset(confi_imag,confi_dis,transform)

class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class CustomCifarDataset(Dataset):
    def __init__(self, samples, con_dis,transform=None):
        super(CustomCifarDataset, self).__init__()

        self.samples = samples
        self.shot_predict_list = con_dis.cpu().numpy().tolist()
        self.transform = transform

    def __getitem__(self, idx):
        img, label, domain = self.samples[idx]
        # img = torch.tensor(img.transpose((2, 0, 1)))
        img = Image.fromarray(np.uint8(img * 255.)).convert('RGB')
        img = self.transform(img)
        pesu_label = self.shot_predict_list[idx]
        
        return img, torch.tensor(label).long(),torch.tensor(pesu_label),idx
        
    def __len__(self):
        return len(self.samples)


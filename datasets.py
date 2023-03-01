import glob
import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self,path, transforms_A=None, transforms_B=None,unaligned=False, mode='train'):
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B
        self.path = path
        self.unaligned = unaligned


        self.files_B = sorted(glob.glob(os.path.join(self.path, '%s/B' % mode) + '/*.*'))
        self.files_A = sorted(glob.glob(os.path.join(self.path, '%s/A' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(self.path, '%s/C' % mode) + '/*.*'))



    def __getitem__(self, index):
        item_A = self.transforms_A(Image.open(self.files_A[index % len(self.files_A)]))
        item_C = []


        if os.path.exists(self.path+'/train/C'):
            item_C = self.transforms_A(Image.open(self.files_C[index % len(self.files_C)]))

        if self.unaligned:
            item_B = self.transforms_B(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transforms_B(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B, 'C': item_C, }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDataset_crop(Dataset):
    def __init__(self,path, img_size=512,transforms_A=None, transforms_B=None,unaligned=False, mode='train'):

        self.transforms_A = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_size/2)
        ])

        self.transforms_B = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_size/2)
        ])

        self.transforms_C = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(img_size / 8)
        ])
        self.path = path
        self.unaligned = unaligned


        self.files_B = sorted(glob.glob(os.path.join(self.path, '%s/B' % mode) + '/*.*'))
        self.files_A = sorted(glob.glob(os.path.join(self.path, '%s/A' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(self.path, '%s/C' % mode) + '/*.*'))



    def __getitem__(self, index):
        # 用该方法获取一个随机种子
        seed = random.randint(0,100)
        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        item_A = self.transforms_A(Image.open(self.files_A[index % len(self.files_A)]))
        item_C = []



        if os.path.exists(self.path+'/train/C'):
            torch.random.manual_seed(seed)
            torch.cuda.random.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            item_C = self.transforms_C(Image.open(self.files_C[index % len(self.files_C)]))


        torch.random.manual_seed(seed)
        torch.cuda.random.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        if self.unaligned:
            item_B = self.transforms_B(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transforms_B(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B, 'C': item_C, }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

import argparse
import os
import time

import cv2
import numpy as np
import torch.nn
import torchvision
from torch.autograd import Variable
from torchvision import transforms, utils
from torchvision.utils import save_image

from datasets import ImageDataset
from model import Generator


parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', type=str, required=True,)
parser.add_argument('--G1_checkpoint_path', type=str, required=True,)
parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--lrG', type=float, default=0.002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument("--batch_size", type=int, default=4, help="number of batch_size")

args = parser.parse_args()


G1 = Generator(in_c=1,out_c=3).cuda()
criterion=torch.nn.MSELoss()


optimizer=torch.optim.Adam(G1.parameters(), lr=args.lrG,betas=(args.beta1, args.beta2))




input_transform = transforms.Compose([
            transforms.ToTensor(),
])


output_transform = transforms.Compose([
            transforms.ToTensor(),
])

dataloader = torch.utils.data.DataLoader(ImageDataset(path=args.datasets_path, transforms_A=input_transform, transforms_B=output_transform,unaligned=False),
                        batch_size=args.batch_size, shuffle=True)

print(len(dataloader))
G1_checkpoint_path=args.G1_checkpoint_path
if not os.path.exists(G1_checkpoint_path):
    os.makedirs(G1_checkpoint_path)
    print('make_checkpoint_path!')


def train():
    epoch=199
    for i in range(epoch):
        j=0
        G1.train()
        torch.cuda.synchronize()
        start=time.time()
        for batch in dataloader:
            print(type((batch['A'])))
            input_imgs = Variable((batch['A']))
            target_imgs = Variable((batch['B']))

            input_imgs=input_imgs.cuda()
            target_imgs=target_imgs.cuda()



            optimizer.zero_grad()
            output_imgs=G1(input_imgs)


            loss=criterion(output_imgs,target_imgs)

            loss.backward()
            optimizer.step()
            if(j%100==0):
                print('epoch: %d'%i,' loss_g: ',loss)


            j+=1
        torch.cuda.synchronize()
        end=time.time()
        print(end-start)
        torch.save(G1.state_dict(), f='./'+G1_checkpoint_path+'/G1_%d.pth' % i)

if __name__ == '__main__':
    train()
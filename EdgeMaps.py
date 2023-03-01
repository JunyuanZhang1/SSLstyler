
import os
import argparse
import torch
from PIL import Image

import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--org_img_folder', type=str, required=True,)
parser.add_argument('--save_path', type=str, required=True,)

args = parser.parse_args()

def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel

def GaussianBlur(batch_img, ksize=3, sigma=0):
    kernel = getGaussianKernel(ksize, sigma)
    B, C, H, W = batch_img.shape
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,
                           stride=1, padding=0, groups=C)
    return weighted_pix
def getFileList(dir, Filelist, ext=None):

    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist


org_img_folder = args.org_img_folder

# 检索文件
imglist = getFileList(org_img_folder, [], 'jpg')

save_path=args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'/train/A')
    os.makedirs(save_path+'/train/B')
    os.makedirs(save_path+'/train/C')

transform_128 = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
])
transform_512 = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor(),
])

i=0
for img in imglist:
    o = Image.open(img)
# -----------128-----------------------------#
    o_128 = transform_128(o).unsqueeze(0)
    o_128_blur = GaussianBlur(o_128,ksize=11)
    o_128_blur = o_128_blur.squeeze(0) * 255
    o_128_blur = o_128_blur.permute(1, 2, 0)
    o_128_blur = np.array(o_128_blur)
    o_128_blur = o_128_blur.astype('uint8')
    r1 = cv2.Canny(o_128_blur,20,40)
    cv2.imwrite(save_path+'/train/C/%d.jpg' % (i), r1)

#-----------512-----------------------------#
    o_512 = transform_512(o).unsqueeze(0)
    o_512_blur = GaussianBlur(o_512,ksize=15)
    o_512_blur = o_512_blur.squeeze(0) * 255
    o_512_blur = o_512_blur.permute(1, 2, 0)
    o_512_blur = np.array(o_512_blur)
    o_512_blur = o_512_blur.astype('uint8')
    r2 = cv2.Canny(o_512_blur, 20,40)

    o_512 = o_512.squeeze(0) * 255
    o_512 = o_512.permute(1, 2, 0)
    o_512 = np.array(o_512)
    o_512 = o_512.astype('uint8')
    #
    b, g, r = cv2.split(o_512)
    o_512 = cv2.merge([r, g, b])

    cv2.imwrite(save_path+'/train/A/%d.jpg' % (i), r2)
    cv2.imwrite(save_path+'/train/B/%d.jpg' % (i), o_512)

    i += 1
    print(i)

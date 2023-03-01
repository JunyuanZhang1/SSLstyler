import argparse

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

from model import Generator
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, required=True,)
parser.add_argument('--save_path', type=str, required=True,)
parser.add_argument('--G2_dir', type=str, required=True,)
parser.add_argument('--G1_dir', type=str, required=True,)
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

G2=Generator(out_c=3).cuda()
G2.load_state_dict(torch.load(args.G2_dir))
G1 = Generator(out_c=3).cuda()
G1.load_state_dict(torch.load(args.G1_dir))

transform_128 = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
])
transform_512 = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor(),
])


o = Image.open(args.img_dir)
# -----------128-----------------------------#
o_128 = transform_128(o).unsqueeze(0)
o_128_blur = GaussianBlur(o_128,ksize=11)
o_128_blur = o_128_blur.squeeze(0) * 255
o_128_blur = o_128_blur.permute(1, 2, 0)
o_128_blur = np.array(o_128_blur)
o_128_blur = o_128_blur.astype('uint8')
r1 = cv2.Canny(o_128_blur,20,40)
r1 = transform_128(r1).unsqueeze(0)

#-----------512-----------------------------#
o_512 = transform_512(o).unsqueeze(0)
o_512_blur = GaussianBlur(o_512,ksize=15)
o_512_blur = o_512_blur.squeeze(0) * 255
o_512_blur = o_512_blur.permute(1, 2, 0)
o_512_blur = np.array(o_512_blur)
o_512_blur = o_512_blur.astype('uint8')
r2 = cv2.Canny(o_512_blur, 20,40)
r2 = transform_128(r2).unsqueeze(0)

output=G2(r2,localenhancer_feature=G1(r1,feature_mode=True))

save_image(output.data, args.save_path , normalize=True)



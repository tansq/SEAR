import numpy as np
import os
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import argparse

parser = argparse.ArgumentParser("eval adversarial sample")
parser.add_argument("--ori_dir", type=str, default='', help='original sample dir')
parser.add_argument("--adv_dir", type=str, default='', help='adversarial sample dir')
parser.add_argument("--gt_dir", type=str, default='', help='groundtruth dir')
args = parser.parse_args()

ori_dir = args.ori_dir
adv_dir = args.adv_dir
gt_dir = args.gt_dir
energy = []
psnr = []
ssim = []
img_names = os.listdir(ori_dir)
with tqdm(total=len(img_names)) as pbar:
    for idx, im_name in enumerate(img_names):
        pbar.update(1)
        pbar.set_description(f"running:{idx + 1}")
        ori = cv2.imread(ori_dir + im_name)
        adv = cv2.imread(adv_dir + im_name)
        psnr.append(peak_signal_noise_ratio(ori, adv))

        ssim.append(structural_similarity(ori, adv, multichannel=True))
        gt = cv2.imread(gt_dir + im_name, 0) / 255
        diff = np.var((adv - ori).flatten())
        manipulation = np.var((adv[gt == 1] - ori[gt == 1]).flatten())
        non_manipulation = np.var((adv[gt == 0] - ori[gt == 0]).flatten())
        energy.append((diff / non_manipulation) / (manipulation / non_manipulation))
print("energy:", np.mean(np.array(energy)))
print("ssim:", np.mean(np.array(ssim)))
print("psnr:", np.mean(np.array(psnr)))

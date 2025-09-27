import cv2
import numpy as np
import os
import heapq


def perturb_mean_var(ori, adv, gt):
    perturb = np.abs(adv.astype(np.float) - ori.astype(np.float))
    perturb_tamper = perturb * gt
    perturb_tamper = perturb_tamper.reshape(-1)
    perturb_tamper = perturb_tamper[perturb_tamper != 0]
    perturb_none_tamper = perturb * (1 - gt)
    perturb_none_tamper = perturb_none_tamper.reshape(-1)
    perturb_none_tamper = perturb_none_tamper[perturb_none_tamper != 0]
    if perturb_tamper.shape[0] != 0:
        tamper_mean = np.round(np.mean(perturb_tamper), 4)
        tamper_variance = np.round(np.var(perturb_tamper), 4)
    else:
        tamper_mean = 0
        tamper_variance = 0
    none_tamper_mean = max(np.round(np.mean(perturb_none_tamper), 4), 0)
    none_tamper_variance = max(np.round(np.var(perturb_none_tamper), 4), 0)
    return tamper_mean, tamper_variance, none_tamper_mean, none_tamper_variance


ori_dir = ""
adv_dir = ""
no_adv_dir = ""
gt_dir = ""
img_names = os.listdir(ori_dir)
tm = []
tv = []
nm = []
nv = []
prior = 0
none_prior = 0
diff = abs(none_prior - prior)
for idx, im_name in enumerate(img_names):
    ori = cv2.imread(ori_dir + im_name)
    adv = cv2.imread(adv_dir + im_name)
    no_adv = cv2.imread(no_adv_dir + im_name)
    gt = cv2.imread(gt_dir + im_name) / 255
    tam_m, tam_v, ntam_m, ntam_v = perturb_mean_var(ori, adv, gt)
    prior = abs(ntam_v - tam_v)
    tam_m, tam_v, ntam_m, ntam_v = perturb_mean_var(ori, no_adv, gt)
    none_prior = abs(ntam_v - tam_v)
    if abs(none_prior - prior) > diff:
        diff = abs(none_prior - prior)
        print(im_name)
    # tm.append(tam_m)
    # tv.append(tam_v)
    # nm.append(ntam_m)
    # nv.append(ntam_v)
    # print(f"tamper mean:{tamper_mean} tamper var:{tamper_variance} none tamper mean:{none_tamper_mean} none tamper var:{none_tamper_variance}")
# print("avg tamper mean:", sum(tm) / len(tm))
# print("avg tamper var:", sum(tv) / len(tv))
# print("avg none tamper mean:", sum(nm) / len(nm))
# print("avg none tamper var:", sum(nv) / len(nv))

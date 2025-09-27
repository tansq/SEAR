import argparse
import os
from tqdm import tqdm
import numpy as np
import cv2
from sklearn import metrics
from model.base import D
from model.mantranet import load_pretrain_model_by_index
from model.ManTraNetv3 import SPAN
from model.models import locatenet

parser = argparse.ArgumentParser("transitional attack")
parser.add_argument("--detector", type=str, default='locatenet', help='Choose attack detector')
parser.add_argument("--input_size", type=int, default=512, help='size of input image')
parser.add_argument("--adv_dir", type=str, default='', help='adversarial sample dir')
parser.add_argument("--gt_dir", type=str, default='', help='groundtruth dir')
parser.add_argument("--save_path", type=str, default='')
parser.add_argument("--detector_weight", type=str, default='', help='path of detector weight')
parser.add_argument("--gpu_id", type=str, default='0', help='gpu devices')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

detector_model = args.detector
input_shape = (args.input_size, args.input_size, 3)
detector_model_path = args.detector_weight
adv_path = args.adv_dir
groundtruth_path = args.gt_dir
adv_save = os.path.join(args.save_path, f"{detector_model}/{adv_path.split('/')[-3]}/adv/")
adv_mask_save = os.path.join(args.save_path, f"{detector_model}/{adv_path.split('/')[-3]}/adv_pred/")
if not os.path.exists(adv_save):
    os.makedirs(adv_save)
    os.makedirs(adv_mask_save)

if detector_model == "locatenet":
    detector = D(input_shape=input_shape)
    detector.load_weights(detector_model_path)
elif detector_model == "mantranet":
    detector = load_pretrain_model_by_index(4, "./pretrained_weights/mantranet")
    detector.load_weights(detector_model_path)
elif detector_model == "span":
    detector = SPAN(input_size=input_shape[0])
    detector.load_weights(detector_model_path)
elif detector_model == "satfl":
    detector = locatenet(0.0002)
else:
    raise Exception('wrong model!')

masks = None
probs = None
auc_list = []
f1_list = []
img_name = os.listdir(adv_path)
with tqdm(total=len(img_name)) as pbar:
    for idx, line in enumerate(img_name):
        pbar.update(1)
        pbar.set_description(f"running:{idx + 1}")
        mask = cv2.imread(os.path.join(groundtruth_path, line))
        mask = cv2.resize(mask, input_shape[:-1], interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(np.mean(mask, axis=-1), axis=-1) / 255
        img = cv2.imread(os.path.join(adv_path, line))
        img = cv2.resize(img, input_shape[:-1], interpolation=cv2.INTER_AREA)
        cv2.imwrite(adv_save + str(idx) + ".png", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # print rgb
        img = img[:, :, ::-1]
        if detector_model != "satfl":
            prob = detector.predict(np.expand_dims(img / 255.0 * 2 - 1, axis=0))
        else:
            _, prob = detector.predict(np.expand_dims(img / 255.0, axis=0))
        cv2.imwrite(adv_mask_save + str(idx) + ".png", (prob.squeeze() * 255).astype(np.uint8),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        auc = metrics.roc_auc_score(np.round(mask.flatten()).astype(np.int), prob.flatten())
        f1 = metrics.f1_score(np.round(mask.flatten()).astype(np.int), np.round(prob.flatten()).astype(np.int))
        auc_list.append(auc)
        f1_list.append(f1)
print("auc:", sum(auc_list) / len(auc_list))
print("f1:", sum(f1_list) / len(f1_list))
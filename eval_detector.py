import argparse
import os
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import time
from model.base import D
from model.mantranet import Mantranet
from model.ManTraNetv3 import SPAN
from utils.dataloader import load_mask, load_rgb, get_data
from model.models import locatenet, RefinedNet

parser = argparse.ArgumentParser("attack detector")
parser.add_argument("--database", type=str, default='nist', help='Choose attack database')
parser.add_argument("--detector", type=str, default='locatenet', help='Choose attack detector')
parser.add_argument("--input_size", type=int, default=512, help='size of input image')
parser.add_argument("--detector_weight", type=str, default='', help='path of detector weight')
parser.add_argument("--gpu_id", type=str, default='0', help='gpu devices')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


database = args.database
detector_model = args.detector
input_shape = (args.input_size, args.input_size, 3)
detector_model_path = args.detector_weight

if detector_model == "locatenet":
    detector = locatenet(lr=0.0002)
elif detector_model == "mantranet":
    detector = Mantranet(4, "./pretrained_weights/mantranet")
elif detector_model == "span":
    detector = SPAN(input_size=input_shape[0])
elif detector_model == "refinednet":
    detector = RefinedNet(lr=0.0002)
else:
    raise Exception('wrong model!')
detector.load_weights(detector_model_path)

base_path, _, valid_file = get_data(database)
print("Total images: ", len(valid_file))
auc_list = []
f1_list = []
time_list = []
with tqdm(total=len(valid_file)) as pbar:
    for idx, line in enumerate(valid_file):
        pbar.update(1)
        pbar.set_description(f"running:{idx+1}")
        mask = load_mask(base_path, line, args.input_size)
        img = load_rgb(base_path, line, args.input_size)
        ori_img = img
        if database == "imd":
            img = img.astype('float32') / 255.
        else:
            img = img.astype('float32') / 255. * 2 - 1
        start_time = time.time()
        if detector_model == "locatenet":
            _, prob = detector.predict(np.expand_dims(img, axis=0))
        elif detector_model == "refinednet":
            prob = detector.predict(np.expand_dims(img, axis=0))
        else:
            prob = detector.predict(np.expand_dims(img, axis=0))[1]
        time_list.append(time.time() - start_time)
        auc = metrics.roc_auc_score(np.round(mask.flatten()).astype(np.int), prob.flatten())
        f1 = metrics.f1_score(np.round(mask.flatten()).astype(np.int), np.round(prob.flatten()).astype(np.int))
        auc_list.append(auc)
        f1_list.append(f1)
print("auc:", sum(auc_list) / len(auc_list))
print("f1:", sum(f1_list) / len(f1_list))
print("inference avg time:", sum(time_list[:20]) / len(time_list[:20]))


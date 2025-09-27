import os
from tqdm import tqdm
import cv2
import time
import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn import metrics
import argparse
from attack.attacker import FGSM, MI_FGSM, BIM, GAN, GAN_limit
from model.base import D, G, Generator
from model.mantranet import load_pretrain_model_by_index
from model.ManTraNetv3 import SPAN
from utils.dataloader import load_rgb, load_mask, get_data
from utils.metrics import cal_tp_fp

parser = argparse.ArgumentParser("attack detector")
parser.add_argument("--database", type=str, default='nist', help='Choose attack database')
parser.add_argument("--detector", type=str, default='locatenet', help='Choose attack detector')
parser.add_argument("--attack", type=str, default='fgsm', help='Choose attack method')
parser.add_argument("--target", type=int, default=0, help='Whether there is a target attack')
parser.add_argument("--input_size", type=int, default=512, help='size of input image')
parser.add_argument("--minclip", type=int, default=-1, help='clip min value')
parser.add_argument("--maxclip", type=int, default=1, help='clip max value')
parser.add_argument("--save_img", type=int, default=1, help='Whether to save the image')
parser.add_argument("--detector_weight", type=str, default='', help='path of detector weight')
parser.add_argument("--generator_weight", type=str, default='', help='path of generator weight')
parser.add_argument("--save_path", type=str, default='', help='save path')
parser.add_argument("--gpu_id", type=str, default='0', help='gpu devices')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
# stop warnning
tf.logging.set_verbosity(tf.logging.ERROR)



database = args.database
detector_model = args.detector
save_img = bool(args.save_img)
attack_method = args.attack
target = bool(args.target)
input_shape = (args.input_size, args.input_size, 3)
detector_model_path = args.detector_weight
generator_model_path = args.generator_weight
adv_save = os.path.join(args.save_path, f"{database}/{detector_model}/{attack_method}_targeted_{target}/adv/")
adv_mask_save = os.path.join(args.save_path, f"{database}/{detector_model}/{attack_method}_targeted_{target}/adv_pred/")
ori_save = os.path.join(args.save_path, f"{database}/{detector_model}/{attack_method}_targeted_{target}/ori/")
ori_mask_save = os.path.join(args.save_path, f"{database}/{detector_model}/{attack_method}_targeted_{target}/gt/")
ori_pred_save = os.path.join(args.save_path, f"{database}/{detector_model}/{attack_method}_targeted_{target}/ori_pred/")
if not os.path.exists(ori_save):
    os.makedirs(ori_save)
    os.makedirs(adv_save)
    os.makedirs(adv_mask_save)
    os.makedirs(ori_mask_save)
    os.makedirs(ori_pred_save)

sess = K.get_session()
if detector_model == "locatenet":
    detector = D(input_shape=input_shape)
    detector.load_weights(detector_model_path)
elif detector_model == "mantranet":
    detector = load_pretrain_model_by_index(4, "./pretrained_weights/mantranet")
    detector.load_weights(detector_model_path)
elif detector_model == "span":
    detector = SPAN(input_size=input_shape[0])
    detector.load_weights(detector_model_path)
else:
    raise Exception('wrong model!')

if attack_method == "fgsm":
    attack = FGSM(input_shape=input_shape, model=detector, eps=0.01, lower=args.minclip, upper=args.maxclip, sess=sess, target=target)
elif attack_method == "bim":
    attack = BIM(input_shape=input_shape, model=detector, eps=0.01, iter=10, lower=args.minclip, upper=args.maxclip, sess=sess, target=target)
elif attack_method == "mi-fgsm":
    attack = MI_FGSM(input_shape=input_shape, model=detector, eps=0.01, momentum=1.0, iter=10, lower=args.minclip, upper=args.maxclip, sess=sess, target=target)
elif attack_method == "gan":
    generator = G(input_shape=input_shape, block_type="vgg")
    generator.load_weights(generator_model_path)
    attack = GAN(generator, lower=args.minclip, upper=args.maxclip)
elif attack_method == "gan_limit":
    generator = G(input_shape=input_shape)
    generator.load_weights(generator_model_path)
    attack = GAN_limit(generator, lower=args.minclip, upper=args.maxclip)
elif attack_method == "advgan":
    generator = Generator(input_shape=input_shape)
    generator.load_weights(generator_model_path)
    attack = GAN(generator, lower=args.minclip, upper=args.maxclip)
else:
    raise Exception("error attack!")

auc_list = []
f1_list = []
reversed_auc_list = []
reversed_f1_list = []
time_list = []
tp = 0
fp = 0
base_path, _, valid_file = get_data(database)
print("database:", database)
print("attack method:", attack_method)
print(f"min clip:{args.minclip}  max clip:{args.maxclip}")
print("Total images: ", len(valid_file))
with tqdm(total=len(valid_file)) as pbar:
    for idx, line in enumerate(valid_file[:]):
        pbar.update(1)
        pbar.set_description(f"running:{idx+1}")
        mask = load_mask(base_path, line, img_size=args.input_size)
        img = load_rgb(base_path, line, img_size=args.input_size)
        ori_img = img
        img = img.astype('float32') / 255. * 2 - 1
        start_time = time.time()
        adv = attack.generate(img, mask)[0]
        time_list.append(time.time() - start_time)
        adv_img = np.round((adv + 1) / 2 * 255.0).astype(np.uint8)
        adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        prob = detector.predict(np.expand_dims(adv, axis=0))
        ori_pred = detector.predict(np.expand_dims(img, axis=0))
        if save_img:
            cv2.imwrite(adv_save + str(idx) + ".png", adv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(ori_save + str(idx) + ".png", ori_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(ori_mask_save + str(idx) + ".png", mask * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(adv_mask_save + str(idx) + ".png", (prob.squeeze() * 255).astype(np.uint8),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(ori_pred_save + str(idx) + ".png", (ori_pred.squeeze() * 255).astype(np.uint8),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        auc = metrics.roc_auc_score(np.round(mask.flatten()).astype(np.int), prob.flatten())
        f1 = metrics.f1_score(np.round(mask.flatten()).astype(np.int), np.round(prob.flatten()).astype(np.int))
        r_auc = metrics.roc_auc_score(np.round(1 - mask.flatten()).astype(np.int), prob.flatten())
        r_f1 = metrics.f1_score(np.round(1 - mask.flatten()).astype(np.int), np.round(prob.flatten()).astype(np.int))
        # tp_, fp_ = cal_tp_fp(np.round(mask.flatten()).astype(np.int),  np.round(prob.flatten()).astype(np.int))
        # tp += tp_
        # fp += fp_
        auc_list.append(auc)
        f1_list.append(f1)
        reversed_auc_list.append(r_auc)
        reversed_f1_list.append(r_f1)
print("auc:", sum(auc_list) / len(auc_list))
print("f1:", sum(f1_list) / len(f1_list))
print("reversed_auc:", sum(reversed_auc_list) / len(reversed_auc_list))
print("reversed_f1:", sum(reversed_f1_list) / len(reversed_f1_list))
print("attack avg time:", sum(time_list[:20]) / len(time_list[:20]))
# print("tp:", tp)
# print("fp:", fp)
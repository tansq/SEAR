import argparse
import os

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from utils.trainer import advgan_train

parser = argparse.ArgumentParser("advgan")
parser.add_argument("--database", type=str, default='nist', help='Choose attack database')
parser.add_argument("--detector", type=str, default='locatenet', help='Choose attack detector')
parser.add_argument("--input_size", type=int, default=512, help='size of input image')
parser.add_argument("--detector_weight", type=str, default='', help='path of detector weight')
parser.add_argument("--weight_save_path", type=str, default='', help='path of save weight')
parser.add_argument("--check_save_path", type=str, default='', help='examine the generated image')
parser.add_argument("--log_file", type=str, default='', help='log file')
parser.add_argument("--epoch", type=int, default=20, help='size of input image')
parser.add_argument("--batch", type=int, default=4, help='size of input image')
parser.add_argument("--lr", type=float, default=0.0002, help='learning rate')
parser.add_argument("--alpha", type=int, default=10, help='weight of bce loss')
parser.add_argument("--beta", type=int, default=1, help='weight of hinge loss')
parser.add_argument("--seed", type=int, default=2021, help='numpy seed')
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

input_shape = (args.input_size, args.input_size, 3)
log_file = os.path.basename(args.log_file)
log_file_path = args.log_file.split(log_file)[0]

advgan_train(model_save_path=args.weight_save_path,
             check_save_path=args.check_save_path,
             log_file_path=log_file_path,
             log_file=log_file,
             detector_model=args.detector,
             input_shape=input_shape,
             dataset=args.database,
             epoch=args.epoch,
             batch=args.batch,
             lr=args.lr,
             seed=args.seed,
             detector_weight=args.detector_weight,
             alpha=args.alpha,
             beta=args.beta)

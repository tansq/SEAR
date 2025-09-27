import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import argparse
from utils.trainer import blackbox_attack

parser = argparse.ArgumentParser("train genenator")
parser.add_argument("--database", type=str, default='nist', help='Choose attack database')
parser.add_argument("--target_model", type=str, default='mantranet', help='Choose attack detector')
parser.add_argument("--block_type", type=str, default='mb', help='the block type of the generator')
parser.add_argument("--input_size", type=int, default=512, help='size of input image')
parser.add_argument("--target_weight", type=str, default='', help='path of target model weight')
parser.add_argument("--proxy_weight", type=str, default='', help='path of proxy model weight')
parser.add_argument("--weight_save_path", type=str, default='', help='path of save weight')
parser.add_argument("--check_save_path", type=str, default='', help='examine the generated image')
parser.add_argument("--log_file", type=str, default='', help='log file')
parser.add_argument("--epoch", type=int, default=20, help='size of input image')
parser.add_argument("--batch", type=int, default=4, help='size of input image')
parser.add_argument("--lr", type=float, default=0.0002, help='learning rate')
parser.add_argument("--bce_weight", type=int, default=10, help='weight of bce loss')
parser.add_argument("--hinge_weight", type=int, default=1, help='weight of hinge loss')
parser.add_argument("--seed", type=int, default=2021, help='numpy seed')
parser.add_argument("--gan_gpu_id", type=str, default='0', help='gan gpu devices')
parser.add_argument("--tg_gpu_id", type=str, default='1', help='gan gpu devices')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gan_gpu_id + ',' + args.tg_gpu_id
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)
# stop warnning
tf.logging.set_verbosity(tf.logging.ERROR)

input_shape = (args.input_size, args.input_size, 3)
log_file = os.path.basename(args.log_file)
log_file_path = args.log_file.split(log_file)[0]

blackbox_attack(model_save_path=args.weight_save_path,
                check_save_path=args.check_save_path,
                log_file_path=log_file_path,
                log_file=log_file,
                target_model_name=args.target_model,
                input_shape=input_shape,
                dataset=args.database,
                epoch=args.epoch,
                batch=args.batch,
                lr=args.lr,
                seed=args.seed,
                target_weight=args.target_weight,
                generator_block_type=args.block_type,
                bce_w=args.bce_weight,
                hinge_w=args.hinge_weight)

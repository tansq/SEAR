from model.ManTraNetv3 import SPAN
from model.mantranet import Mantranet
from model.base import G, GAN_V2, dynamic_distillation, D
from model.models import locatenet
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# s = SPAN()
# generator = G(input_shape=(256, 256, 3), block_type="vgg")
# gan = GAN_V2(generator, s, "span", input_shape=(256, 256, 3), lr=0.0002, bce_weight=10, hinge_weight=1)
# print(gan.summary())
# l = D((512, 512, 3))
# m = Mantranet(4, "./pretrained_weights/mantranet")
# dynamic_distillation(l, m, (512, 512, 3))
m = locatenet(0.0002)
m.load_weights("")
print(m.summary())


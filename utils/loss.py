# from keras_contrib.losses import DSSIMObjective as dssim
from keras import backend as K
from tensorflow.losses import huber_loss
import tensorflow as tf
from keras.losses import binary_crossentropy


# def mix_loss(y_true, y_pred):
#     alpha = 0.86
#     dloss = dssim()
#     # print("+++++++++++++++++++", y_true)
#     # print("+++++++++++++++++++", y_pred)
#     # losses = alpha * huber_loss(y_true,y_pred)
#     losses = alpha * dloss(y_true, y_pred) + (1 - alpha) * huber_loss(y_true, y_pred)
#     # losses = huber_loss(y_true, y_pred)
#     return losses


def psnr_ssim(y_true, y_pred):
    # n_samples = K.int_shape(y_true)[0]
    # mse = keras.losses.mean_squared_error(y_true, y_pred)
    # psnr = -10.0 * K.log(mse+K.epsilon()) / np.log(10)
    ps = K.round(tf.image.psnr(y_true, y_pred, 2) * 100) + tf.image.ssim(y_true, y_pred, 2)
    return ps


def binary_loss(y_true, y_pred):
    return -binary_crossentropy(y_true, y_pred)


def cw_loss(y_true, y_pred):
    confidence = 0
    real = K.sum(y_true * y_pred, axis=[1, 2, 3])
    other = K.max((1.0 - y_true) * y_pred - y_true * 100, axis=[1, 2, 3])
    loss0 = K.maximum(real - other + confidence, 0)
    # loss1 = K.mean(K.square(y_pred - y_true))
    return loss0


def hinge_loss(y_true, y_pred):
    c = 0
    l2 = K.sqrt(K.sum(y_pred ** 2, axis=[1, 2, 3]))
    # l2 = K.maximum(l2, 0)
    # li = K.max(K.abs(y_pred - y_true), axis=[1, 2, 3])
    # return K.mean(li * 10 + l2)
    return l2


def pretext_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred), axis=[1, 2, 3])


def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def IoU_fun(eps=1e-6):
    def IoU(y_true, y_pred):
        # if np.max(y_true) == 0.0:
        #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        #
        ious = K.mean((intersection + eps) / (union + eps), axis=0)
        return K.mean(ious)

    return IoU


def IoU_loss_fun(eps=1e-6):
    def IoU_loss(y_true, y_pred):
        return 1 - IoU_fun(eps=eps)(y_true=y_true, y_pred=y_pred)

    return IoU_loss


def bce_focal_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + binary_focal_loss()(y_true, y_pred)


def bce_iou_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + IoU_loss_fun()(y_true, y_true)


def focal_loss(y_true, y_pred):
    return binary_focal_loss()(y_true, y_pred)


def iou_loss(y_true, y_pred):
    return IoU_loss_fun()(y_true, y_true)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)
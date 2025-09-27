import tensorflow as tf
import numpy as np
from keras.losses import binary_crossentropy
import cv2


class FGSM(object):
    def __init__(self, input_shape, model, eps=0.01, lower=0, upper=1, sess=None, target=False):
        self.eps = eps * (upper - lower)
        self.lower = lower
        self.upper = upper
        self.sess = sess
        self.model = model
        self.y_true_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
        self.y_true = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)),
                                  dtype=tf.float32)
        self.y_true_op = tf.assign(self.y_true, self.y_true_ph)
        self.y_pred = self.model.output
        if target:
            self.y_target = np.zeros((1, input_shape[0], input_shape[1], 1), dtype=np.float32)
            self.loss = binary_crossentropy(self.y_true, self.y_pred) - binary_crossentropy(self.y_target, self.y_pred)
        else:
            self.loss = binary_crossentropy(self.y_true, self.y_pred)
        self.gradient = tf.gradients(self.loss, self.model.input)
        self.gradient = self.gradient[0]
        self.adv_noise = tf.sign(self.gradient) * self.eps

    def generate(self, image, mask):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            y_t = np.expand_dims(mask, axis=0).astype(np.float32)
        else:
            y_t = mask
        x_min = np.clip(image - self.eps, self.lower, self.upper)
        x_max = np.clip(image + self.eps, self.lower, self.upper)
        # tamper_area = np.repeat(y_t.reshape(1, y_t.shape[1], y_t.shape[2], 1), 3, axis=-1)
        tamper_area = 1
        self.sess.run(self.y_true_op, feed_dict={self.y_true_ph: y_t})
        noise = self.sess.run(self.adv_noise, feed_dict={self.model.input: image})
        x_adv = image + noise * tamper_area
        x_adv = np.clip(x_adv, x_min, x_max)
        return x_adv


class BIM(object):
    def __init__(self, input_shape, model, eps=0.01, iter=10, lower=0, upper=1, sess=None, target=False):
        self.eps = eps * (upper - lower)
        self.eps_iter = self.eps / iter
        self.lower = lower
        self.upper = upper
        self.iter = iter
        self.sess = sess
        self.model = model
        self.y_true_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
        self.y_true = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)),
                                  dtype=tf.float32)
        self.y_true_op = tf.assign(self.y_true, self.y_true_ph)
        self.y_pred = self.model.output
        if target:
            self.y_target = np.zeros((1, input_shape[0], input_shape[1], 1), dtype=np.float32)
            self.loss = binary_crossentropy(self.y_true, self.y_pred) - binary_crossentropy(self.y_target, self.y_pred)
        else:
            self.loss = binary_crossentropy(self.y_true, self.y_pred)
        self.gradient = tf.gradients(self.loss, self.model.input)
        self.gradient = self.gradient[0]
        self.adv_noise = tf.sign(self.gradient) * self.eps_iter

    def generate(self, image, mask):
        if len(image.shape) == 3:
            x_adv = np.expand_dims(image, axis=0)
            y_t = np.expand_dims(mask, axis=0).astype(np.float32)
        else:
            x_adv = image
            y_t = mask
        x_min = np.clip(image - self.eps, self.lower, self.upper)
        x_max = np.clip(image + self.eps, self.lower, self.upper)
        self.sess.run(self.y_true_op, feed_dict={self.y_true_ph: y_t})
        # mask_dilated = mask
        # tamper_area = np.repeat(y_t.reshape(1, y_t.shape[1], y_t.shape[2], 1), 3, axis=-1)
        tamper_area = 1
        for i in range(self.iter):
            noise = self.sess.run(self.adv_noise, feed_dict={self.model.input: x_adv})
            x_adv = x_adv + noise * tamper_area
            x_adv = np.clip(x_adv, x_min, x_max)
        return x_adv


class MI_FGSM(object):
    def __init__(self, input_shape, model, eps=0.01, momentum=1.0, iter=10, lower=0, upper=1, sess=None, target=False):
        self.eps = eps * (upper - lower)
        self.eps_iter = self.eps / iter
        self.lower = lower
        self.upper = upper
        self.momentum = momentum
        self.iter = iter
        self.sess = sess
        self.model = model
        self.y_true_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
        self.y_true = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)),
                                  dtype=tf.float32)
        self.y_true_op = tf.assign(self.y_true, self.y_true_ph)
        self.y_pred = self.model.output
        if target:
            self.y_target = np.zeros((1, input_shape[0], input_shape[1], 1), dtype=np.float32)
            self.loss = binary_crossentropy(self.y_true, self.y_pred) - binary_crossentropy(self.y_target, self.y_pred)
        else:
            self.loss = binary_crossentropy(self.y_true, self.y_pred)
        self.gradient = tf.gradients(self.loss, self.model.input)
        self.gradient = self.gradient[0]

    def generate(self, image, mask):
        if len(image.shape) == 3:
            x_adv = np.expand_dims(image, axis=0)
            y_t = np.expand_dims(mask, axis=0).astype(np.float32)
        else:
            x_adv = image
            y_t = mask
        grad = np.zeros(x_adv.shape)
        x_min = np.clip(image - self.eps, self.lower, self.upper)
        x_max = np.clip(image + self.eps, self.lower, self.upper)
        # tamper_area = np.repeat(y_t.reshape(1, y_t.shape[1], y_t.shape[2], 1), 3, axis=-1)
        tamper_area = 1
        self.sess.run(self.y_true_op, feed_dict={self.y_true_ph: y_t})
        for i in range(self.iter):
            g = self.sess.run(self.gradient, feed_dict={self.model.input: x_adv})
            g /= np.linalg.norm(g.reshape(-1), ord=1)
            grad = self.momentum * grad + g
            x_adv = x_adv + self.eps_iter * np.sign(grad) * tamper_area
            x_adv = np.clip(x_adv, x_min, x_max)
        return x_adv


class GAN(object):
    def __init__(self, model, lower=0, upper=1):
        self.model = model
        self.lower = lower
        self.upper = upper

    def generate(self, image, mask=None):
        if len(image.shape) == 3:
            img = np.expand_dims(image, axis=0)
        else:
            img = image
        adv_pert = self.model.predict(img)
        return np.clip(adv_pert + img, self.lower, self.upper)


class GAN_limit(object):
    def __init__(self, model, lower=0, upper=1):
        self.model = model
        self.lower = lower
        self.upper = upper

    def generate(self, image, mask):
        if len(image.shape) == 3:
            img = np.expand_dims(image, axis=0)
        else:
            img = image
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        # mask_dilated = cv2.dilate(mask, kernel).reshape(512, 512, 1)
        mask_dilated = mask
        tamper_area = np.repeat(mask_dilated.reshape(1, mask.shape[0], mask.shape[1], 1), 3, axis=-1)
        # tamper_area = 1
        adv_pert = self.model.predict(img) * tamper_area
        return np.clip(adv_pert + img, self.lower, self.upper)
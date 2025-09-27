from keras import backend as K
import tensorflow as tf
import numpy as np


def auc(y_true, y_pred):
    # y_p = tf.ceil(y_pred, name=None)
    # y_true = layers.Average()([y_true])
    # y_pred = layers.Average([y_pred])
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    # auc = roc_auc_score(y_true,y_pred)
    return auc


def f1(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    precision = tf.metrics.precision(y_true, y_pred)[1]
    recall = tf.metrics.recall(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def cal_tp_fp(y_true, y_pred):
    none_tamper = np.where(y_true == 0)[0]
    tamper = np.where(y_true != 0)[0]
    tp = np.count_nonzero(y_pred[tamper] == 1)
    fp = np.count_nonzero(y_pred[none_tamper] == 1)
    fn = np.count_nonzero(tamper) - tp
    tn = np.count_nonzero(none_tamper) - fp
    # print("tp: ", tp)
    # print("fp: ", fp)
    return tp, fp

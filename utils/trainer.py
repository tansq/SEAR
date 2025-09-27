import os
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import cv2
import logging
import keras.backend as K
import tensorflow as tf
from utils.dataloader import get_data, load_img_array
from model.base import G, D, dynamic_distillation, Generator, Discriminator, advgan
from model import base
from model.mantranet import Mantranet
from model.ManTraNetv3 import SPAN
from model.models import locatenet, RefinedNet
from attack.attacker import FGSM, BIM, MI_FGSM, GAN, GAN_limit
from utils.loss import hinge_loss, binary_loss


def train_check_img(check_img, generator, epoch, save_path="./check/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    perturb = generator.predict(check_img)
    g_img = perturb + check_img
    res = np.concatenate((check_img, g_img), axis=2)
    res = (res + 1) / 2
    res = np.clip(res, 0., 1.0)
    res = (res[0] * 255.0).astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path + str(epoch) + "_" + str(np.sum(np.abs(perturb))) + ".png", res,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def train_check_img_v2(check_img, check_mask, generator, epoch, save_path="./check/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    perturb = generator.predict(check_img) * check_mask
    g_img = perturb + check_img
    res = np.concatenate((check_img, g_img), axis=2)
    res = (res + 1) / 2
    res = np.clip(res, 0., 1.0)
    res = (res[0] * 255.0).astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path + str(epoch) + "_" + str(np.sum(np.abs(perturb))) + ".png", res,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        if l.name == 'depthwise_conv2d_1':
            l.trainable = False
            continue
        l.trainable = val


def transfer_mask(mask):
    h, w = mask.shape[:2]
    y, x, _ = np.where(mask == 1)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    left_top = center_x ** 2 + center_y ** 2
    right_top = (w - center_x) ** 2 + center_y ** 2
    left_bottom = center_x ** 2 + (h - center_y) ** 2
    right_bottom = (w - center_x) ** 2 + (h - center_y) ** 2
    dis_list = [left_top, right_top, left_bottom, right_bottom]
    max_dis_idx = dis_list.index(max(dis_list))
    new = np.zeros(mask.shape)
    if max_dis_idx == 0:
        new[:y_max - y_min, :x_max - x_min] = mask[y_min:y_max, x_min:x_max]
    elif max_dis_idx == 1:
        new[:y_max - y_min, w - (x_max - x_min):] = mask[y_min:y_max, x_min:x_max]
    elif max_dis_idx == 2:
        new[h - (y_max - y_min):, :x_max - x_min] = mask[y_min:y_max, x_min:x_max]
    else:
        new[h - (y_max - y_min):, w - (x_max - x_min):] = mask[y_min:y_max, x_min:x_max]
    return new


def train_detector(model_save_path, log_file_path, log_file, detector_model, input_shape, dataset, epoch, batch, lr,
                   seed, target=None,
                   attack_method=None, pretrained_model=None, generator_model=None):
    np.random.seed(seed)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file_path + log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    base_path, train_files, valid_files = get_data(dataset)

    if detector_model == "locatenet":
        detector = D(input_shape=input_shape, lr=lr)
    elif detector_model == "mantranet":
        detector = Mantranet(4, "./pretrained_weights/mantranet", lr=lr)
    elif detector_model == "span":
        detector = SPAN(lr=lr, input_size=input_shape[0])
    elif detector_model == "refinednet":
        detector = RefinedNet(lr)
    else:
        raise Exception("error detector!")

    if pretrained_model is not None:
        detector.load_weights(pretrained_model, skip_mismatch=True)

    if attack_method is not None and target is not None:
        sess = K.get_session()
        if attack_method == "fgsm":
            attack = FGSM(input_shape=input_shape, model=detector, eps=0.01, lower=-1, upper=1,
                          sess=sess, target=target)
        elif attack_method == "bim":
            attack = BIM(input_shape=input_shape, model=detector, eps=0.01, iter=10, lower=-1,
                         upper=1, sess=sess, target=target)
        elif attack_method == "mi-fgsm":
            attack = MI_FGSM(input_shape=input_shape, model=detector, eps=0.01, momentum=1.0, iter=10,
                             lower=-1, upper=1, sess=sess, target=target)
        elif attack_method == "gan":
            generator = G(input_shape=input_shape)
            generator.load_weights(generator_model)
            attack = GAN(generator, lower=-1, upper=1)
        elif attack_method == "gan_limit":
            generator = G(input_shape=input_shape)
            generator.load_weights(generator_model)
            attack = GAN_limit(generator, lower=-1, upper=1)
        else:
            raise Exception("error attack!")

    ori_train_x, ori_train_y = load_img_array(base_path, train_files[:], input_shape[0])
    ori_val_x, ori_val_y = load_img_array(base_path, valid_files[:], input_shape[0])
    max_val_f1 = 0
    flag = True
    for i in range(epoch):
        if attack_method is not None and target is not None:
            if i % 3 == 0:
                train_x = ori_train_x.copy()
                train_y = ori_train_y.copy()
                val_x = ori_val_x.copy()
                val_y = ori_val_y.copy()
                print("generate new adversarial samples")
                with tqdm(total=train_x.shape[0]) as pbar:
                    for idx in range(train_x.shape[0]):
                        pbar.update()
                        pbar.set_description("generate train adversarial samples")
                        train_x[idx] = attack.generate(np.expand_dims(train_x[idx], axis=0),
                                                       np.expand_dims(train_y[idx], axis=0))[0]
                with tqdm(total=val_x.shape[0]) as pbar:
                    for idx in range(val_x.shape[0]):
                        pbar.update()
                        pbar.set_description("generate val adversarial samples")
                        val_x[idx] = attack.generate(np.expand_dims(val_x[idx], axis=0),
                                                     np.expand_dims(val_y[idx], axis=0))[0]
                train_x = np.concatenate((train_x, ori_train_x), axis=0)
                train_y = np.concatenate((train_y, ori_train_y), axis=0)
                val_x = np.concatenate((val_x, ori_val_x), axis=0)
                val_y = np.concatenate((val_y, ori_val_y), axis=0)
            else:
                print("using old adversarial")
        else:
            train_x = ori_train_x
            train_y = ori_train_y
            val_x = ori_val_x
            val_y = ori_val_y
        # if flag and i == 50:
        #     K.set_value(detector.optimizer.lr, lr / 2)  # set new lr
        # elif flag and i == 150:
        #     K.set_value(detector.optimizer.lr, lr / 4)  # set new lr
        indices = np.arange(0, train_x.shape[0])
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]
        loss_avg = []
        train_f1_avg = []
        val_f1_avg = []
        with tqdm(total=train_x.shape[0]) as pbar:
            for b in range(0, train_x.shape[0], batch):
                pbar.update(batch)
                if b + batch >= train_x.shape[0]:
                    batch_x = train_x[b:]
                    batch_y = train_y[b:]
                else:
                    batch_x = train_x[b:b + batch]
                    batch_y = train_y[b:b + batch]
                    # train locate
                if 70 <= i < 140:
                    K.set_value(detector.optimizer.lr, lr / 2)
                elif i >= 140:
                    K.set_value(detector.optimizer.lr, lr / 4)
                train_loss, train_f1 = detector.train_on_batch(batch_x, batch_y)
                loss_avg.append(train_loss)
                train_f1_avg.append(train_f1)
                pbar.set_description(
                    "Train Epoch:%d f1:%.4f loss:%.4f" %
                    (i + 1,
                     sum(train_f1_avg) / len(train_f1_avg),
                     sum(loss_avg) / len(loss_avg)))
        with tqdm(total=val_x.shape[0]) as pbar:
            for b in range(0, val_x.shape[0], batch):
                pbar.update(batch)
                if b + batch >= val_x.shape[0]:
                    batch_x = val_x[b:]
                    batch_y = val_y[b:]
                else:
                    batch_x = val_x[b:b + batch]
                    batch_y = val_y[b:b + batch]
                pred_mask = detector.predict(batch_x)
                val_f1 = metrics.f1_score(batch_y.flatten().astype(np.int),
                                          np.round(pred_mask.flatten()).astype(np.int))
                val_f1_avg.append(val_f1)
                pbar.set_description(
                    "Val Epoch:%d f1:%.4f" %
                    (i + 1,
                     sum(val_f1_avg) / len(val_f1_avg)))
        if sum(val_f1_avg) / len(val_f1_avg) > max_val_f1:
            avg_f1 = sum(val_f1_avg) / len(val_f1_avg)
            max_val_f1 = avg_f1
            detector.save_weights(os.path.join(model_save_path, f"epoch{i + 1}_f1_{avg_f1}.h5"))
        logger.info(
            "Epoch:%d train_f1:%.4f train_loss:%.4f val_f1:%.4f" %
            (i + 1,
             sum(train_f1_avg) / len(train_f1_avg),
             sum(loss_avg) / len(loss_avg),
             sum(val_f1_avg) / len(val_f1_avg)))


def train_generator(model_save_path, check_save_path, log_file_path, log_file, detector_model, input_shape, dataset,
                    epoch, batch, lr, seed,
                    detector_weight=None, generator_block_type='mb', bce_w=10, hinge_w=1):
    np.random.seed(seed)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file_path + log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    base_path, train_files, valid_files = get_data(dataset)

    if detector_model == "locatenet":
        detector = D(input_shape=input_shape, lr=lr)
    elif detector_model == "mantranet":
        detector = Mantranet(4, "./pretrained_weights/mantranet", lr=lr)
    elif detector_model == "span":
        detector = SPAN(lr=lr, input_size=input_shape[0])
    else:
        raise Exception("error detector!")

    detector.load_weights(detector_weight)
    make_trainable(detector, False)
    generator = G(input_shape=input_shape, block_type=generator_block_type)
    gan = base.GAN(generator, detector, detector_model, input_shape=input_shape, lr=lr, bce_weight=bce_w,
                   hinge_weight=hinge_w)

    train_x, train_y = load_img_array(base_path, train_files[:], input_shape[0])
    val_x, val_y = load_img_array(base_path, valid_files[:], input_shape[0])
    # 训练中途检查生成器效果
    check_img, _ = load_img_array(base_path, valid_files[:1], input_shape[0])

    min_val_f1 = 1
    min_val_auc = 1

    for i in range(epoch):
        indices = np.arange(0, train_x.shape[0])
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]
        with tqdm(total=train_x.shape[0]) as pbar:
            bce_loss_avg = []
            train_f1_avg = []
            gan_loss_avg = []
            hinge_loss_avg = []
            for b in range(0, train_x.shape[0], batch):
                pbar.update(batch)
                if b + batch >= train_x.shape[0]:
                    batch_x = train_x[b:]
                    batch_y = train_y[b:]
                else:
                    batch_x = train_x[b:b + batch]
                    batch_y = train_y[b:b + batch]
                # train G
                res = gan.train_on_batch(batch_x, [batch_y, np.zeros(batch_x.shape)])
                gan_loss_avg.append(res[0])

                bce_loss_avg.append(res[1])
                hinge_loss_avg.append(res[2])
                train_f1_avg.append(res[3])
                pbar.set_description(
                    "Epoch:%d train_f1:%.4f bce_loss:%.4f loss:%.4f hinge_loss:%.4f" %
                    (i + 1,
                     sum(train_f1_avg) / len(train_f1_avg),
                     sum(bce_loss_avg) / len(bce_loss_avg),
                     sum(gan_loss_avg) / len(gan_loss_avg),
                     sum(hinge_loss_avg) / len(hinge_loss_avg)))
            train_check_img(check_img, generator, i + 1, check_save_path)
            logger.info(
                "Train Epoch:%d f1:%.4f bce_loss:%.4f loss:%.4f hinge_loss:%.4f" %
                (i + 1,
                 sum(train_f1_avg) / len(train_f1_avg),
                 sum(bce_loss_avg) / len(bce_loss_avg),
                 sum(gan_loss_avg) / len(gan_loss_avg),
                 sum(hinge_loss_avg) / len(hinge_loss_avg)))
        with tqdm(total=val_x.shape[0]) as pbar:
            val_f1_avg = []
            val_auc_avg = []
            for b in range(0, val_x.shape[0], batch):
                pbar.update(batch)
                if b + batch >= val_x.shape[0]:
                    batch_x = val_x[b:]
                    batch_y = val_y[b:]
                else:
                    batch_x = val_x[b:b + batch]
                    batch_y = val_y[b:b + batch]
                pred_mask, adv_x = gan.predict(batch_x)
                val_f1 = metrics.f1_score(batch_y.flatten().astype(np.int),
                                          np.round(pred_mask.flatten()).astype(np.int))
                val_f1_avg.append(val_f1)
                pbar.set_description("Epoch:%d val_f1:%.4f" % (i + 1, sum(val_f1_avg) / len(val_f1_avg)))
            logger.info(
                "Valid Epoch:%d f1:%.4f" %
                (i + 1,
                 sum(val_f1_avg) / len(val_f1_avg)))
            if sum(val_f1_avg) / len(val_f1_avg) < min_val_f1:
                avg_f1 = sum(val_f1_avg) / len(val_f1_avg)
                min_val_f1 = avg_f1
                generator.save(model_save_path + f"epoch{i + 1}_val_f1_{avg_f1}.h5")
            #     val_auc = metrics.roc_auc_score(batch_y.flatten().astype(np.int),
            #                                     pred_mask.flatten())
            #     val_auc_avg.append(val_auc)
            #     pbar.set_description("Epoch:%d val_auc:%.4f" % (i + 1, sum(val_auc_avg) / len(val_auc_avg)))
            # logger.info(
            #     "Valid Epoch:%d val_auc:%.4f" %
            #     (i + 1,
            #      sum(val_auc_avg) / len(val_auc_avg)))
            # if sum(val_auc_avg) / len(val_auc_avg) < min_val_auc:
            #     avg_auc = sum(val_auc_avg) / len(val_auc_avg)
            #     min_val_auc = avg_auc
            #     generator.save(model_save_path + f"epoch{i + 1}_val_auc_{avg_auc}.h5")


def train_generator_v2(model_save_path, check_save_path, log_file_path, log_file, detector_model, input_shape, dataset,
                       epoch, batch, lr, seed, detector_weight=None,
                       generator_block_type='mb', bce_w=10, hinge_w=1):
    np.random.seed(seed)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file_path + log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    base_path, train_files, valid_files = get_data(dataset)

    if detector_model == "locatenet":
        detector = D(input_shape=input_shape, lr=lr)
    elif detector_model == "mantranet":
        detector = Mantranet(4, "./pretrained_weights/mantranet", lr=lr)
    elif detector_model == "span":
        detector = SPAN(lr=lr, input_size=input_shape[0])
    else:
        raise Exception("error detector!")

    detector.load_weights(detector_weight)
    make_trainable(detector, False)
    generator = G(input_shape=input_shape, block_type=generator_block_type)
    gan = base.GAN_V2(generator, detector, detector_model, input_shape=input_shape, lr=lr, bce_weight=bce_w,
                      hinge_weight=hinge_w)

    train_x, train_y = load_img_array(base_path, train_files[:], input_shape[0])
    val_x, val_y = load_img_array(base_path, valid_files[:], input_shape[0])
    # 训练中途检查生成器效果
    check_img, check_mask = load_img_array(base_path, valid_files[:1], input_shape[0])
    check_mask = np.repeat(check_mask, 3, axis=-1)

    min_val_f1 = 1
    min_val_auc = 1
    flag = True
    val_batch = 1
    # 计算valid loss
    sess = K.get_session()
    y_true_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
    y_true = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)), dtype=tf.float32)
    y_true_op = tf.assign(y_true, y_true_ph)
    y_pred_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
    y_pred = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)), dtype=tf.float32)
    y_pred_op = tf.assign(y_pred, y_pred_ph)
    pert_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 3))
    pert = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 3)), dtype=tf.float32)
    pert_op = tf.assign(pert, pert_ph)
    zero_mask_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 3))
    zero_mask = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 3)), dtype=tf.float32)
    zero_mask_op = tf.assign(zero_mask, zero_mask_ph)
    bce_op = binary_loss(y_true, y_pred)
    hinge_op = hinge_loss(zero_mask, pert)
    for i in range(epoch):
        # if flag and i == 50:
        #     K.set_value(gan.optimizer.lr, lr / 2)  # set new lr
        #     K.set_value(generator.optimizer.lr, lr / 2)  # set new lr
        #     flag = False
        indices = np.arange(0, train_x.shape[0])
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]
        with tqdm(total=train_x.shape[0]) as pbar:
            bce_loss_avg = []
            train_f1_avg = []
            gan_loss_avg = []
            hinge_loss_avg = []
            for b in range(0, train_x.shape[0], batch):
                pbar.update(batch)
                if b + batch >= train_x.shape[0]:
                    batch_x = train_x[b:]
                    batch_y = train_y[b:]
                else:
                    batch_x = train_x[b:b + batch]
                    batch_y = train_y[b:b + batch]
                # dilated
                batch_y_d = batch_y
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                # for j, y_i in enumerate(batch_y_d):
                #     batch_y_d[j] = cv2.dilate(y_i, kernel).reshape(512, 512, 1)
                # train G
                res = gan.train_on_batch([batch_x, np.repeat(batch_y_d, 3, axis=-1)],
                                         [batch_y, np.zeros(batch_x.shape), batch_x * batch_y])
                gan_loss_avg.append(res[0])
                bce_loss_avg.append(res[1])
                hinge_loss_avg.append(res[2])
                train_f1_avg.append(res[4])
                pbar.set_description(
                    "Train Epoch:%d train_f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                    (i + 1,
                     sum(train_f1_avg) / len(train_f1_avg),
                     sum(bce_loss_avg) / len(bce_loss_avg),
                     sum(hinge_loss_avg) / len(hinge_loss_avg),
                     sum(gan_loss_avg) / len(gan_loss_avg)))
            train_check_img_v2(check_img, check_mask, generator, i + 1, check_save_path)
            logger.info(
                "Train Epoch:%d f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                (i + 1,
                 sum(train_f1_avg) / len(train_f1_avg),
                 sum(bce_loss_avg) / len(bce_loss_avg),
                 sum(hinge_loss_avg) / len(hinge_loss_avg),
                 sum(gan_loss_avg) / len(gan_loss_avg)))
        with tqdm(total=val_x.shape[0]) as pbar:
            val_f1_avg = []
            val_auc_avg = []
            val_bce_loss_avg = []
            val_gan_loss_avg = []
            val_hinge_loss_avg = []
            for b in range(0, val_x.shape[0], val_batch):
                pbar.update(val_batch)
                if b + val_batch >= val_x.shape[0]:
                    batch_x = val_x[b:]
                    batch_y = val_y[b:]
                else:
                    batch_x = val_x[b:b + val_batch]
                    batch_y = val_y[b:b + val_batch]
                # dilated
                batch_y_d = batch_y
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                # for j, y_i in enumerate(batch_y_d):
                #     batch_y_d[j] = cv2.dilate(y_i, kernel).reshape(512, 512, 1)
                adv_pert = generator.predict(batch_x)
                adv = np.clip(adv_pert + batch_x, -1, 1)
                pred_mask = detector.predict(adv)
                sess.run(y_true_op, feed_dict={y_true_ph: batch_y})
                sess.run(y_pred_op, feed_dict={y_pred_ph: pred_mask})
                sess.run(pert_op, feed_dict={pert_ph: adv_pert})
                sess.run(zero_mask_op, feed_dict={zero_mask_ph: np.zeros(batch_x.shape)})
                val_bce_loss = np.mean(sess.run(bce_op))
                val_hinge_loss = np.mean(sess.run(hinge_op))
                val_f1 = metrics.f1_score(np.round(batch_y.flatten()).astype(np.int),
                                          np.round(pred_mask.flatten()).astype(np.int))
                val_f1_avg.append(val_f1)
                val_hinge_loss_avg.append(val_hinge_loss)
                val_bce_loss_avg.append(val_bce_loss)
                val_gan_loss_avg.append(bce_w * val_bce_loss + hinge_w * val_hinge_loss)
                pbar.set_description(
                    "Valid Epoch:%d valid_f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                    (i + 1,
                     sum(val_f1_avg) / len(val_f1_avg),
                     sum(val_bce_loss_avg) / len(val_bce_loss_avg),
                     sum(val_hinge_loss_avg) / len(val_hinge_loss_avg),
                     sum(val_gan_loss_avg) / len(val_gan_loss_avg)))
            logger.info(
                "Valid Epoch:%d f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                (i + 1,
                 sum(val_f1_avg) / len(val_f1_avg),
                 sum(val_bce_loss_avg) / len(val_bce_loss_avg),
                 sum(val_hinge_loss_avg) / len(val_hinge_loss_avg),
                 sum(val_gan_loss_avg) / len(val_gan_loss_avg)))
            print((sum(val_f1_avg) / len(val_f1_avg)) < min_val_f1)
            print((sum(val_hinge_loss_avg) / len(val_hinge_loss_avg)) < 30)
            if (sum(val_f1_avg) / len(val_f1_avg)) < min_val_f1 and (
                    sum(val_hinge_loss_avg) / len(val_hinge_loss_avg)) < 30:
                avg_f1 = sum(val_f1_avg) / len(val_f1_avg)
                min_val_f1 = avg_f1
                generator.save(model_save_path + f"epoch{i + 1}_val_f1_{avg_f1}.h5")

def blackbox_attack(model_save_path, check_save_path, log_file_path, log_file, target_model_name, input_shape, dataset,
                    epoch, batch, lr, seed, target_weight, gan_gpu_id='0', tg_gpu_id='1',
                    generator_block_type='mb', bce_w=10, hinge_w=1):
    np.random.seed(seed)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file_path + log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    base_path, train_files, valid_files = get_data(dataset)
    # proxy_model = D(input_shape=input_shape, lr=lr)
    # proxy_model.load_weights(proxy_weight)
    if target_model_name == "mantranet":
        target_model = Mantranet(4, "./pretrained_weights/mantranet", lr=lr)
        proxy_model = Mantranet(4, "./pretrained_weights/mantranet", lr=lr)
    elif target_model_name == "span":
        target_model = SPAN(lr=lr, input_size=input_shape[0])
        proxy_model = SPAN(lr=lr, input_size=input_shape[0])
    elif target_model_name == "locatenet":
        target_model = locatenet(lr=lr)
        proxy_model = locatenet(lr=lr)
    else:
        raise Exception("error target model!")
    target_model.load_weights(target_weight)
    make_trainable(proxy_model, False)
    make_trainable(target_model, False)
    generator = G(input_shape=input_shape, block_type=generator_block_type)
    gan = base.GAN_V2(generator, proxy_model, target_model_name, input_shape=input_shape, lr=lr, bce_weight=bce_w,
                      hinge_weight=hinge_w)
    distiller = base.dynamic_distillation(proxy_model, input_shape=input_shape, lr=lr)
    train_x, train_y = load_img_array(base_path, train_files[:], input_shape[0])
    val_x, val_y = load_img_array(base_path, valid_files[:], input_shape[0])
    # 训练中途检查生成器效果
    check_img, check_mask = load_img_array(base_path, valid_files[:1], input_shape[0])
    check_mask = np.repeat(check_mask, 3, axis=-1)

    min_val_f1 = 1
    min_val_auc = 1
    flag = True
    # 计算valid loss
    val_batch = 1
    sess = K.get_session()
    y_true_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
    y_true = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)), dtype=tf.float32)
    y_true_op = tf.assign(y_true, y_true_ph)
    y_pred_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
    y_pred = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)), dtype=tf.float32)
    y_pred_op = tf.assign(y_pred, y_pred_ph)
    pert_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 3))
    pert = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 3)), dtype=tf.float32)
    pert_op = tf.assign(pert, pert_ph)
    zero_mask_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 3))
    zero_mask = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 3)), dtype=tf.float32)
    zero_mask_op = tf.assign(zero_mask, zero_mask_ph)
    bce_op = binary_loss(y_true, y_pred)
    hinge_op = hinge_loss(zero_mask, pert)
    for i in range(epoch):
        # if flag and i == 50:
        #     K.set_value(gan.optimizer.lr, lr / 2)  # set new lr
        #     K.set_value(generator.optimizer.lr, lr / 2)  # set new lr
        #     flag = False
        indices = np.arange(0, train_x.shape[0])
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]
        with tqdm(total=train_x.shape[0]) as pbar:
            bce_loss_avg = []
            train_f1_avg = []
            gan_loss_avg = []
            hinge_loss_avg = []
            ori_distilling_loss_avg = []
            adv_distilling_loss_avg = []
            for b in range(0, train_x.shape[0], batch):
                pbar.update(batch)
                if b + batch >= train_x.shape[0]:
                    batch_x = train_x[b:]
                    batch_y = train_y[b:]
                else:
                    batch_x = train_x[b:b + batch]
                    batch_y = train_y[b:b + batch]
                # dilated
                batch_y_d = batch_y
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                # for j, y_i in enumerate(batch_y_d):
                #     batch_y_d[j] = cv2.dilate(y_i, kernel).reshape(512, 512, 1)
                # train G frozen proxy
                make_trainable(generator, True)
                make_trainable(proxy_model, False)
                with tf.device('/gpu:0'):
                    res = gan.train_on_batch([batch_x, np.repeat(batch_y_d, 3, axis=-1)],
                                             [batch_y, np.zeros(batch_y.shape), batch_x * batch_y])
                gan_loss_avg.append(res[0])
                bce_loss_avg.append(res[1])
                hinge_loss_avg.append(res[2])
                train_f1_avg.append(res[4])
                # pred_mask = proxy_model.predict(adv)
                # print(np.sum(pred_mask))
                # train proxy frozen generator
                make_trainable(generator, False)
                make_trainable(proxy_model, True)
                with tf.device('/gpu:0'):
                    adv_pert = generator.predict(batch_x)
                    adv = np.clip(adv_pert + batch_x, -1, 1)
                with tf.device('/gpu:1'):
                    target_ori_pred = target_model.predict(batch_x)
                    # target_adv_pred = target_model.predict(adv)
                with tf.device('/gpu:0'):
                    # res = distiller.train_on_batch(x=[batch_x, adv],
                    #                                y=[target_ori_pred, target_adv_pred])
                    # res = distiller.train_on_batch(x=[adv, batch_x],
                    #                                y=[target_adv_pred, target_ori_pred])

                    res = proxy_model.train_on_batch([batch_x], [target_ori_pred[0], target_ori_pred[1]])
                ori_distilling_loss_avg.append(res[0])
                adv_distilling_loss_avg.append(res[0])
                pbar.set_description(
                    "Train Epoch:%d train_f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f ori_distilling_loss:%.4f adv_distilling_loss:%.4f" %
                    (i + 1,
                     sum(train_f1_avg) / len(train_f1_avg),
                     sum(bce_loss_avg) / len(bce_loss_avg),
                     sum(hinge_loss_avg) / len(hinge_loss_avg),
                     sum(gan_loss_avg) / len(gan_loss_avg),
                     sum(ori_distilling_loss_avg) / len(ori_distilling_loss_avg) if len(
                         ori_distilling_loss_avg) > 0 else 0,
                     sum(adv_distilling_loss_avg) / len(adv_distilling_loss_avg) if len(
                         adv_distilling_loss_avg) > 0 else 0))
            train_check_img_v2(check_img, check_mask, generator, i + 1, check_save_path)
            logger.info(
                "Train Epoch:%d f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f ori_distilling_loss:%.4f adv_distilling_loss:%.4f" %
                (i + 1,
                 sum(train_f1_avg) / len(train_f1_avg),
                 sum(bce_loss_avg) / len(bce_loss_avg),
                 sum(hinge_loss_avg) / len(hinge_loss_avg),
                 sum(gan_loss_avg) / len(gan_loss_avg),
                 sum(ori_distilling_loss_avg) / len(ori_distilling_loss_avg) if len(ori_distilling_loss_avg) > 0 else 0,
                 sum(adv_distilling_loss_avg) / len(adv_distilling_loss_avg) if len(
                     adv_distilling_loss_avg) > 0 else 0))
        with tqdm(total=val_x.shape[0]) as pbar:
            val_f1_avg = []
            val_auc_avg = []
            val_bce_loss_avg = []
            val_gan_loss_avg = []
            val_hinge_loss_avg = []
            make_trainable(generator, False)
            make_trainable(proxy_model, False)
            for b in range(0, val_x.shape[0], val_batch):
                pbar.update(val_batch)
                if b + val_batch >= val_x.shape[0]:
                    batch_x = val_x[b:]
                    batch_y = val_y[b:]
                else:
                    batch_x = val_x[b:b + val_batch]
                    batch_y = val_y[b:b + val_batch]
                # dilated
                batch_y_d = batch_y
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                # for j, y_i in enumerate(batch_y_d):
                #     batch_y_d[j] = cv2.dilate(y_i, kernel).reshape(512, 512, 1)
                with tf.device('/gpu:0'):
                    adv_pert = generator.predict(batch_x)
                    adv = np.clip(adv_pert + batch_x, -1, 1)
                    pred_mask = target_model.predict(adv)[0]
                    sess.run(y_true_op, feed_dict={y_true_ph: batch_y})
                    sess.run(y_pred_op, feed_dict={y_pred_ph: pred_mask})
                    sess.run(pert_op, feed_dict={pert_ph: adv_pert})
                    sess.run(zero_mask_op, feed_dict={zero_mask_ph: np.zeros(batch_x.shape)})
                    val_bce_loss = np.mean(sess.run(bce_op))
                    val_hinge_loss = np.mean(sess.run(hinge_op))
                val_f1 = metrics.f1_score(np.round(batch_y.flatten()).astype(np.int),
                                          np.round(pred_mask.flatten()).astype(np.int))
                val_f1_avg.append(val_f1)
                val_hinge_loss_avg.append(val_hinge_loss)
                val_bce_loss_avg.append(val_bce_loss)
                val_gan_loss_avg.append(bce_w * val_bce_loss + hinge_w * val_hinge_loss)
                pbar.set_description(
                    "Valid Epoch:%d valid_f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                    (i + 1,
                     sum(val_f1_avg) / len(val_f1_avg),
                     sum(val_bce_loss_avg) / len(val_bce_loss_avg),
                     sum(val_hinge_loss_avg) / len(val_hinge_loss_avg),
                     sum(val_gan_loss_avg) / len(val_gan_loss_avg)))
            logger.info(
                "Valid Epoch:%d f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                (i + 1,
                 sum(val_f1_avg) / len(val_f1_avg),
                 sum(val_bce_loss_avg) / len(val_bce_loss_avg),
                 sum(val_hinge_loss_avg) / len(val_hinge_loss_avg),
                 sum(val_gan_loss_avg) / len(val_gan_loss_avg)))
            if sum(val_f1_avg) / len(val_f1_avg) < min_val_f1 and i > 2:
                avg_f1 = sum(val_f1_avg) / len(val_f1_avg)
                min_val_f1 = avg_f1
                generator.save(model_save_path + f"epoch{i + 1}_val_f1_{avg_f1}.h5")
            #     val_auc = metrics.roc_auc_score(batch_y.flatten().astype(np.int),
            #                                     pred_mask.flatten())
            #     val_auc_avg.append(val_auc)
            #     pbar.set_description("Epoch:%d val_auc:%.4f" % (i + 1, sum(val_auc_avg) / len(val_auc_avg)))
            # logger.info(
            #     "Valid Epoch:%d val_auc:%.4f" %
            #     (i + 1,
            #      sum(val_auc_avg) / len(val_auc_avg)))
            # if sum(val_auc_avg) / len(val_auc_avg) < min_val_auc:
            #     avg_auc = sum(val_auc_avg) / len(val_auc_avg)
            #     min_val_auc = avg_auc
            #     generator.save(model_save_path + f"epoch{i + 1}_val_auc_{avg_auc}.h5")


def advgan_train(model_save_path, check_save_path, log_file_path, log_file, detector_model, input_shape, dataset,
                 epoch, batch, lr, seed, detector_weight=None, alpha=10, beta=1):
    np.random.seed(seed)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_file_path + log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    base_path, train_files, valid_files = get_data(dataset)

    if detector_model == "locatenet":
        detector = D(input_shape=input_shape, lr=lr)
    elif detector_model == "mantranet":
        detector = Mantranet(4, "./pretrained_weights/mantranet", lr=lr)
    elif detector_model == "span":
        detector = SPAN(lr=lr, input_size=input_shape[0])
    else:
        raise Exception("error detector!")

    detector.load_weights(detector_weight)
    make_trainable(detector, False)
    generator = Generator(input_shape=input_shape)
    discriminator = Discriminator(input_shape=input_shape)
    gan = advgan(generator, discriminator, detector, detector_model, base_path, input_shape, lr, alpha, beta)
    train_x, train_y = load_img_array(base_path, train_files[:], input_shape[0])
    val_x, val_y = load_img_array(base_path, valid_files[:], input_shape[0])
    # 训练中途检查生成器效果
    check_img, check_mask = load_img_array(base_path, valid_files[:1], input_shape[0])
    check_mask = np.repeat(check_mask, 3, axis=-1)

    min_val_f1 = 1
    val_batch = 1
    # 计算valid loss
    sess = K.get_session()
    y_true_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
    y_true = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)), dtype=tf.float32)
    y_true_op = tf.assign(y_true, y_true_ph)
    y_pred_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 1))
    y_pred = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 1)), dtype=tf.float32)
    y_pred_op = tf.assign(y_pred, y_pred_ph)
    pert_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 3))
    pert = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 3)), dtype=tf.float32)
    pert_op = tf.assign(pert, pert_ph)
    zero_mask_ph = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 3))
    zero_mask = tf.Variable(initial_value=np.zeros((1, input_shape[0], input_shape[1], 3)), dtype=tf.float32)
    zero_mask_op = tf.assign(zero_mask, zero_mask_ph)
    bce_op = binary_loss(y_true, y_pred)
    hinge_op = hinge_loss(zero_mask, pert)
    for i in range(epoch):
        # if flag and i == 50:
        #     K.set_value(gan.optimizer.lr, lr / 2)  # set new lr
        #     K.set_value(generator.optimizer.lr, lr / 2)  # set new lr
        #     flag = False
        indices = np.arange(0, train_x.shape[0])
        np.random.shuffle(indices)
        train_x = train_x[indices]
        train_y = train_y[indices]
        with tqdm(total=train_x.shape[0]) as pbar:
            bce_loss_avg = []
            train_f1_avg = []
            gan_loss_avg = []
            hinge_loss_avg = []
            disc_ce_loss_avg = []
            for b in range(0, train_x.shape[0], batch):
                pbar.update(batch)
                if b + batch >= train_x.shape[0]:
                    batch_x = train_x[b:]
                    batch_y = train_y[b:]
                else:
                    batch_x = train_x[b:b + batch]
                    batch_y = train_y[b:b + batch]
                # dilated
                # batch_y_d = batch_y
                # train discriminator
                make_trainable(generator, False)
                make_trainable(discriminator, True)
                res = discriminator.train_on_batch(batch_x, np.ones(batch_x.shape[0]))
                adv_pert = generator.predict(batch_x)
                adv = np.clip(adv_pert + batch_x, -1, 1)
                res = discriminator.train_on_batch(adv, np.zeros(batch_x.shape[0]))
                # train generator
                make_trainable(generator, True)
                make_trainable(discriminator, False)
                res = gan.train_on_batch(x=[batch_x],
                                         y=[np.zeros(batch_x.shape), batch_y, np.zeros(batch_x.shape[0])])
                gan_loss_avg.append(res[0])
                hinge_loss_avg.append(res[1])
                bce_loss_avg.append(res[2])
                disc_ce_loss_avg.append(res[3])
                train_f1_avg.append(res[4])
                pbar.set_description(
                    "Train Epoch:%d train_f1:%.4f bce_loss:%.4f hinge_loss:%.4f disc_ce_loss:%.4f loss:%.4f" %
                    (i + 1,
                     sum(train_f1_avg) / len(train_f1_avg),
                     sum(bce_loss_avg) / len(bce_loss_avg),
                     sum(hinge_loss_avg) / len(hinge_loss_avg),
                     sum(disc_ce_loss_avg) / len(disc_ce_loss_avg),
                     sum(gan_loss_avg) / len(gan_loss_avg)))
            train_check_img_v2(check_img, check_mask, generator, i + 1, check_save_path)
            logger.info(
                "Train Epoch:%d f1:%.4f bce_loss:%.4f hinge_loss:%.4f disc_ce_loss:%.4f loss:%.4f" %
                (i + 1,
                 sum(train_f1_avg) / len(train_f1_avg),
                 sum(bce_loss_avg) / len(bce_loss_avg),
                 sum(hinge_loss_avg) / len(hinge_loss_avg),
                 sum(disc_ce_loss_avg) / len(disc_ce_loss_avg),
                 sum(gan_loss_avg) / len(gan_loss_avg)))

        with tqdm(total=val_x.shape[0]) as pbar:
            val_f1_avg = []
            val_bce_loss_avg = []
            val_gan_loss_avg = []
            val_hinge_loss_avg = []
            make_trainable(generator, False)
            make_trainable(discriminator, False)
            for b in range(0, val_x.shape[0], val_batch):
                pbar.update(val_batch)
                if b + val_batch >= val_x.shape[0]:
                    batch_x = val_x[b:]
                    batch_y = val_y[b:]
                else:
                    batch_x = val_x[b:b + val_batch]
                    batch_y = val_y[b:b + val_batch]
                adv_pert = generator.predict(batch_x)
                adv = np.clip(adv_pert + batch_x, -1, 1)
                pred_mask = detector.predict(adv)
                sess.run(y_true_op, feed_dict={y_true_ph: batch_y})
                sess.run(y_pred_op, feed_dict={y_pred_ph: pred_mask})
                sess.run(pert_op, feed_dict={pert_ph: adv_pert})
                sess.run(zero_mask_op, feed_dict={zero_mask_ph: np.zeros(batch_x.shape)})
                val_bce_loss = np.mean(sess.run(bce_op))
                val_hinge_loss = np.mean(sess.run(hinge_op))
                val_f1 = metrics.f1_score(np.round(batch_y.flatten()).astype(np.int),
                                          np.round(pred_mask.flatten()).astype(np.int))
                val_f1_avg.append(val_f1)
                val_hinge_loss_avg.append(val_hinge_loss)
                val_bce_loss_avg.append(val_bce_loss)
                val_gan_loss_avg.append(alpha * val_bce_loss + beta * val_hinge_loss)
                pbar.set_description(
                    "Valid Epoch:%d valid_f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                    (i + 1,
                     sum(val_f1_avg) / len(val_f1_avg),
                     sum(val_bce_loss_avg) / len(val_bce_loss_avg),
                     sum(val_hinge_loss_avg) / len(val_hinge_loss_avg),
                     sum(val_gan_loss_avg) / len(val_gan_loss_avg)))
            logger.info(
                "Valid Epoch:%d f1:%.4f bce_loss:%.4f hinge_loss:%.4f loss:%.4f" %
                (i + 1,
                 sum(val_f1_avg) / len(val_f1_avg),
                 sum(val_bce_loss_avg) / len(val_bce_loss_avg),
                 sum(val_hinge_loss_avg) / len(val_hinge_loss_avg),
                 sum(val_gan_loss_avg) / len(val_gan_loss_avg)))
            print((sum(val_f1_avg) / len(val_f1_avg)) < min_val_f1)
            print((sum(val_hinge_loss_avg) / len(val_hinge_loss_avg)) < 30)
            if (sum(val_f1_avg) / len(val_f1_avg)) < min_val_f1 and (
                    sum(val_hinge_loss_avg) / len(val_hinge_loss_avg)) < 30:
                avg_f1 = sum(val_f1_avg) / len(val_f1_avg)
                min_val_f1 = avg_f1
                generator.save(model_save_path + f"epoch{i + 1}_val_f1_{avg_f1}.h5")

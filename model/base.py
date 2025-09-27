import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
# from keras.applications.resnext import ResNeXt101
from keras.layers import Conv2D, Lambda, Input, ELU
from keras.layers import UpSampling2D, Concatenate, \
    DepthwiseConv2D, Add, ReLU, Multiply, Activation, Flatten, Dense, LeakyReLU, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
from layer.block import vgg_block, mb_conv_block, CAM, CBAM
from utils.loss import hinge_loss, psnr_ssim, binary_loss, pretext_loss, binary_crossentropy
from utils.metrics import f1

gpus = 1

# def encode(img):
#     # label = np.round(cv2.cvtColor((img).astype('float32'),cv2.COLOR_RGB2GRAY))
#     label = np.round(np.average(img, axis=-1))
#     label[label == 0] = 2.
#     label[label == 1] = 0
#     label[label == 2.] = 1
#     '''
#         if np.max(img)>10:
#             img /= 255
#         label = np.zeros((256,256))
#         for i in range(img.shape[0]):
#             for j in range(img.shape[1]):
#                 try:
#                     if list(img[i,j]) != [1,1,1]:
#                         label[i,j] = 1.
#                 except: None
#         return label
#     '''
#     return label
#
#
# def encode_files(imgs):
#     labels = []
#     for i in imgs:
#         labels.append(encode(i))
#     # print labels[0]
#     return np.array(labels)


def srm_init(shape, dtype=None):
    hpf = np.zeros(shape, dtype=np.float32)

    hpf[:, :, 0, 0] = np.array(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.0
    hpf[:, :, 0, 1] = np.array(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12.
    hpf[:, :, 0, 2] = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.0

    return hpf


# def deconv_stack(data, filters, s=1, short_cut=True):
#     filters = filters * 2
#     output = Conv2D(filters, (3, 3), strides=s, padding='same')(data)
#     output = ELU()(output)
#     output = Conv2D(filters, (3, 3), strides=1, padding='same')(output)
#     output = ELU()(output)
#     if short_cut:
#         data = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
#         data = ELU()(data)
#     output = keras.layers.concatenate([output, data], axis=-1)
#     output = BatchNormalization()(output)
#     output = ELU()(output)
#     return output


# def isp(x, filters, s=1):
#     x = deconv_stack(x, filters, short_cut=True)
#     out = Lambda(lambda x: tf.depth_to_space(x, 2))(x)
#     return out


def det_deconv(x, filters):
    x = UpSampling2D()(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x


def D(input_shape, lr=0.0002):
    def conv(x, filters, kernel=3, strides=1, dilation=1):
        x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, dilation_rate=dilation, padding='same',
                   activation='relu')(x)
        # x = ELU()(x)
        return x

    rgb = Input(shape=input_shape)
    # x = Conv2D(filters=3,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same',input_shape=(256,256,1))(rgb)
    coarse_in = DepthwiseConv2D(depth_multiplier=3, kernel_size=(5, 5), padding='same', depthwise_initializer=srm_init)(
        rgb)
    coarse_in.trainable = False
    x = coarse_in
    # x = Conv2D(filters=3,kernel_size=(5,5),strides=1,kernel_initializer=srm_init,padding='same',input_shape=(256,256,1))(rgb)
    # x = Conv2D(filters=32)

    x1 = vgg_block(x, filters=64, pooling=True)
    x2 = vgg_block(x1, filters=128, pooling=True)
    x3 = vgg_block(x2, filters=256, is_seven=True)
    # x3 = vgg_block(x3,filters=256,is_seven=True)
    x = conv(x3, 256, dilation=2)
    x = conv(x, 256, dilation=4)
    x = conv(x, 256, dilation=8)
    x = conv(x, 256, dilation=16)
    x3 = x
    x3 = Concatenate(name='concat1')([x2, x3])
    x4 = det_deconv(x3, 64)
    x5 = vgg_block(x4, filters=64)
    x5 = Concatenate(name='cocat2')([x1, x5])
    x6 = det_deconv(x5, 32)

    out1 = Conv2D(1, 7, activation='sigmoid', padding='same', name='out1')(x6)

    model = Model(inputs=[rgb], outputs=[out1])
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)

    # print(model.summary())
    # optimizer = SGD(lr=lr, clipvalue=0.2)
    optimizer = Adam(lr=lr)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=[f1])
    # print(model.metrics_names)
    return model


def G(input_shape, lr=0.0002, block_type="vgg", use_cam=False):
    forgery = Input(shape=input_shape)

    # out2 = resx_block(x12,filters=3,last=True)

    # out2 = Conv2D(3,7,activation='sigmoid',padding='same',name='out2')(x)
    def conv(data, filters, strides=1, kernel=3, short_cut=True):
        output = Conv2D(filters, kernel, strides=strides, padding='same')(data)
        output = ELU()(output)
        output = Conv2D(filters, kernel, strides=1, padding='same')(output)
        output = ELU()(output)
        if short_cut:
            data = Conv2D(filters, kernel, strides=strides, padding='same')(data)
            data = ELU()(data)
        output = keras.layers.add([output, data])
        output = BatchNormalization(axis=-1)(output)
        output = ELU()(output)
        return output

    def conv_d(x, filters, kernel=3, strides=1, dilation=1):
        x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, dilation_rate=dilation, padding='same')(x)
        x = BatchNormalization(axis=-1)(x)
        x = ReLU()(x)
        # x = ELU()(x)
        return x

    # def conv(x,filters,kernel=3,strides=1,dilation=1):

    # x = Conv2D(filters=filters,kernel_size=kernel,strides=strides,dilation_rate=dilation,padding='same')(x)
    # x = ELU()(x)
    # return x

    fine_in = forgery
    # res
    if block_type == "res":
        x = conv(fine_in, 32, kernel=5)
        x1 = conv(x, 64, strides=2)
        x2 = conv(x1, 64)
        x3 = conv(x2, 128, strides=2)
        x4 = conv(x3, 128)
        x5 = conv(x4, 256, strides=2)
        x6 = conv(x5, 256)
        x7 = conv(x6, 512)
    # vgg
    elif block_type == "vgg":
        x = conv(fine_in, 32, kernel=5)
        x1 = vgg_block(x, 64, pooling=True)
        x2 = vgg_block(x1, 64)
        x3 = vgg_block(x2, 128, pooling=True)
        x4 = vgg_block(x3, 128)
        x5 = vgg_block(x4, 256, pooling=True)
        x6 = vgg_block(x5, 256)
        x7 = vgg_block(x6, 512, is_seven=True)
    # mobile
    elif block_type == "mb":
        x = conv(fine_in, 32, kernel=5)
        x1 = mb_conv_block(x, 64, strides=(2, 2), block_id=1)
        x2 = mb_conv_block(x1, 64, block_id=2)
        x3 = mb_conv_block(x2, 128, strides=(2, 2), block_id=3)
        x4 = mb_conv_block(x3, 128, block_id=4)
        x5 = mb_conv_block(x4, 256, strides=(2, 2), block_id=5)
        x6 = mb_conv_block(x5, 256, block_id=6)
        x7 = mb_conv_block(x6, 512, block_id=7)
    # diverse branch
    # if block_type == "db":
    #     x = conv(fine_in, 32, kernel=5)
    #     x1 = diverse_branch_block(x, 64, stride=(2, 2))
    #     x2 = diverse_branch_block(x1, 64)
    #     x3 = diverse_branch_block(x2, 128, stride=(2, 2))
    #     x4 = diverse_branch_block(x3, 128)
    #     x5 = diverse_branch_block(x4, 256, stride=(2, 2))
    #     x6 = diverse_branch_block(x5, 256)
    #     x7 = diverse_branch_block(x6, 512)
    else:
        raise Exception("error block!")

    x7 = conv_d(x7, 256, dilation=2)
    x7 = conv_d(x7, 256, dilation=4)
    x7 = conv_d(x7, 256, dilation=8)
    x7 = conv_d(x7, 256, dilation=16)

    if use_cam:
        cam = CAM()(x7)
        cam = CBAM(x7, 256, block_id=8)
        cam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
        cam = BatchNormalization(axis=-1)(cam)
        cam = Activation('relu')(cam)
        # cam = Dropout(0.5)(cam)
        cam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
        x8 = det_deconv(cam, 128)
    else:
        x8 = det_deconv(x7, 128)
    x8 = Concatenate()([x4, x8])
    x9 = det_deconv(x8, 64)
    x9 = Concatenate()([x2, x9])
    x10 = det_deconv(x9, 32)
    x10 = Concatenate()([x, x10])
    fforgery = Conv2D(3, 7, padding='same', activation='tanh', name='fforgery')(x10)
    # 这里做线性变换
    # fforgery_limit = Lambda(lambda x: 0.1 * (x + 1) - 0.1)(fforgery)
    # fforgery_conc = Concatenate()([fforgery_limit, fforgery_limit, fforgery_limit])
    # out = Add()([forgery, fforgery_limit])
    # out = Add()([forgery, fforgery])
    # fforgery = Lambda(lambda x:K.clip(x,-1,1))(fforgery)#Activation('tanh')(fforgery)
    model = Model(inputs=forgery, outputs=fforgery)

    optimizer = Adam(lr=lr)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    # print(model.summary())
    model.compile(loss=hinge_loss,
                  optimizer=optimizer,
                  metrics=[psnr_ssim])
    # print(model.metrics_names)

    return model

def GAN(g, locate, detector_model, input_shape, lr=0.0002, bce_weight=10, hinge_weight=1):
    forgery = Input(shape=input_shape, name='gan_input')
    tf.get_variable_scope().reuse_variables()
    perturb = g(forgery)
    fforgery = Add()([perturb, forgery])
    fforgery = Lambda(lambda x: K.clip(x, -1., 1.))(fforgery)
    # fforgery = Lambda(lambda x:K.clip(x,-1,1),name='out')(fforgery)
    # locate.trainable = False
    # dis.trainable = False
    # concatenate
    locate_out = locate(fforgery)

    optimizer = Adam(lr=lr)
    # optimizer = Adam(lr=0.0002, clipvalue=0.5)
    model = Model(inputs=forgery, outputs=[locate_out, perturb])

    # print(model.summary())
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    # need to modify
    if detector_model == "locatenet":
        model.compile(loss={'model_1': binary_loss, "model_2": hinge_loss},
                      loss_weights=[bce_weight, hinge_weight],
                      optimizer=optimizer,
                      metrics={'model_1': f1, 'add_17': psnr_ssim})
    elif detector_model == "mantranet":
        model.compile(loss={'sigNet': binary_loss, "model_2": hinge_loss},
                      loss_weights=[bce_weight, hinge_weight],
                      optimizer=optimizer,
                      metrics={'sigNet': f1, 'add_17': psnr_ssim})
    elif detector_model == "span":
        model.compile(loss={'sigNet': binary_loss, "multiply_1": hinge_loss},
                      loss_weights=[bce_weight, hinge_weight],
                      optimizer=optimizer,
                      metrics={'sigNet': f1, 'add_17': psnr_ssim})
    # print(model.metrics_names)

    return model


def GAN_V2(g, locate, detector_model, input_shape, lr=0.0002, bce_weight=10, hinge_weight=1):
    forgery = Input(shape=input_shape, name='gan_input')
    tamper_mask = Input(shape=input_shape, name='tamper_mask_input')
    tf.get_variable_scope().reuse_variables()
    perturb = g(forgery)
    # 将扰动限制在篡改区域
    limited_perturb = Multiply()([perturb, tamper_mask])
    fforgery = Add()([perturb, forgery])
    fforgery = Lambda(lambda x: K.clip(x, -1., 1.))(fforgery)
    if detector_model != "refinednet":
        out2 = locate(fforgery)
    else:
        out1, out2 = locate(fforgery)

    optimizer = Adam(lr=lr)
    model = Model(inputs=[forgery, tamper_mask], outputs=[out2, perturb, limited_perturb])

    print(model.summary())
    # print(g.summary())
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    # need to modify
    if detector_model == "locatenet":
        model.compile(loss={'model_1': binary_loss, "model_2": hinge_loss, "multiply_1": pretext_loss},
                      loss_weights=[bce_weight, hinge_weight, 0.001],
                      optimizer=optimizer,
                      metrics={'model_1': f1, 'add_17': psnr_ssim})
    elif detector_model == "mantranet":
        model.compile(loss={'sigNet': binary_loss, "multiply_1": hinge_loss, "multiply_1": pretext_loss},
                      loss_weights=[bce_weight, hinge_weight, hinge_weight],
                      optimizer=optimizer,
                      metrics={'sigNet': f1, 'add_17': psnr_ssim})
    elif detector_model == "span":
        model.compile(loss={'sigNet': binary_loss, "multiply_1": hinge_loss, "multiply_1": pretext_loss},
                      loss_weights=[bce_weight, hinge_weight, hinge_weight],
                      optimizer=optimizer,
                      metrics={'sigNet': f1, 'add_17': psnr_ssim})
    print(model.metrics_names)

    return model


def dynamic_distillation(detector, input_shape, lr=0.0002):
    forgery = Input(shape=input_shape, name='origin_input')
    gan_forgery = Input(shape=input_shape, name='gan_output')
    # tf.get_variable_scope().reuse_variables()
    # 单独处理一下satfl
    _, d_ori_out = detector(forgery)
    # d_adv_out = detector(gan_forgery)
    optimizer = Adam(lr=lr)
    model = Model(inputs=[forgery], outputs=[d_ori_out])
    # model = Model(inputs=[forgery, gan_forgery], outputs=[d_ori_out, d_adv_out])
    # model = Model(inputs=[gan_forgery, forgery], outputs=[d_adv_out, d_ori_out])
    # print(model.summary())
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer)
    # print(model.metrics_names)
    return model


def Discriminator(input_shape):
    def conv(x, filters, kernel=5, strides=2, dilation=1):
        x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, dilation_rate=dilation, padding='same')(x)
        x = LeakyReLU()(x)
        return x

    dis_in = Input(shape=input_shape)
    x = conv(dis_in, 64)
    x1 = conv(x, 128)
    x2 = conv(x1, 256)
    x3 = conv(x2, 512)
    x4 = Flatten()(x3)
    x4 = Dense(1, activation="sigmoid")(x4)
    model = Model(dis_in, x4)
    optimizer = Adam(lr=0.0002, clipvalue=0.5)
    # print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return model


def Generator(input_shape):
    def conv(x, filters, kernel=3, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def transconv(x, filters, kernel=3, strides=1):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def res_block(x, filters=32, kernel=3, strides=1):
        conv1 = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)
        conv2 = Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding="same")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)
        x = Add()([x, conv2])
        return x

    forgery = Input(shape=input_shape)
    c1 = conv(forgery, 8, 3, 1)
    d1 = conv(c1, 16, 3, 2)
    d2 = conv(d1, 32, 3, 2)
    rb1 = res_block(d2, 32)
    rb2 = res_block(rb1, 32)
    rb3 = res_block(rb2, 32)
    rb4 = res_block(rb3, 32)

    u1 = transconv(rb4, 16, 3, 2)
    u2 = transconv(u1, 8, 3, 2)
    out = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding="same", activation="tanh")(u2)
    model = Model(forgery, out)
    optimizer = Adam(lr=0.0002, clipvalue=0.5)
    model.compile(loss=hinge_loss,
                  optimizer=optimizer)
    return model


def advgan(gm, dm, fm, model_name, data_path, input_shape, lr=0.0002, alpha=1, beta=10):
    forgery = Input(shape=input_shape, name='gan_input')
    tf.get_variable_scope().reuse_variables()
    perturb = gm(forgery)
    # 将扰动限制在篡改区域
    fforgery = Add()([perturb, forgery])
    if 'IMD2020' in data_path:
        fforgery = Lambda(lambda x: K.clip(x, 0, 1.))(fforgery)
    else:
        fforgery = Lambda(lambda x: K.clip(x, -1., 1.))(fforgery)
    out1 = fm(fforgery)
    out2 = dm(fforgery)

    optimizer = Adam(lr=lr)
    model = Model(inputs=[forgery], outputs=[perturb, out1, out2])

    print(model.summary())
    # print(g.summary())
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    # need to modify
    if model_name == "locatenet":
        model.compile(loss={'model_2': hinge_loss, "model_1": binary_loss, "model_3": binary_crossentropy},
                      loss_weights=[beta, alpha, 1],
                      optimizer=optimizer,
                      metrics={'model_1': f1})
    elif model_name == "mantranet":
        model.compile(loss={'model_1': hinge_loss, "sigNet": binary_loss, "model_2": binary_crossentropy},
                      loss_weights=[beta, alpha, 1],
                      optimizer=optimizer,
                      metrics={'sigNet': f1})
    elif model_name == "span":
        model.compile(loss={'model_1': hinge_loss, "sigNet": binary_loss, "model_2": binary_crossentropy},
                      loss_weights=[beta, alpha, 1],
                      optimizer=optimizer,
                      metrics={'sigNet': f1})
    print(model.metrics_names)

    return model


if __name__ == '__main__':
    # train code
    pass

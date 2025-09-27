import keras
import keras.layers as layers
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer
from keras.layers import Concatenate, Add
from keras.layers import Conv2D, Lambda, Reshape, Activation, Dense, GlobalAveragePooling2D, GlobalMaxPool2D, Multiply


class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma * bcTd + input
        return out


def CBAM(inputs, gate_channel, reduction_ratio=16, stride=1, block_id=1):
    x = inputs
    x = mb_conv_block(x, gate_channel, block_id=block_id)

    # Channel Attention mudule
    avgpool = GlobalAveragePooling2D()(x)  # channel avgpool
    maxpool = GlobalMaxPool2D()(x)  # channel maxpool
    # Shared MLP
    avg_Dense_layer1 = Dense(gate_channel // reduction_ratio, activation='relu')(avgpool)  # channel fc1
    avg_out = Dense(gate_channel, activation='relu')(avg_Dense_layer1)  # channel fc2
    max_Dense_layer1 = Dense(gate_channel // reduction_ratio, activation='relu')(maxpool)  # channel fc1
    max_out = Dense(gate_channel, activation='relu')(max_Dense_layer1)  # channel fc2

    channel = Add()([avg_out, max_out])
    channel = Activation('sigmoid')(channel)  # channel sigmoid
    channel = Reshape((1, 1, gate_channel))(channel)
    channel_out = Multiply()([x, channel])

    # Spatial Attention mudule
    avgpool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_out)
    maxpool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_out)
    spatial = Concatenate(axis=3)([avgpool, maxpool])
    # kernel filter 7x7 follow the paper
    spatial = Conv2D(1, (7, 7), strides=1, padding='same')(spatial)  # spatial conv2d
    spatial_out = Activation('sigmoid')(spatial)  # spatial sigmoid

    CBAM_out = Multiply()([channel_out, spatial_out])

    # residual connection

    return CBAM_out


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma * aaTa + input
        return out


def vgg_block(x, filters, pooling=False, is_seven=False, last=False, name='out1'):
    # 2*(3x3)= 5x5  3*(3*3) = 7x7
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    if is_seven:
        x = layers.Conv2D(filters, (3, 3),
                          padding='same')(x)
        if last:
            x = Activation('sigmoid', name=name)(x)
        else:
            x = Activation('relu')(x)
    if pooling:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x


def diverse_branch_block(x, filters, stride=(1, 1)):
    # for 3×3 Conv
    conv1_1x1 = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(x)
    conv1_1x1_bn = layers.BatchNormalization()(conv1_1x1)
    conv2_1x1 = layers.Conv2D(filters, (1, 1))(x)
    conv2_1x1_bn = layers.BatchNormalization()(conv2_1x1)
    conv2_1x1_kxk = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(conv2_1x1_bn)
    conv2_1x1_kxk_bn = layers.BatchNormalization()(conv2_1x1_kxk)
    conv3_1x1 = layers.Conv2D(filters, (1, 1))(x)
    conv3_1x1_bn = layers.BatchNormalization()(conv3_1x1)
    conv3_1x1_avg = layers.AvgPool2D(pool_size=(3, 3), strides=stride, padding='same')(conv3_1x1_bn)
    conv3_1x1_avg_bn = layers.BatchNormalization()(conv3_1x1_avg)
    conv4_kxk = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
    conv4_kxk_bn = layers.BatchNormalization()(conv4_kxk)
    add_all = layers.Add()([conv1_1x1_bn, conv2_1x1_kxk_bn, conv3_1x1_avg_bn, conv4_kxk_bn])
    out = layers.ReLU()(add_all)
    return out


def mb_conv_block(inputs, pointwise_conv_filters, alpha=1,
                  depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

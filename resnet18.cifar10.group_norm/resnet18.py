"""

ResNet18 model

"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.layers import conv2d, flatten
from tensorflow.contrib.layers import batch_norm, group_norm


DATA_FORMAT = 'channels_first'
PADDING = 'SAME'
BN_EPS = 1e-5


def group_normalization(x, name, group, eps=1e-5):

    with tf.variable_scope(name):

        N, C, H, W = x.get_shape().as_list()
        x = tf.reshape(x, [-1, group, C // group, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, [-1, C, H, W])

        gamma = tf.get_variable('gamma', [1, C, 1, 1], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, C, 1, 1], initializer=tf.constant_initializer(0.0))

    return x * gamma + beta


def create_residual_block(prefix, x, out_ch, training, stride=1, proj=False):
    # X --> CONV + BN + RELU + CONV + BN

    y = conv2d(x, out_ch, 3, padding=PADDING, strides=stride,
               data_format=DATA_FORMAT, name=prefix+'_conv1_3x3')

    y = group_normalization(y, prefix+'_gn0', 32)
    # y = group_norm(y, 32, channels_axis=-3, reduction_axes=[-2, -1])
    y = tf.nn.relu(y)

    y = conv2d(y, out_ch, 3, padding=PADDING, strides=1,
               data_format=DATA_FORMAT, name=prefix+'_conv2_3x3')

    y = group_normalization(y, prefix+'_gn1', 32)
    # y = group_norm(y, 32, channels_axis=-3, reduction_axes=[-2, -1])

    if proj:
        x = conv2d(x, out_ch, 1, padding=PADDING, strides=stride,
                   data_format=DATA_FORMAT, name=prefix+'_proj_conv')

        x = group_normalization(x, prefix+'_proj_gn', 32)
        # x = group_norm(x, 32, channels_axis=-3, reduction_axes=[-2, -1])

    y = x + y
    y = tf.nn.relu(y)

    return y


def resnet18(x, num_classes, training):

    x = conv2d(x, 64, 7, padding=PADDING, strides=2, data_format=DATA_FORMAT, name='conv1')
    x = group_normalization(x, 'gn0', 32)
    # x = group_norm(x, 32, channels_axis=-3, reduction_axes=[-2, -1])
    # x = batch_norm(x, decay=0.9, scale=True, is_training=training, epsilon=BN_EPS, data_format='NCHW')
    x = tf.nn.relu(x)

    stage = [2, 2, 2, 2]
    channels = [64, 128, 256, 512]

    for i, s, c in zip(list(range(4)), stage, channels):
        for l in range(s):
            stride = 1 if l > 0 else 2
            proj = False if l > 0 else True
            x = create_residual_block('res_block_{}_{}'.format(i+1, l+1),
                                      x, c, stride=stride, proj=proj,
                                      training=training)

    # BxCx1x1
    x = flatten(x)
    x = slim.fully_connected(x, num_classes, activation_fn=None)

    return x


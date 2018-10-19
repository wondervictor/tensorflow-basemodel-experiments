"""

ResNet18 model

"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.layers import conv2d, flatten
from tensorflow.contrib.layers import batch_norm


DATA_FORMAT = 'channels_first'
PADDING = 'SAME'
BN_EPS = 1e-5


def create_residual_block(prefix, x, out_ch, training, stride=1, proj=False):
    # X --> CONV + BN + RELU + CONV + BN

    y = conv2d(x, out_ch, 3, padding=PADDING, strides=stride,
               data_format=DATA_FORMAT, name=prefix+'_conv1_3x3')
    y = batch_norm(y, decay=0.9, scale=True, epsilon=BN_EPS, is_training=training, data_format='NCHW')
    y = tf.nn.relu(y)
    y = conv2d(y, out_ch, 3, padding=PADDING, strides=1,
               data_format=DATA_FORMAT, name=prefix+'_conv2_3x3')
    y = batch_norm(y, decay=0.9, scale=True, epsilon=BN_EPS, is_training=training, data_format='NCHW')
    if proj:
        x = conv2d(x, out_ch, 1, padding=PADDING, strides=stride,
                   data_format=DATA_FORMAT, name=prefix+'_proj_conv')
        x = batch_norm(x, decay=0.9, scale=True, is_training=training, epsilon=BN_EPS, data_format='NCHW')

    y = x + y
    y = tf.nn.relu(y)
    return y


def resnet18(x, num_classes, training):

    x = conv2d(x, 64, 7, padding=PADDING, strides=2, data_format=DATA_FORMAT, name='conv1')
    x = batch_norm(x, decay=0.9, scale=True, is_training=training, epsilon=BN_EPS, data_format='NCHW')
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


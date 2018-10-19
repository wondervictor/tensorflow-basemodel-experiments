"""
Image Classification with CIFAR-10

"""

import tqdm
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from resnet18 import resnet18
from dataset import get_data_loader


def get_learning_rate(lr, epoch):
    if epoch < 80:
        return lr
    elif epoch < 120:
        return lr * 0.1
    elif epoch < 160:
        return lr * 1e-2
    elif epoch < 210:
        return lr * 1e-3


def weight_decay_param_filter(name):
    if name.find('kernel') != -1 or name.find('weights') != -1:
        return True
    else:
        return False


def train_model(args):

    bn_training = tf.placeholder(dtype=tf.bool, shape=[], name='bn_training')
    x = tf.placeholder(dtype=tf.float32, shape=[None, 3, 32, 32], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, args.num_classes], name='y')

    summary_writer = tf.summary.FileWriter(args.log_dir)
    weight_decay = args.weight_decay

    with tf.name_scope('resnet18'):
        pred = resnet18(x, args.num_classes, bn_training)

    with tf.variable_scope('train'):

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        cross_entropy_loss = tf.reduce_mean(slim.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))

        weight_decay_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
             if weight_decay_param_filter(v.name)]
        )
        loss = cross_entropy_loss + weight_decay_loss
        train_optimizer = slim.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = train_optimizer.minimize(loss)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)

    with tf.variable_scope('test'):

        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_loader, test_loader = get_data_loader(args.data_dir, batch_size=args.batch_size,
                                                num_workers=args.num_workers)

    base_lr = args.lr
    best_acc = 0.0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        pbar = tqdm.tqdm(range(args.epoch))
        for epoch in pbar:
            lr = get_learning_rate(base_lr, epoch)
            _losses = []
            for idx, batch_data in enumerate(train_loader):
                images, labels = batch_data
                _loss, _ = sess.run([loss, train_op],
                                    feed_dict={x: images, y: labels, learning_rate: lr, bn_training: True})
                _losses.append(_loss)

            _test_accuracy = []
            for _, batch_data in enumerate(test_loader):
                images, labels = batch_data
                acc = sess.run(accuracy, feed_dict={x: images, y: labels, bn_training: True})
                _test_accuracy.append(acc)

            cur_acc = 100*np.mean(_test_accuracy)
            best_acc = max(cur_acc, best_acc)
            pbar.set_description("e:{} loss:{:.3f} acc:{:.2f}% best:{:.2f}% lr:{:.5f}".
                                 format(epoch, np.mean(_losses), cur_acc, best_acc, lr))

            pbar.set_description("e:{} loss:{:.3f} acc:{:.2f}% lr:{:.5f}".
                                 format(epoch, np.mean(_losses), 100*np.mean(_test_accuracy), lr))


def inference():
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='training data path', default='../data/cifar-10-batches-py')
    parser.add_argument('-log', '--log_dir', type=str, help='training log dir', default='log')
    parser.add_argument('-l', '--lr', type=float, default=0.1, help='base learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='training batch size')
    parser.add_argument('-e', '--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('-nw', '--num_workers', type=int, default=2, help='data workers')
    parser.add_argument('-nc', '--num_classes', type=int, default=10, help='num classes for classification')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, help='weight decay')

    args = parser.parse_args()
    train_model(args)


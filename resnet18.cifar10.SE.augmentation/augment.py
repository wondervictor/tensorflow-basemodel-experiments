"""

"""
import cv2
import random
import numpy as np


def grayscale(img):
    # R, G，B
    w = np.array([0.587, 0.299, 0.114]).reshape(1, 1, 3)
    gs = np.zeros(img.shape[:2])
    gs = (img * w).sum(axis=2, keepdims=True)
    return gs


def brightness_aug(img, val):
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    img = img * alpha

    return img


def contrast_aug(img, val):
    gs = grayscale(img)
    gs[:] = gs.mean()
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    img = img * alpha + gs * (1 - alpha)

    return img


def saturation_aug(img, val):
    gs = grayscale(img)
    alpha = 1. + val * (np.random.rand() * 2 - 1)
    img = img * alpha + gs * (1 - alpha)

    return img


def color_jitter(img, brightness, contrast, saturation):
    augs = [(brightness_aug, brightness),
            (contrast_aug, contrast),
            (saturation_aug, saturation)]
    random.shuffle(augs)

    for aug, val in augs:
        img = aug(img, val)

    return img


def image_flip(image, *, prob=0.5, flipCode=1):
    """ flip the image
    :param prob: probability to perform the flip
    :param flipCode: > 0 for horizontal, 0 for vertical, < 0 for both
    """
    rnd = np.random.rand()
    if rnd < prob:
        return cv2.flip(image, flipCode)
    else:
        return image


def augment(img):

    img = color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.4)
    img = image_flip(img, prob=0.5)
    img = np.minimum(255, np.maximum(0, img))

    return img

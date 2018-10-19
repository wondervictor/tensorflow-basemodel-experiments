"""

CIFAR-10 Dataset

"""

import pickle
import numpy as np
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


def cifar10_train_dataset(data_dir):
    """ CIFAR-10 dataset

    Args:
        data_dir: str, path for data ['data_batch_1',..., ''test_batch]
    """

    files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
             'data_batch_5']

    training_data = []
    for fp in files:
        with open(osp.join(data_dir, fp), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        batch = list(zip(batch[b'data'], batch[b'labels']))
        training_data += batch

    return training_data


def cifar10_test_dataset(data_dir):

    with open(osp.join(data_dir, 'test_batch'), 'rb') as f:
        testing_data = pickle.load(f, encoding='bytes')
    return list(zip(testing_data[b'data'], testing_data[b'labels']))


class DataAugmentator:

    def __init__(self):
        pass


class CIFAR10Dataset(Dataset):

    def __init__(self, data_dir, split_name='train'):

        if split_name == 'test':
            data = cifar10_test_dataset(data_dir)
        else:
            data = cifar10_train_dataset(data_dir)
        self._raw_data = data
        self._mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape((3, 1, 1))
        self._std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape((3, 1, 1))

    def __len__(self):
        return len(self._raw_data)

    def __getitem__(self, idx):

        img, label_idx = self._raw_data[idx]
        img = np.reshape(img, (3, 32, 32))
        # TODO: Augmentation
        img = img / 255.0
        img = (img-self._mean)/self._std
        label = np.zeros(10)
        label[label_idx] = 1

        return img, label


def collate_fn(batch):
    transposed = list(zip(*batch))
    labels = np.stack(transposed[1], 0)
    images = np.stack(transposed[0], 0)

    return images, labels


def get_data_loader(data_dir, batch_size, num_workers):

    train_dataset = CIFAR10Dataset(data_dir)
    test_dataset = CIFAR10Dataset(data_dir, split_name='test')

    train_data_loader = DataLoader(train_dataset, batch_size,
                                   shuffle=True, num_workers=num_workers,
                                   collate_fn=collate_fn)

    test_data_loader = DataLoader(test_dataset, 500, shuffle=False, num_workers=1,
                                  collate_fn=collate_fn)

    return train_data_loader, test_data_loader





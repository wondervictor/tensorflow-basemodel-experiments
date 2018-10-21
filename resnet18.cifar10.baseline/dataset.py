"""

CIFAR-10 Dataset

"""
import os
import sys
import pickle
import numpy as np
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import cifar


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


class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        img = img.data.numpy()
        label = np.zeros(10)
        label[target] = 1
        return img, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def collate_fn(batch):
    transposed = list(zip(*batch))
    labels = np.stack(transposed[1], 0)
    images = np.stack(transposed[0], 0)

    return images, labels


def get_data_loader(data_dir, batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    train_dataset = CIFAR10(data_dir, train=True, transform=transform_train)
    test_dataset = CIFAR10(data_dir, train=False, transform=transform_test)
    # train_dataset = CIFAR10Dataset(data_dir)
    # test_dataset = CIFAR10Dataset(data_dir, split_name='test')

    train_data_loader = DataLoader(train_dataset, batch_size,
                                   shuffle=True, num_workers=num_workers,
                                   collate_fn=collate_fn)

    test_data_loader = DataLoader(test_dataset, 500, shuffle=False, num_workers=1,
                                  collate_fn=collate_fn)

    return train_data_loader, test_data_loader





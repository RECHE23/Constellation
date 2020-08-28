# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from .. import *


class DataSplit:
    def __init__(self, dataset,
                 test_train_split=TEST_TRAIN_SPLIT,
                 val_train_split=VAL_TRAIN_SPLIT,
                 shuffle=False):
        """
        Constructor of the class.

        :param dataset: The dataset to split.
        :param test_train_split: The split ratio for the test set.
        :param val_train_split: The split ratio for the train set.
        :param shuffle: Should the samples be shuffled?
        """
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = \
            self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = \
            train_indices[: validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_split(self, batch_size=BATCH_SIZE):
        """
        Provide the train, validation and test sets loaders.

        :param batch_size: The batch size to be used.
        :return: A tuple composed of the train, validation and test sets loaders.
        """
        self.train_loader = self.get_train_loader(batch_size=batch_size)
        self.val_loader = self.get_validation_loader(batch_size=batch_size)
        self.test_loader = self.get_test_loader(batch_size=batch_size)
        return self.train_loader, self.val_loader, self.test_loader

    def get_train_loader(self, batch_size=BATCH_SIZE):
        """
        Provide the train set loader.

        :param batch_size: The batch size to be used.
        :return: The train set loader
        """
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size,
                                       sampler=self.train_sampler,
                                       shuffle=False)
        return self.train_loader

    def get_validation_loader(self, batch_size=BATCH_SIZE):
        """
        Provide the validation set loader.

        :param batch_size: The batch size to be used.
        :return: The validation set loader
        """
        self.val_loader = DataLoader(self.dataset, batch_size=batch_size,
                                     sampler=self.val_sampler,
                                     shuffle=False)
        return self.val_loader

    def get_test_loader(self, batch_size=BATCH_SIZE):
        """
        Provide the test set loader.

        :param batch_size: The batch size to be used.
        :return: The test set loader
        """
        self.test_loader = DataLoader(self.dataset,
                                      batch_size=batch_size,
                                      sampler=self.test_sampler,
                                      shuffle=False)
        return self.test_loader

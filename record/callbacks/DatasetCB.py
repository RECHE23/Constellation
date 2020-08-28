# -*- coding: utf-8 -*-

from poutyne.framework import Callback
from poutyne.utils import torch_to_numpy


class DatasetCB(Callback):
    """
    This callback saves the information about the dataset.
    """
    def __init__(self, base_callback, dataset, batch_size):
        """
        Constructor of the class.

        :param base_callback: The base callback.
        :param dataset: The dataset being used.
        :param batch_size: The size of a batch.
        """
        super().__init__()
        self.base_callback = base_callback
        self.dataset = dataset
        self.data = dataset.data
        self.targets = torch_to_numpy(dataset.targets)
        self.record = self.base_callback.record
        self.experiment = self.base_callback.experiment

        self.experiment['data'] = torch_to_numpy(dataset.data)
        self.experiment['targets'] = torch_to_numpy(dataset.targets)
        self.experiment['feature_names'] = dataset.feature_names
        self.experiment['target_names'] = dataset.feature_names
        self.experiment['dataset_name'] = dataset.name
        self.experiment['batch_size'] = batch_size

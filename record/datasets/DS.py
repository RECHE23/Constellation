# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from poutyne.utils import numpy_to_torch


class DS(Dataset):
    """
    A wrapper class used to import datasets into a PyTorch friendly format.
    """

    def __init__(self, data: np.ndarray, target: np.ndarray,
                 feature_names: list = None, target_names: list = None,
                 description: str = None, device: torch.device = None,
                 name: str = None) -> None:
        """
        Constructor of the class.

        :param data: The original samples matrix (as a PyTorch Tensor).
        :param target: The target classes (-1 or 1, as a PyTorch Tensor).
        :param feature_names: The name of the features, as stored in 'data'.
        :param target_names: The target names (1 is the specified class,
                                              -1 isn't the specified class).
        :param description: The description of the dataset.
        :param device: The device used by PyTorch.
        :param name: The name of the dataset.
        """
        self.type = 'Dataset'
        self.data = numpy_to_torch(data)
        self.targets = numpy_to_torch(target.reshape(-1, 1))
        self.feature_names = feature_names
        self.target_names = target_names
        self.description = description
        self.name = name
        self.device = None
        self.to(device)
        super(DS, self).__init__()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> [Tensor, Tensor]:
        """
        Returns a sample and its label for a given index.
        """
        return [self.data[index], self.targets[index]]

    def to(self, device: torch.device) -> None:
        """
        Assigns a device to the dataset.

        :param device: Specified device.
        :return: The PyTorch device.
        """
        self.data = self.data.to(device=device, dtype=torch.float)
        self.targets = self.targets.to(device=device, dtype=torch.float)

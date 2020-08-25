# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from record import *
from .. import *


class MNIST784DS(DS):
    """
    A dataset imported from OpenML with 10 classes and 784 dimensions.
    """

    def __init__(self, device: torch.device = None,
                 selected_class: str = '0', scale: bool = False,
                 fraction: float = 0.1) -> None:
        """
        Constructor of the class.

        :param device: The device used by PyTorch.
        :param selected_class: The selected class (0 to 9)
        :param scale: Should we scale the values?
        :param fraction: Fraction of the MNIST dataset to be used.
        """
        assert 0. < fraction <= 1.
        data = fetch_openml('mnist_784', version=1,
                            data_home=DATASETS_PATH,
                            return_X_y=False)
        X, y = data['data'].astype(dtype=np.float), data['target']
        X, _, y, _ = train_test_split(X, y, stratify=y,
                                      test_size=(1 - fraction),
                                      random_state=DEFAULT_SEED)
        y = -np.where(y == selected_class, -1.,
                      np.ones(shape=y.shape, dtype=np.float))
        y = y.astype(dtype=np.float)
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        feature_names = data['feature_names']
        target_names = [selected_class, f'not {selected_class}']
        description = data['DESCR']
        name = "MNIST 784"
        super(MNIST784DS, self).__init__(data=X, target=y,
                                         feature_names=feature_names,
                                         target_names=target_names,
                                         description=description,
                                         device=device,
                                         name=name)

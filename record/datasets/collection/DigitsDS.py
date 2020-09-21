# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from .. import *


class DigitsDS(DS):
    """
    A dataset imported from Sci-Kit Learn with 10 classes and 64 dimensions.
    A class can be selected during initialisation.
    """

    def __init__(self, device: torch.device = None,
                 selected_class: int = 1, scale: bool = False) -> None:
        """
        Constructor of the class.

        :param device: The device used by PyTorch.
        :param selected_class: The selected class (0 to 9)
        :param scale: Should we scale the values?
        """
        data = load_digits(return_X_y=False)
        X, y = data['data'], data['target']
        y = -np.where(y == selected_class, -1.,
                      np.ones(shape=y.shape, dtype=float))
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        feature_names = data['feature_names']
        target_names = [data['target_names'][selected_class],
                        f'not {data["target_names"][selected_class]}']
        description = data['DESCR']
        name = "Digits"
        super(DigitsDS, self).__init__(data=X, target=y,
                                       feature_names=feature_names,
                                       target_names=target_names,
                                       description=description,
                                       device=device,
                                       name=name)

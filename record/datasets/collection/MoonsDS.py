# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from .. import *


class MoonsDS(DS):
    """
    A prebuilt two moons dataset from Sci-Kit Learn.
    """

    def __init__(self, device: torch.device = None,
                 scale: bool = False, size: int = 256) -> None:
        """
        Constructor of the class.

        :param device: The device used by PyTorch.
        :param scale: Should we scale the values?
        """
        X, y = make_moons(size, noise=0.125, random_state=0)
        y = np.where(y == 0, -1., y)
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        feature_names = ['x_1', 'x_2']
        target_names = ['upper moon', 'lower moon']
        description = "Make two interleaving half circles.\n" \
                      "A simple toy dataset to visualize clustering " \
                      "and classification algorithms."
        name = "Moons"
        super(MoonsDS, self).__init__(data=X, target=y,
                                      feature_names=feature_names,
                                      target_names=target_names,
                                      description=description,
                                      device=device,
                                      name=name)

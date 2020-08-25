# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from .. import *


class CirclesDS(DS):
    """
    A prebuilt concentric circles dataset from Sci-Kit Learn.
    """

    def __init__(self, device: torch.device = None,
                 scale: bool = False, size: int = 256) -> None:
        """
        Constructor of the class.

        :param device: The device used by PyTorch.
        :param scale: Should we scale the values?
        """
        X, y = make_circles(n_samples=size, noise=0.09, random_state=0)
        y = np.where(y == 0, -1., y)
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        feature_names = ['x_1', 'x_2']
        target_names = ['inner circle', 'outer circle']
        description = "Make a large circle containing a smaller circle in " \
                      "2d.\nA simple toy dataset to visualize clustering " \
                      "and classification algorithms."
        name = "Circles"
        super(CirclesDS, self).__init__(data=X, target=y,
                                        feature_names=feature_names,
                                        target_names=target_names,
                                        description=description,
                                        device=device,
                                        name=name)

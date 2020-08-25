# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from .. import *


class Blobs5DDS(DS):
    """
    A prebuilt 5D blobs dataset from Sci-Kit Learn.
    """

    def __init__(self, device: torch.device = None,
                 scale: bool = False, size: int = 256) -> None:
        """
        Constructor of the class.

        :param device: The device used by PyTorch.
        :param scale: Should we scale the values?
        """
        X, y = make_blobs(n_samples=size, n_features=5, random_state=3,
                          centers=[[0.32, 0.13, -0.17, 0.31, -0.32],
                                   [-0.26, -0.31, 0.23, -0.37, 0.29]],
                          cluster_std=0.75)
        y = np.where(y == 0, -1., y)
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        feature_names = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5']
        target_names = ['blob 1', 'blob 2']
        description = "Generate isotropic Gaussian blobs for clustering.\n" \
                      "A simple toy dataset to visualize clustering " \
                      "and classification algorithms."
        name = "5D Blobs"
        super(Blobs5DDS, self).__init__(data=X, target=y,
                                        feature_names=feature_names,
                                        target_names=target_names,
                                        description=description,
                                        device=device,
                                        name=name)

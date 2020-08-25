# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from .. import *


class BlobsDS(DS):
    """
    A prebuilt two blobs dataset from Sci-Kit Learn.
    """

    def __init__(self, device: torch.device = None,
                 scale: bool = False, size: int = 256) -> None:
        """
        Constructor of the class.

        :param device: The device used by PyTorch.
        :param scale: Should we scale the values?
        """
        X, y = make_blobs(n_samples=size, centers=2, n_features=2,
                          random_state=0)
        y = np.where(y == 0, -1., y)
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        feature_names = ['x_1', 'x_2']
        target_names = ['blob 1', 'blob 2']
        description = "Generate isotropic Gaussian blobs for clustering.\n" \
                      "A simple toy dataset to visualize clustering " \
                      "and classification algorithms."
        name = "Blobs"
        super(BlobsDS, self).__init__(data=X, target=y,
                                      feature_names=feature_names,
                                      target_names=target_names,
                                      description=description,
                                      device=device,
                                      name=name)

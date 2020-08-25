# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from .. import *


class BreastCancerDS(DS):
    """
    A dataset imported from Sci-Kit Learn with 2 classes and 20 dimensions.
    """

    def __init__(self, device: torch.device = None,
                 scale: bool = False) -> None:
        """
        Constructor of the class.

        :param device: The device used by PyTorch.
        :param scale: Should we scale the values?
        """
        data = load_breast_cancer(return_X_y=False)
        X, y = data['data'], data['target']
        y = np.where(y == 0, -1., y)
        if scale:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        feature_names = data['feature_names']
        target_names = data['target_names']
        description = data['DESCR']
        name = "Breast cancer"
        super(BreastCancerDS, self).__init__(data=X, target=y,
                                             feature_names=feature_names,
                                             target_names=target_names,
                                             description=description,
                                             device=device,
                                             name=name)

# -*- coding: utf-8 -*-

import torch
from torch import Tensor


def accuracy(input: Tensor, target: Tensor) -> float:
    """
    Accuracy for [-1, +1] binary classification.

    :param input: A Tensor of characteristics.
    :param target: The correct label associated to the characteristics.
    :return: The accuracy score.
    """
    prediction = torch.sign(input)
    return (prediction == target).float().sum() / target.shape[0]

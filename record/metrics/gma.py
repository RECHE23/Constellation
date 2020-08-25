# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from sys import float_info

eps = float_info.epsilon


def gma(input: Tensor, target: Tensor) -> float:
    """
    Geometric mean of accuracy for classification.

    :param input: A Tensor of characteristics.
    :param target: The correct label associated to the characteristics.
    :return: The geometric mean of accuracy.
    """
    prediction = torch.sign(input)
    C = torch.unique(target)
    A = torch.empty(C.shape, dtype=torch.float)
    for i, c in enumerate(C):
        s = target == c
        t = (prediction == c) & s
        A[i] = t.float().sum() / (s.float().sum() + eps)

    return torch.prod(A) ** (1 / len(C))

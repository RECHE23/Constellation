# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torch.nn import Module


class Swish(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Swish}(x) = x\cdot\sigma(x) = \frac{x}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = Swish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(input)

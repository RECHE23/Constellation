# -*- coding: utf-8 -*-

"""
INSTRUCTIONS:

1- Use the class 'DS' to wrap your datasets.

    For example, the minimalist usage of the 'DS' class is:
    >>> ExampleDS = DS(data=X, target=y)
    Where X and y are Numpy arrays.

    It is also possible to use class inheritance to define datasets.
    For examples, see the content of 'Constellation/datasets/collection'.

    To have a better understanding of how to use the 'DS' class, have a look at
    the class definition. It is defined in 'Constellation/datasets/DS.py'.

2- Insert your datasets in the 'datasets' list.

    Constellation retrieves the datasets from the 'datasets' list.

"""

from record import BlobsDS, MoonsDS, CirclesDS, IrisDS, Blobs5DDS, \
    MNIST784DS, BreastCancerDS, DS

datasets = [
    BlobsDS(scale=True),
    MoonsDS(scale=True),
    CirclesDS(scale=True),
    # BreastCancerDS(scale=True),
    # Blobs5DDS(scale=True),
    # IrisDS(scale=True, selected_class=1),
    # MNIST784DS(scale=True, selected_class='3')
           ]

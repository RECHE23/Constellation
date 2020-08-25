# -*- coding: utf-8 -*-

from sklearn.decomposition import KernelPCA
from .. import DEFAULT_SEED


def kernel_pca(n_components: int = None, random_state: int = DEFAULT_SEED,
               kernel: str = "rbf", **kwargs):
    return KernelPCA(n_components=n_components,
                     random_state=random_state,
                     kernel=kernel,
                     fit_inverse_transform=True,
                     **kwargs)

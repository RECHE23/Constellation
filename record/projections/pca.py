# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from .. import DEFAULT_SEED


def pca(n_components: int = None, random_state: int = DEFAULT_SEED, **kwargs):
    return PCA(n_components=n_components,
               random_state=random_state,
               **kwargs)

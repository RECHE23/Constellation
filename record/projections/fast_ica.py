# -*- coding: utf-8 -*-

from sklearn.decomposition import FastICA
from .. import DEFAULT_SEED


def fast_ica(n_components: int = None, random_state: int = DEFAULT_SEED,
             **kwargs):
    return FastICA(n_components=n_components,
                   random_state=random_state,
                   **kwargs)

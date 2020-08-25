# -*- coding: utf-8 -*-

from sklearn.decomposition import TruncatedSVD
from .. import DEFAULT_SEED


def truncated_svd(n_components: int = None, random_state: int = DEFAULT_SEED,
                  **kwargs):
    return TruncatedSVD(n_components=n_components,
                        random_state=random_state,
                        **kwargs)

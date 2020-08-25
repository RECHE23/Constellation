# -*- coding: utf-8 -*-

from umap import UMAP
from .. import DEFAULT_SEED


def umap(n_components: int = None, random_state: int = DEFAULT_SEED, **kwargs):
    return UMAP(n_components=n_components,
                random_state=random_state,
                transform_seed=random_state,
                target_metric='categorical',
                verbose=False,
                **kwargs)

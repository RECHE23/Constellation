# -*- coding: utf-8 -*-

from .LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from .. import DEFAULT_SEED


def lda(n_components: int = None, random_state: int = DEFAULT_SEED, **kwargs):
    return LinearDiscriminantAnalysis(n_components=n_components,
                                      random_state=random_state,
                                      **kwargs)

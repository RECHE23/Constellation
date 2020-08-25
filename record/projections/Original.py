# -*- coding: utf-8 -*-

from .. import DEFAULT_SEED


class Original(object):
    def __init__(self, n_components=None, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        return

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


def original(n_components: int = None, random_state: int = DEFAULT_SEED,
             **kwargs):
    return Original(n_components=n_components,
                    random_state=random_state)

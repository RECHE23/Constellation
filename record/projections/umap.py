# -*- coding: utf-8 -*-

from umap import UMAP
from .. import DEFAULT_SEED


def umap(n_components: int = None, random_state: int = DEFAULT_SEED, **kwargs):
    return UMAP(n_components=2,
                random_state=random_state,
                transform_seed=random_state,
                target_metric='categorical',
                verbose=False,
                a=None, angular_rp_forest=False, b=None,
                force_approximation_algorithm=False, init='spectral',
                learning_rate=1.0,
                local_connectivity=1.0, low_memory=False,
                metric='euclidean',
                metric_kwds=None, min_dist=0.1,
                n_epochs=None,
                n_neighbors=15, negative_sample_rate=5,
                output_metric='euclidean',
                output_metric_kwds=None,
                repulsion_strength=1.0,
                set_op_mix_ratio=1.0, spread=1.0,
                target_metric_kwds=None, target_n_neighbors=-1,
                target_weight=0.5,
                transform_queue_size=4.0,
                unique=False,
                **kwargs
                )

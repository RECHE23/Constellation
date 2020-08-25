# -*- coding: utf-8 -*-

from .fast_ica import fast_ica
from .kernel_pca import kernel_pca
from .lda import lda
from .Original import original
from .pca import pca
from .truncated_svd import truncated_svd
from .umap import umap

projections_list = {'fast_ica': fast_ica,
                    'kernel_pca': kernel_pca,
                    'lda': lda,
                    'original': original,
                    'pca': pca,
                    'umap': umap}

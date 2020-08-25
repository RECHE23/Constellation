from .fast_ica import fast_ica
from .kernel_pca import kernel_pca
from .lda import lda
from .pca import pca
from .truncated_svd import truncated_svd
from .umap import umap
from .LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from .list import projections_list

__all__ = ['fast_ica',
           'kernel_pca',
           'lda',
           'pca',
           'truncated_svd',
           'umap',
           'LinearDiscriminantAnalysis',
           'projections_list']

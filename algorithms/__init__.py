"""
Clustering Algorithms Package.

This package contains implementations of various clustering algorithms required
for Work 3. It includes wrappers for Scikit-Learn algorithms (Agglomerative, GMM)
and custom "Own Code" implementations for K-Means variants, Fuzzy Clustering,
and PCA.

Modules
-------
- agg_clustering: Wrapper for Agglomerative Clustering.
- gmm_clustering: Wrapper for Gaussian Mixture Models.
- kmeans: Standard K-Means (Lloyd's Algorithm).
- kmeansfekm: Far Efficient K-Means (Improved Initialization).
- kernel_kmeans: Intelligent Kernel K-Means (Non-linear).
- fuzzy_c_means: Fuzzy C-Means (Standard and Suppressed).
- pca: Custom Principal Component Analysis.
"""

from .agg_clustering import run_agglomerative_once
from .gmm_clustering import run_gmm_once
from .kmeans import KMeans
from .kmeansfekm import KMeansFEKM
from .kernel_kmeans import KernelKMeans
from .fuzzy_c_means import FuzzyCMeans
from .pca import PCA
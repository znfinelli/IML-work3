"""
Agglomerative Clustering Implementation (Wrapper).

This script provides a wrapper around Scikit-Learn's AgglomerativeClustering
to facilitate Task 1.2 of the assignment. It allows for dynamic configuration
of distance metrics (e.g., Euclidean, Cosine) and linkage methods (Single,
Average, Complete, Ward).

References
----------
[1] Work 3 Description, UB, 2025, "1.1.1 Tasks", Point 2, pp. 2.
[2] Support Slides Session 1, Salamó, 2025, "Agglomerative Clustering", pp. 12-23.
"""

import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, Any


def run_agglomerative_once(
        X: np.ndarray,
        n_clusters: int,
        metric: str,
        linkage: str,
        dataset_name: str,
) -> Dict[str, Any]:
    """
    Runs Agglomerative Clustering once with a specific configuration.

    This function instantiates the sklearn model, fits it to the data,
    measures execution time, and returns the results for analysis.

    Parameters
    ----------
    X : np.ndarray
        The input feature matrix (will be cast to float32).
    n_clusters : int
        The number of clusters to find.
    metric : str
        The distance metric to use. Valid options depend on linkage:
        - "euclidean", "cosine", "manhattan" (if linkage != "ward").
        - "euclidean" is mandatory if linkage is "ward".
    linkage : str
        The linkage criterion ("ward", "complete", "average", "single").
        [1] Work 3 Description, UB, 2025, pp. 2.
    dataset_name : str
        Name of the dataset being processed (for logging purposes).

    Returns
    -------
    dict
        A dictionary containing:
        - Metadata (algorithm, parameters, runtime).
        - "labels": The resulting cluster labels (kept in memory).
    """
    # Optimization: Float32 is often sufficient for clustering and saves memory
    X = X.astype(np.float32)

    start = time.perf_counter()

    # Note: 'metric' replaces 'affinity' in newer sklearn versions (>1.2)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage
    )

    labels = model.fit_predict(X)
    runtime = time.perf_counter() - start

    result = {
        "dataset": dataset_name,
        "algorithm": "Agglomerative",
        "n_clusters": n_clusters,
        "metric": metric,
        "linkage": linkage,
        "runtime_sec": runtime,
        # labels are NOT CSV-friendly; keep them in memory only for validation
        "labels": labels,
    }
    return result
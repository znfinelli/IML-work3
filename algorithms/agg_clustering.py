import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def run_agglomerative_once(
    X: np.ndarray,
    n_clusters: int,
    metric: str,
    linkage: str,
    dataset_name: str,
) -> dict:
    """
    Runs AgglomerativeClustering once with given configuration.
    Returns a dict with config + runtime (labels kept in-memory).
    """
    X = X.astype(np.float32)

    start = time.perf_counter()
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
        # labels are NOT CSV-friendly; keep them in memory only
        "labels": labels,
    }
    return result

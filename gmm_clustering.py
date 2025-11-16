import time
import numpy as np
from sklearn.mixture import GaussianMixture

def run_gmm_once(
    X: np.ndarray,
    n_components: int,
    init_params: str,
    dataset_name: str,
) -> dict:
    """
    Runs GaussianMixture once with given configuration.
    Returns config + runtime + BIC/log-likelihood.
    """
    start = time.perf_counter()
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        init_params=init_params,
        random_state=42
    )
    gmm.fit(X)
    labels = gmm.predict(X)
    runtime = time.perf_counter() - start

    result = {
        "dataset": dataset_name,
        "algorithm": "GaussianMixture",
        "n_components": n_components,
        "init_params": init_params,
        "runtime_sec": runtime,
        "bic": gmm.bic(X),
        "avg_log_likelihood": gmm.score(X),
        "labels": labels,
    }
    return result

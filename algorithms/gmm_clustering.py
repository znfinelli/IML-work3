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

    # --- SAFETY CHECK 1: Remove any lingering NaNs ---
    # If parser missed something, this catches it to prevent crash
    if np.isnan(X).any():
        X = np.nan_to_num(X)

    # --- SAFETY CHECK 2: Adjust Regularization ---
    # Mushroom is 100% categorical (binary), so it needs the strongest regularization.
    # Adult is mixed, so 1e-3 is usually enough.
    if dataset_name == "mushroom":
        reg_covar = 1e-1  # Stronger regularization for pure binary data
    elif dataset_name == "adult":
        reg_covar = 1e-3  # Moderate regularization for mixed data
    else:
        reg_covar = 1e-4  # Default for numeric data (pen-based)

    start = time.perf_counter()
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        init_params=init_params,
        reg_covar=reg_covar,  # <--- UPDATED
        random_state=42
    )

    try:
        gmm.fit(X)
        labels = gmm.predict(X)

        # Calculate scores
        bic = gmm.bic(X)
        score = gmm.score(X)

    except Exception as e:
        # If it still crashes, return a "failed" result instead of killing the script
        print(f"GMM Crash prevented on {dataset_name} (k={n_components}): {e}")
        return {
            "dataset": dataset_name,
            "algorithm": "GaussianMixture",
            "n_components": n_components,
            "init_params": init_params,
            "runtime_sec": 0,
            "bic": 0,
            "avg_log_likelihood": 0,
            "labels": np.zeros(X.shape[0]),  # Return dummy labels
            "error": str(e)
        }

    runtime = time.perf_counter() - start

    result = {
        "dataset": dataset_name,
        "algorithm": "GaussianMixture",
        "n_components": n_components,
        "init_params": init_params,
        "runtime_sec": runtime,
        "bic": bic,
        "avg_log_likelihood": score,
        "labels": labels,
    }
    return result
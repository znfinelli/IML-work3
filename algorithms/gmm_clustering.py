"""
Gaussian Mixture Model Implementation (Wrapper).

This script provides a wrapper around Scikit-Learn's GaussianMixture class
to fulfill Task 1.3. It is designed to handle different initialization
parameters and component counts while enforcing the 'full' covariance type
as strictly required by the project description.

References
----------
[1] Work 3 Description, UB, 2025, "1.1.1 Tasks", Point 3, pp. 2.
[2] Support Slides Session 1, Salamó, 2025, "Gaussian Mixture Models in Python", pp. 29.
"""

import time
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict, Any


def run_gmm_once(
        X: np.ndarray,
        n_components: int,
        init_params: str,
        dataset_name: str,
) -> Dict[str, Any]:
    """
    Runs Gaussian Mixture Model clustering once with a specific configuration.

    This function enforces `covariance_type="full"` as per assignment rules.
    It includes regularization stability checks to prevent crashes on high-dimensional
    categorical datasets (e.g., Mushroom, Adult) where covariance matrices
    may become singular.

    Parameters
    ----------
    X : np.ndarray
        The input feature matrix.
    n_components : int
        The number of mixture components to use.
    init_params : str
        The initialization method ('kmeans', 'random', etc.).
    dataset_name : str
        The name of the dataset (used to tune stability regularization).

    Returns
    -------
    dict
        A dictionary containing:
        - Metadata (algorithm, parameters, runtime).
        - "bic": Bayesian Information Criterion score.
        - "avg_log_likelihood": The average log likelihood of the data.
        - "labels": The resulting cluster assignments.
    """

    # --- SAFETY CHECK 1: Remove any lingering NaNs ---
    # Although parser handles this, a final check prevents C-level crashes in sklearn
    if np.isnan(X).any():
        X = np.nan_to_num(X)

    # --- SAFETY CHECK 2: Adjust Regularization ---
    # GMM 'full' covariance is sensitive to singular matrices in categorical data.
    # We apply stronger regularization (reg_covar) for known categorical sets.
    if dataset_name == "mushroom":
        reg_covar = 1e-1  # Stronger regularization for pure binary/sparse data
    elif dataset_name == "adult":
        reg_covar = 1e-3  # Moderate regularization for mixed data
    else:
        reg_covar = 1e-4  # Default sklearn behavior for numeric data

    start = time.perf_counter()

    # Requirement: "For the covariance type maintain it as 'full'." [Work 3, p. 2]
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        init_params=init_params,
        reg_covar=reg_covar,
        random_state=42
    )

    try:
        gmm.fit(X)
        labels = gmm.predict(X)

        # Calculate scores for analysis
        bic = gmm.bic(X)
        score = gmm.score(X)

    except Exception as e:
        # Return a "failed" result structure to ensure experiment loops don't break
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
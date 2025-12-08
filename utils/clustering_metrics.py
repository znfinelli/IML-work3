"""
Clustering validation metrics for Work 3.

This module implements the evaluation metrics required to analyze clustering
performance as specified in Section 1.1.2 of the assignment. It includes
wrappers for Scikit-Learn metrics (ARI, DBI) and custom implementations
for Purity and F-Measure based on the course slides.

References
----------
[1] Work 3 Description, UB, 2025, "1.1.2 Presenting and Interpreting Clustering Results", pp. 4.
[2] Support Slides Session 3, UB, 2025, "External Indexes".
"""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    confusion_matrix,
)
from typing import Dict, Any, Union


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Purity score for a clustering result.

    Purity is defined as the ratio of correctly assigned points to the total number
    of points, assuming each cluster is assigned to the class which is most
    frequent in that cluster.

    Formula
    -------
    Purity = (1 / N) * sum_k (max_j (n_kj))
    where n_kj is the number of points in cluster k belonging to class j.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_pred : np.ndarray
        Predicted cluster labels.

    Returns
    -------
    float
        The purity score in range [0, 1].

    References
    ----------
    [1] Support Slides Session 3, UB, 2025, "Purity Example", Slide 15[cite: 902].
    [2] Work 3 Description, UB, 2025, p. 4.
    """
    # Rows = True Classes, Cols = Predicted Clusters
    cm = confusion_matrix(y_true, y_pred)

    # axis=0 looks down columns (finding max class count per cluster)
    # This aligns with the definition: sum(max intersection) / total samples
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def f_measure_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the overall F-measure for the clustering.

    This implements the definition provided in Session 3, Table 1 (Row 3).
    It calculates the F-measure for each class against the clusters and 
    computes a weighted average based on class size.

    Formula
    -------
    F = sum_i (n_i / N) * max_j (F(i, j))
    where F(i, j) is the F1-score for class i and cluster j.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_pred : np.ndarray
        Predicted cluster labels.

    Returns
    -------
    float
        The weighted F-measure score in range [0, 1].

    References
    ----------
    [1] Support Slides Session 3, UB, 2025, "Summary of external indexes", Table 1, Row 3.
    [2] Work 3 Description, UB, 2025, p. 4.
    """
    cm = confusion_matrix(y_true, y_pred)
    N = cm.sum()
    n_classes, n_clusters = cm.shape

    F_overall = 0.0

    # Iterate over every true class 'i'
    for i in range(n_classes):
        n_c = cm[i, :].sum()  # Total items in class i
        if n_c == 0:
            continue

        best_F_ck = 0.0

        # Check against every cluster 'j' to find the best match for class 'i'
        for j in range(n_clusters):
            n_k = cm[:, j].sum()  # Total items in cluster j
            n_ck = cm[i, j]  # Intersection

            if n_ck == 0 or n_k == 0:
                continue

            precision = n_ck / n_k
            recall = n_ck / n_c

            if precision + recall == 0:
                continue

            F_ck = 2 * precision * recall / (precision + recall)
            if F_ck > best_F_ck:
                best_F_ck = F_ck

        # Add weighted contribution
        F_overall += (n_c / N) * best_F_ck

    return F_overall


def compute_clustering_metrics(
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Computes a suite of clustering validation metrics.

    Calculates ARI, Purity, Davies-Bouldin Index, and F-Measure as requested
    in the assignment description.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix (required for Davies-Bouldin Index).
    y_true : np.ndarray
        The ground truth class labels.
    y_pred : np.ndarray
        The predicted cluster labels.

    Returns
    -------
    Dict[str, float]
        Dictionary containing 'ari', 'purity', 'davies_bouldin', and 'f_measure'.

    References
    ----------
    [1] Work 3 Description, UB, 2025, "1.1.2 Presenting and Interpreting Clustering Results", p. 4.
    """
    ari = adjusted_rand_score(y_true, y_pred)
    dbi = davies_bouldin_score(X, y_pred)
    purity = purity_score(y_true, y_pred)
    f_measure = f_measure_score(y_true, y_pred)

    return {
        "ari": ari,
        "purity": purity,
        "davies_bouldin": dbi,
        "f_measure": f_measure,
    }


def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False
) -> np.ndarray:
    """
    Generates the confusion matrix between true classes and clusters.

    This assists in the requirement: "To show the results, you can use a
    confusion matrix, for example".

    Parameters
    ----------
    y_true : np.ndarray
        True class labels.
    y_pred : np.ndarray
        Cluster labels.
    normalize : bool, default=False
        If True, normalizes the matrix by true class counts (rows sum to 1).

    Returns
    -------
    np.ndarray
        The confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm.astype(float) / row_sums
    return cm
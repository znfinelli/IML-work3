import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    confusion_matrix,
)


def purity_score(y_true, y_pred) -> float:
    """
    Purity = sum_c max_k n_ck / N.
    """
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=1)) / np.sum(cm)


def f_measure_score(y_true, y_pred) -> float:
    """
    Overall clustering F-measure:
      For each true class c:
        F_c = max_k F1(c, k)
      F = sum_c (n_c / N) * F_c
    """
    cm = confusion_matrix(y_true, y_pred)
    N = cm.sum()
    n_classes, n_clusters = cm.shape

    F_overall = 0.0

    for i in range(n_classes):
        n_c = cm[i, :].sum()
        if n_c == 0:
            continue

        best_F_ck = 0.0
        for j in range(n_clusters):
            n_k = cm[:, j].sum()
            n_ck = cm[i, j]

            if n_ck == 0 or n_k == 0:
                continue

            precision = n_ck / n_k
            recall = n_ck / n_c
            if precision + recall == 0:
                continue

            F_ck = 2 * precision * recall / (precision + recall)
            if F_ck > best_F_ck:
                best_F_ck = F_ck

        F_overall += (n_c / N) * best_F_ck

    return F_overall


def compute_clustering_metrics(X, y_true, y_pred):
    """
    Compute ARI, purity, Davies–Bouldin index, and F-measure.
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


def get_confusion_matrix(y_true, y_pred, normalize: bool = False):
    """
    Convenience helper if you later want to inspect / save confusion matrices
    for selected configurations.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    return cm

"""
Confusion Matrix Generator.

This script executes the specific visualization task of comparing clustering
results against ground truth labels using a Confusion Matrix. It is designed
specifically for the Pen-based dataset (digits 0-9) to illustrate specific
misclassifications (e.g., confusing '1' with '7').

It utilizes the Hungarian Algorithm to optimally map unsupervised cluster IDs
to true class labels before plotting.

References
----------
[1] Work 3 Description, UB, 2025, "1.1.2 Presenting and Interpreting Clustering Results".

Usage:
    Run from project root: python -m analysis.confusion_matrix
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Import project modules
from utils.parser import preprocess_single_arff
from algorithms.kmeansfekm import KMeansFEKM

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CM_CONFIG = {
    "dataset_name": "pen-based",
    "dataset_path": "datasets/pen-based.arff",
    "output_directory": "plots",
    "output_filename": "pen_based_confusion_matrix.png",
    "n_clusters": 10,       # Known ground truth for digits 0-9
    "random_state": 42,     # For reproducibility
    "figsize": (10, 8),     # Plot dimensions
    "cmap": "Blues",        # Color map for heatmap
    "dpi": 150
}
# ---------------------------------------------------------


def map_clusters_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Maps clustering labels to ground truth labels using the Hungarian Algorithm.
    This ensures Cluster 'X' aligns with Digit 'X' based on maximum overlap.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Raw cluster assignments (0..k-1).

    Returns
    -------
    np.ndarray
        Remapped predictions matching the ground truth label space.
    """
    # Create contingency matrix
    n_classes = max(y_true.max(), y_pred.max()) + 1
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(y_pred.size):
        matrix[y_pred[i], y_true[i]] += 1

    # Find optimal assignment (maximize trace / diagonal overlap)
    # linear_sum_assignment minimizes cost, so we pass negative counts
    row_ind, col_ind = linear_sum_assignment(-matrix)

    # Create the mapping dictionary: {cluster_id: true_digit_label}
    mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Remap predictions
    new_preds = np.array([mapping[label] for label in y_pred])
    return new_preds


def run_confusion_matrix():
    """
    Main execution flow for generating the confusion matrix.
    """
    print(f"--- Generating Confusion Matrix for {CM_CONFIG['dataset_name']} ---")

    # 1. Load Data
    path = CM_CONFIG["dataset_path"]
    if not os.path.exists(path):
        print(f"[Error] Dataset not found at {path}")
        return

    print(f"Loading data from {path}...")
    X, y_true, _ = preprocess_single_arff(path, drop_class=False)

    # Ensure y is integer type for accurate indexing
    y_true = y_true.astype(int)

    # 2. Run Algorithm
    k = CM_CONFIG["n_clusters"]
    print(f"Running FEKM (K={k})...")
    fekm = KMeansFEKM(n_clusters=k, random_state=CM_CONFIG["random_state"])
    y_pred_raw = fekm.fit_predict(X)

    # 3. Map Clusters
    print("Mapping clusters to true labels (Hungarian Algorithm)...")
    y_pred_mapped = map_clusters_to_labels(y_true, y_pred_raw)

    # 4. Generate Plot
    cm = confusion_matrix(y_true, y_pred_mapped)

    plt.figure(figsize=CM_CONFIG["figsize"])
    sns.heatmap(cm, annot=True, fmt='d', cmap=CM_CONFIG["cmap"], cbar=False)

    plt.title(f"Confusion Matrix: FEKM vs Ground Truth ({CM_CONFIG['dataset_name']})", fontsize=14)
    plt.xlabel("Predicted Label (Mapped)", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Save
    out_dir = CM_CONFIG["output_directory"]
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, CM_CONFIG["output_filename"])

    plt.savefig(save_path, dpi=CM_CONFIG["dpi"])
    plt.close()

    print(f"Success! Plot saved to: {save_path}")


if __name__ == "__main__":
    run_confusion_matrix()
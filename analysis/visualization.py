"""
Visualization Pipeline for Work 3.

This script executes the visualization tasks defined in Section 1.2 of the
assignment. It orchestrates the following:
1. Loading and preprocessing datasets.
2. Dimensionality reduction using Custom PCA (Task 1.2.1) and t-SNE (Task 1.2.4).
3. Visual comparison of reconstruction quality (Task 1.2.1 Step 9).
4. Comparison of clustering results (FEKM) on original vs. reduced data (Task 1.2.3).

References
----------
[1] Work 3 Description, UB, 2025, "1.2 Methodology of the visualization analysis", pp. 5-6.
[2] Support Slides Session 4, Salamó, 2025, "Dimensionality reduction & visualization".
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import Custom Algorithms
from algorithms.pca import PCA
from algorithms.kmeansfekm import KMeansFEKM
from utils.parser import preprocess_single_arff

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

OUTPUT_DIR = "../plots"
VIZ_SAMPLES = 3000  # Downsampling limit for t-SNE performance


def plot_reconstruction(
        X_orig: np.ndarray,
        X_reconst: np.ndarray,
        title: str,
        filename: str
):
    """
    Plots Original vs Reconstructed data (Features 1 & 2).

    Satisfies Task 1.2.1 Step 9: "Reconstruct the data set... Additionally, plot the data set."

    Parameters
    ----------
    X_orig : np.ndarray
        Original dataset.
    X_reconst : np.ndarray
        Reconstructed dataset after PCA inverse transform.
    title : str
        Plot title.
    filename : str
        Output filename.
    """
    # Need at least 2 features to make a 2D scatter
    if X_orig.shape[1] < 2:
        print("  [Warning] < 2 features, skipping reconstruction plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1) Original data: feature 1 vs feature 2 [1]
    sns.scatterplot(
        x=X_orig[:, 0],
        y=X_orig[:, 1],
        ax=axes[0],
        alpha=0.5,
        s=15,
    )
    axes[0].set_title("Original Data (Feat 1 vs Feat 2)")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # 2) Reconstructed data
    sns.scatterplot(
        x=X_reconst[:, 0],
        y=X_reconst[:, 1],
        ax=axes[1],
        alpha=0.5,
        s=15,
    )
    axes[1].set_title("Reconstructed Data (from PCA)")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Saved reconstruction plot] {filename}")


def plot_side_by_side(
        X_pca: np.ndarray,
        X_tsne: np.ndarray,
        labels: np.ndarray,
        title: str,
        filename: str,
        dataset_name: str
):
    """
    Generates a side-by-side plot: PCA (Left) vs t-SNE (Right).

    Satisfies Task 1.2.1 Step 8 ("Plot the new subspace") and Task 1.2.4.

    Parameters
    ----------
    X_pca : np.ndarray
        Data projected onto first 2 principal components.
    X_tsne : np.ndarray
        Data projected via t-SNE.
    labels : np.ndarray
        Class labels or cluster assignments for coloring.
    title : str
        Main title for the figure.
    filename : str
        Output filename.
    dataset_name : str
        Name of dataset for subtitle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def plot_on_axis(ax, data, title_suffix):
        df_plot = pd.DataFrame(data, columns=['C1', 'C2'])
        df_plot['label'] = labels

        # Handle high cardinality (e.g. noise points) by switching palettes
        n_colors = len(np.unique(labels))
        palette = "viridis" if n_colors > 10 else "tab10"

        sns.scatterplot(
            data=df_plot, x='C1', y='C2', hue='label',
            palette=palette, s=15, alpha=0.6, ax=ax, legend="full"
        )
        ax.set_title(f"{dataset_name} - {title_suffix}")
        ax.grid(True, linestyle='--', alpha=0.3)

    # Plot PCA
    plot_on_axis(axes[0], X_pca, "Custom PCA")

    # Plot t-SNE
    plot_on_axis(axes[1], X_tsne, "Sklearn t-SNE")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"  [Saved] {filename}")


def run_visualizations():
    """
    Main execution loop for visualization tasks.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting Visualization Pipeline. Saving to '{OUTPUT_DIR}/'...\n")

    for ds_name, file_path in DATASETS_MAP.items():
        print(f"--- Processing {ds_name} ---")

        # 1. Load Data
        if not os.path.exists(file_path):
            print(f"  [Error] File not found: {file_path}")
            continue

        try:
            X, y, _ = preprocess_single_arff(file_path, drop_class=False)
        except Exception as e:
            print(f"  [Error] Processing {ds_name}: {e}")
            continue

        # 2. Downsample for Visualization Speed (Optional but recommended)
        if X.shape[0] > VIZ_SAMPLES:
            print(f"  Downsampling from {X.shape[0]} to {VIZ_SAMPLES} for t-SNE...")
            np.random.seed(42)
            indices = np.random.choice(X.shape[0], VIZ_SAMPLES, replace=False)
            X = X[indices]
            if y is not None:
                y = y[indices]

        # 3. Run Custom PCA (From Scratch)
        # Requirement: "Show this information in console" (verbose=True) [1]
        print(f"  Running Custom PCA (2 Components)...")
        pca = PCA(n_components=2, verbose=True)

        start = time.time()
        X_pca = pca.fit_transform(X)
        print(f"    PCA Done in {time.time() - start:.2f}s")

        # --- PCA Reconstruction Demo (Task 1.2.1 Step 9) ---
        print(f"  Performing PCA reconstruction for {ds_name}...")
        X_reconstructed = pca.inverse_transform(X_pca)

        plot_reconstruction(
            X, X_reconstructed,
            f"PCA Reconstruction Quality ({ds_name})",
            f"{ds_name}_03_Reconstruction.png",
        )

        # 4. Run t-SNE (Sklearn)
        # Task 1.2.4: Visualize using t-SNE [1]
        print(f"  Running t-SNE (2 Components)...")
        start = time.time()
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        X_tsne = tsne.fit_transform(X)
        print(f"    t-SNE Done in {time.time() - start:.2f}s")

        # 5. Plot 1: Ground Truth (Class Labels)
        if y is not None:
            plot_side_by_side(
                X_pca, X_tsne, y,
                f"Ground Truth Classes ({ds_name})",
                f"{ds_name}_01_GroundTruth.png",
                ds_name
            )

        # 6. Plot 2: Improved K-Means Clustering (Original Data)
        # Requirement: "visualize... result of... k-Means... WITHOUT dimensionality reduction" [1]
        k = len(np.unique(y)) if y is not None else 3
        print(f"  Running FEKM Clustering on Original Data (k={k})...")

        fekm = KMeansFEKM(n_clusters=k, random_state=42)
        labels_fekm = fekm.fit_predict(X)

        plot_side_by_side(
            X_pca, X_tsne, labels_fekm,
            f"FEKM Result (Original Data)",
            f"{ds_name}_02_Clustering_Original.png",
            ds_name
        )

        # 7. Plot 3: Improved K-Means Clustering (Reduced Data)
        # Requirement: "visualize... result of... k-Means... WITH dimensionality reduction" [1]
        print(f"  Running FEKM Clustering on PCA-Reduced Data (k={k})...")

        # Clustering on the 2D PCA result
        fekm_reduced = KMeansFEKM(n_clusters=k, random_state=42)
        labels_fekm_reduced = fekm_reduced.fit_predict(X_pca)

        plot_side_by_side(
            X_pca, X_tsne, labels_fekm_reduced,
            f"FEKM Result (PCA Reduced Data)",
            f"{ds_name}_04_Clustering_Reduced.png",
            ds_name
        )

        print(f"  Done with {ds_name}.\n")


if __name__ == "__main__":
    run_visualizations()
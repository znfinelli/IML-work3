import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Import your Custom Algorithms
from algorithms.pca import PCA
from algorithms.kmeansfekm import KMeansFEKM  # Using FEKM as the "Improved" algorithm
from utils.parser import preprocess_single_arff

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# We use the same datasets, but we will downsample them for t-SNE speed
DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

OUTPUT_DIR = "../plots"
VIZ_SAMPLES = 3000  # Max samples for t-SNE (Safety limit for speed)


def plot_reconstruction(X_orig, X_reconst, title, filename):
    """
    Plot Original vs Reconstructed data on feature 1 & 2.
    This explicitly demonstrates PCA reconstruction quality (Task 1.2.1 Step 9).
    """
    # Need at least 2 features to make a 2D scatter
    if X_orig.shape[1] < 2:
        print("  [Warning] < 2 features, skipping reconstruction plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1) Original data: feature 1 vs feature 2
    # This satisfies Task 1.2.1 Step 2 (Plot original dataset)
    sns.scatterplot(
        x=X_orig[:, 0],
        y=X_orig[:, 1],
        ax=axes[0],
        alpha=0.5,
        s=15,
    )
    axes[0].set_title("Original Data (feat 1 vs feat 2)")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # 2) Reconstructed data: feature 1 vs feature 2
    # This satisfies Task 1.2.1 Step 9 (Reconstruct and plot)
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


def plot_side_by_side(X_pca, X_tsne, labels, title, filename, dataset_name):
    """
    Generates a side-by-side plot: PCA (Left) vs t-SNE (Right).
    Satisfies Task 1.2.1 Step 8 (Plot new subspace) and Section 1.2.2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Helper to plot on a specific axis
    def plot_on_axis(ax, data, title_suffix):
        df_plot = pd.DataFrame(data, columns=['C1', 'C2'])
        df_plot['label'] = labels

        # Check if we have too many categories (e.g. noise)
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

    # Save
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"  [Saved] {filename}")


def run_visualizations():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting Visualization Pipeline. Saving to '{OUTPUT_DIR}/'...\n")

    for ds_name, file_path in DATASETS_MAP.items():
        print(f"--- Processing {ds_name} ---")

        # 1. Load Data
        try:
            X, y, _ = preprocess_single_arff(file_path, drop_class=False)
        except Exception as e:
            print(f"Skipping {ds_name}: {e}")
            continue

        # 2. Downsample for Visualization Speed
        if X.shape[0] > VIZ_SAMPLES:
            print(f"  Downsampling from {X.shape[0]} to {VIZ_SAMPLES} for t-SNE...")
            np.random.seed(42)  # Deterministic for report
            indices = np.random.choice(X.shape[0], VIZ_SAMPLES, replace=False)
            X = X[indices]
            if y is not None:
                y = y[indices]

        # 3. Run Custom PCA (From Scratch)
        # Task: Print Covariance/Eigenvalues to console [PDF 1.2.1]
        print(f"  Running Custom PCA (2 Components)...")
        # Ensure your algorithms/pca.py supports verbose=True to print Steps 4, 5, 6
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
        # ----------------------------------------------------

        # 4. Run t-SNE (Sklearn)
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
        # Satisfies: "visualize... improved version of the k-Means... WITHOUT dimensionality reduction" [PDF 130]
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
        # Satisfies: "visualize... improved version of the k-Means... WITH dimensionality reduction" [PDF 130]
        print(f"  Running FEKM Clustering on PCA-Reduced Data (k={k})...")

        # We perform clustering on the 2D PCA result (X_pca)
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
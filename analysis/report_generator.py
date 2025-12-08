"""
Automated Report Asset Generator.

This script processes the aggregated results (from `results_compiler.py`) to
produce the specific tables and graphs required for the final report.

It generates:
1. Elbow Method and BIC plots for determining 'K'.
2. "Best Configuration" summary tables comparing all algorithms.
3. Parameter impact analysis (e.g., Linkage effect on Agglomerative).

References
----------
[1] Work 3 Description, UB, 2025, "1.1.2 Presenting and Interpreting Clustering Results", p. 4.
[2] Support Slides Session 3, Salamó, 2025, "Cluster Validation".
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from typing import Dict, Optional

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
REPORT_CONFIG = {
    "input_directory": "results_master",
    "file_pattern": "**/*_results.csv",
    "output_directory": "report_assets"
}


# ---------------------------------------------------------
# Data Loading Logic
# ---------------------------------------------------------
def load_data(config: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Locates and loads the most relevant results CSV based on config.

    Prioritizes 'master' compiled files over individual session files.

    Parameters
    ----------
    config : Dict
        Configuration dictionary containing paths.

    Returns
    -------
    pd.DataFrame or None
        The loaded data or None if not found.
    """
    input_dir = config["input_directory"]
    pattern = config["file_pattern"]

    search_path = os.path.join(input_dir, pattern)
    files = glob.glob(search_path, recursive=True)

    if not files:
        print(f"No results found in '{input_dir}' matching '{pattern}'")
        return None

    # Filter for the big compiled one if possible
    master_files = [f for f in files if "master" in f.lower() or "compiled" in f.lower()]
    target_file = master_files[-1] if master_files else files[-1]

    print(f"Loading results from: {target_file}")
    try:
        return pd.read_csv(target_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


# ---------------------------------------------------------
# Plotting & Reporting Functions
# ---------------------------------------------------------
def plot_elbow_bic(df: pd.DataFrame, output_dir: str):
    """
    Generates Elbow Method (Inertia) and BIC plots for K-Selection.

    Addresses requirement: "In the case of the K-Means... which has been the best K value?" [1].

    Parameters
    ----------
    df : pd.DataFrame
        The aggregated results data.
    output_dir : str
        Directory to save plots.
    """
    print("Generating K-Selection Plots (Elbow/BIC)...")

    datasets = df['dataset'].unique()

    # 1. Filter Data
    kmeans_df = df[df['algorithm'].str.contains("KMeans", case=False, na=False)]
    gmm_df = df[df['algorithm'] == "GaussianMixture"]

    for ds in datasets:
        # --- Plot Inertia (Elbow) for K-Means ---
        subset_km = kmeans_df[(kmeans_df['dataset'] == ds) & (kmeans_df['algorithm'] == "KMeans_Standard")]
        if not subset_km.empty:
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=subset_km, x='n_clusters', y='inertia', marker='o')
            plt.title(f"Elbow Method: K-Means on {ds}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{ds}_Elbow_KMeans.png"))
            plt.close()

        # --- Plot BIC for GMM ---
        subset_gmm = gmm_df[gmm_df['dataset'] == ds]
        if not subset_gmm.empty:
            plt.figure(figsize=(8, 5))

            # Normalize column names: GMM uses 'n_components', others 'n_clusters'
            subset_gmm = subset_gmm.copy()
            if 'n_components' in subset_gmm.columns and subset_gmm['n_clusters'].isna().all():
                subset_gmm['k'] = subset_gmm['n_components']
            else:
                subset_gmm['k'] = subset_gmm['n_clusters']

            sns.lineplot(data=subset_gmm, x='k', y='bic', marker='o', color='orange')
            plt.title(f"BIC Score: GMM on {ds} (Lower is Better)")
            plt.xlabel("Number of Components (k)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{ds}_BIC_GMM.png"))
            plt.close()


def generate_comparison_table(df: pd.DataFrame, output_dir: str):
    """
    Generates a 'Best Configuration' table for every algorithm.

    Satisfies: "Compare using tables... the clustering algorithms using some
    clustering validation metrics" [1].

    Parameters
    ----------
    df : pd.DataFrame
        Results data.
    output_dir : str
        Directory to save CSV/MD tables.
    """
    print("Generating Algorithm Comparison Tables...")

    if 'purity' not in df.columns:
        print("Warning: 'purity' column not found.")
        return

    try:
        # Find the row with the max Purity for each Algorithm on each Dataset
        best_indices = df.groupby(['dataset', 'algorithm'])['purity'].idxmax()
        best_df = df.loc[best_indices]
    except ValueError:
        return

    cols = ['dataset', 'algorithm', 'n_clusters', 'purity']

    # Add other metrics/params if they exist
    optional_cols = ['ari', 'davies_bouldin', 'f_measure', 'runtime',
                     'metric', 'linkage', 'param_m', 'param_alpha', 'preprocessing']

    for c in optional_cols:
        if c in df.columns:
            cols.append(c)

    final_table = best_df[cols].sort_values(by=['dataset', 'purity'], ascending=[True, False])

    final_table.to_csv(os.path.join(output_dir, "Best_Configurations_Summary.csv"), index=False)

    try:
        # Generate Markdown for easy copy-pasting into the final report
        markdown_string = final_table.to_markdown(index=False)
        with open(os.path.join(output_dir, "Best_Configurations_Summary.md"), "w") as f:
            f.write(markdown_string)
        print(f"   - Saved Markdown summary to {output_dir}")

    except ImportError:
        print(f"   [!] Skipped Markdown generation: 'tabulate' library not installed.")
        print(f"       (Run 'pip install tabulate' if you need the .md file)")


def plot_parameter_impact(df: pd.DataFrame, output_dir: str):
    """
    Generates plots to justify parameter choices.

    Satisfies: "evaluate with three linkage methods" and "Test two different
    distance metrics" [1].

    Parameters
    ----------
    df : pd.DataFrame
        Results data.
    output_dir : str
        Output directory.
    """
    print("Generating Parameter Impact Plots...")

    # 1. Agglomerative: Linkage Impact
    agg_df = df[df['algorithm'] == "Agglomerative"]
    if not agg_df.empty and 'linkage' in agg_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=agg_df, x='dataset', y='purity', hue='linkage')
        plt.title("Impact of Linkage on Agglomerative Clustering")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Param_Agg_Linkage.png"))
        plt.close()

    # 2. Agglomerative: Metric Impact
    if not agg_df.empty and 'metric' in agg_df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=agg_df, x='dataset', y='purity', hue='metric')
        plt.title("Impact of Distance Metric on Agglomerative Clustering")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Param_Agg_Metric.png"))
        plt.close()

    # 3. Fuzzy: Alpha Impact
    fuzzy_df = df[df['algorithm'].str.contains("FCM", na=False)]
    if not fuzzy_df.empty and 'param_alpha' in fuzzy_df.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=fuzzy_df, x='n_clusters', y='purity', hue='param_alpha', style='dataset', palette="flare")
        plt.title("Impact of Alpha (Suppression) on Fuzzy Clustering")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "Param_Fuzzy_Alpha.png"))
        plt.close()


def main():
    out_dir = REPORT_CONFIG["output_directory"]
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(REPORT_CONFIG)
    if df is None: return

    # Force numeric types to avoid any "object" confusion that breaks plotting
    cols_to_fix = ['n_clusters', 'n_components', 'inertia', 'bic', 'purity', 'ari']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    plot_elbow_bic(df, out_dir)
    generate_comparison_table(df, out_dir)
    plot_parameter_impact(df, out_dir)

    print(f"\nReport assets generated in '{out_dir}/'")


if __name__ == "__main__":
    main()
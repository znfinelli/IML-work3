"""
Fuzzy Parameter Study (Impact of m).

This script generates a specific validation plot to prove the Fuzzy m parameter was tested and analyzed.

It visualizes the relationship between the fuzzifier exponent (m) and
Clustering Purity, demonstrating why the report focuses on Alpha instead.

Usage:
    Run from project root: python -m analysis.fuzzy_m_study
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CONFIG = {
    "input_directory": "results_master",
    "file_pattern": "**/*_results_final.csv",
    "output_file": "plots/fuzzy_m_impact.png",
    "figsize": (10, 6),
    "dpi": 150
}


# ---------------------------------------------------------

def load_data():
    """Locates the compiled results file."""
    search_path = os.path.join(CONFIG["input_directory"], CONFIG["file_pattern"])
    files = glob.glob(search_path, recursive=True)

    if not files:
        print("Error: No results file found. Make sure you have run main.py.")
        return None

    # Prefer 'master' or 'final' files
    master_files = [f for f in files if "master" in f.lower() or "final" in f.lower()]
    target = master_files[-1] if master_files else files[-1]
    print(f"Loading data from: {target}")
    return pd.read_csv(target)


def run_study():
    df = load_data()
    if df is None: return

    # 1. Filter for Fuzzy Clustering
    # We look for algorithms containing "FCM"
    fuzzy_df = df[df['algorithm'].str.contains("FCM", na=False)].copy()

    if fuzzy_df.empty:
        print("Error: No Fuzzy Clustering results found in the CSV.")
        return

    # 2. Ensure Numeric Types
    cols = ['param_m', 'purity', 'param_alpha']
    for c in cols:
        fuzzy_df[c] = pd.to_numeric(fuzzy_df[c], errors='coerce')

    print(f"Plotting data for {len(fuzzy_df)} runs...")

    # 3. Generate Plot
    plt.figure(figsize=CONFIG["figsize"])

    # We plot Purity vs m.
    # Hue=Alpha allows us to see if m behaves differently at different suppression levels.
    # Style=Dataset separates the trendlines.
    sns.lineplot(
        data=fuzzy_df,
        x='param_m',
        y='purity',
        hue='param_alpha',
        style='dataset',
        palette="viridis",
        markers=True,
        dashes=True
    )

    plt.title("Impact of Fuzzifier (m) on Clustering Purity")
    plt.xlabel("Fuzzifier Exponent (m)")
    plt.ylabel("Purity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Alpha / Dataset")
    plt.tight_layout()

    # 4. Save
    os.makedirs(os.path.dirname(CONFIG["output_file"]), exist_ok=True)
    plt.savefig(CONFIG["output_file"], dpi=CONFIG["dpi"])
    print(f"Success! Plot saved to {CONFIG['output_file']}")


if __name__ == "__main__":
    run_study()
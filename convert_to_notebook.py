import nbformat as nbf

# Initialize a new notebook object
nb = nbf.v4.new_notebook()

# --------------------------------------------------------------------------
# CELL 1: Imports
# --------------------------------------------------------------------------
source_imports = r"""
import os
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Utilities
# NOTE: Ensure the 'utils' folder is uploaded to Colab files!
from utils.parser import preprocess_single_arff
from utils.clustering_metrics import compute_clustering_metrics

# Session 3 Algorithms/Reduction
# NOTE: Ensure the 'algorithms' folder is uploaded to Colab files!
from algorithms.pca import PCA
from algorithms.kmeansfekm import KMeansFEKM
from algorithms.kernel_kmeans import KernelKMeans
"""

# --------------------------------------------------------------------------
# CELL 2: Configuration
# --------------------------------------------------------------------------
source_config = r"""
# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RUN_CONFIG = {
    "datasets": {
        "pen-based": False,
        "adult": True,
        "mushroom": False
    },
    "algorithms": {
        "KMeansFEKM": False,
        "KernelKMeans": True
    }
}

DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

# PCA Configuration
# We test specific lower dimensions (2D, 3D for viz, 5D for mild reduction)
PCA_COMPONENTS_LIST = [2, 3, 5]

# Clustering Configuration
N_CLUSTERS_LIST = list(range(2, 11))
N_RUNS = 5  # Reduced runs for PCA experiments to save time (Deterministic PCA)
PARTIAL_SAVE_INTERVAL = 2
"""

# --------------------------------------------------------------------------
# CELL 3: Helpers
# --------------------------------------------------------------------------
source_helpers = r"""
# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def generate_task_list():
    tasks = []
    for ds_name, ds_enabled in RUN_CONFIG["datasets"].items():
        if not ds_enabled: continue

        # Iterate through target dimensions
        for n_comp in PCA_COMPONENTS_LIST:

            # 1. FEKM Tasks on Reduced Data
            if RUN_CONFIG["algorithms"]["KMeansFEKM"]:
                for k in N_CLUSTERS_LIST:
                    # FEKM is deterministic, so 1 run is enough per configuration
                    tasks.append({
                        "algorithm": "KMeans_FEKM",
                        "class": KMeansFEKM,
                        "dataset": ds_name,
                        "n_components": n_comp,
                        "n_clusters": k,
                        "metric": "euclidean",
                        "run_id": 0
                    })

            # 2. Kernel K-Means Tasks on Reduced Data
            if RUN_CONFIG["algorithms"]["KernelKMeans"]:
                for k in N_CLUSTERS_LIST:
                    # Intelligent Kernel KMeans is also deterministic
                    tasks.append({
                        "algorithm": "Kernel_KMeans",
                        "class": KernelKMeans,
                        "dataset": ds_name,
                        "n_components": n_comp,
                        "n_clusters": k,
                        "kernel": "rbf",  # Use RBF as standard for Kernel KM
                        "run_id": 0
                    })
    return tasks


def save_dataframe(data, folder, filename):
    if isinstance(data, pd.DataFrame):
        if data.empty: return
        df_to_save = data
    elif not data:
        return
    else:
        df_to_save = pd.DataFrame(data)

    os.makedirs(folder, exist_ok=True)
    df_to_save.to_csv(os.path.join(folder, filename), index=False)
"""

# --------------------------------------------------------------------------
# CELL 4: Main Execution
# --------------------------------------------------------------------------
source_main = r"""
# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
def main():
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results_session3/run_{session_id}"
    dirs = [
        base_dir,
        os.path.join(base_dir, "partial"),
        os.path.join(base_dir, "by_dataset"),
        os.path.join(base_dir, "by_algorithm")
    ]
    for d in dirs: os.makedirs(d, exist_ok=True)

    print(f"Session 3 Runner (PCA + Clustering) Started: {session_id}")
    print("Generating task list...")
    all_tasks = generate_task_list()

    if not all_tasks:
        print("No tasks configured.")
        return

    global_results = []
    current_ds_results = []
    current_ds_name = None

    # Cache for PCA-transformed data to avoid re-computing PCA for every K
    # Format: { (dataset_name, n_components): X_reduced }
    pca_cache = {}

    # Raw data cache
    X_orig, y_orig = None, None

    pbar = tqdm(all_tasks, unit="exp")

    for i, task in enumerate(pbar):
        ds_name = task["dataset"]
        n_comp = task["n_components"]
        algo_name = task["algorithm"]

        desc = f"{ds_name} | PCA({n_comp}D) -> {algo_name} | k={task['n_clusters']}"
        pbar.set_description(f"{desc:<60}")

        # 1. Load Original Data if needed
        if ds_name != current_ds_name:
            # Save previous dataset results
            if current_ds_name and current_ds_results:
                save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")
                current_ds_results = []
                pca_cache = {}  # Clear PCA cache for new dataset

            try:
                # Load fresh data
                X_orig, y_orig, _ = preprocess_single_arff(DATASETS_MAP[ds_name], drop_class=False)
                current_ds_name = ds_name
            except Exception as e:
                pbar.write(f"Error loading {ds_name}: {e}")
                continue

        # 2. Perform or Retrieve PCA Reduction
        # We check if we can actually reduce to n_components (e.g. can't reduce 4 dims to 5)
        if n_comp >= X_orig.shape[1]:
            # Skip invalid reduction requests
            continue

        if n_comp not in pca_cache:
            # Run Custom PCA
            try:
                pca = PCA(n_components=n_comp)
                X_reduced = pca.fit_transform(X_orig)
                pca_cache[n_comp] = X_reduced
            except Exception as e:
                pbar.write(f"PCA Failed for {ds_name} ({n_comp}D): {e}")
                continue
        else:
            X_reduced = pca_cache[n_comp]

        # 3. Run Clustering on Reduced Data
        start_time = time.perf_counter()
        res = {}

        try:
            # Prepare arguments
            kwargs = {
                "n_clusters": task["n_clusters"],
                "random_state": task["run_id"]
            }

            # Add algo-specific params
            if "kernel" in task:
                kwargs["kernel"] = task["kernel"]
            if "metric" in task:
                kwargs["metric"] = task["metric"]

            # Initialize and Fit
            model = task["class"](**kwargs)

            # fit_predict on X_reduced
            labels = model.fit_predict(X_reduced)

            # Record Results
            res = {
                "dataset": ds_name,
                "preprocessing": f"PCA_{n_comp}D",
                "n_features_orig": X_orig.shape[1],
                "n_features_reduced": n_comp,
                "algorithm": algo_name,
                "n_clusters": task["n_clusters"],
                "run_id": task["run_id"],
                "inertia": getattr(model, 'inertia_', 0),
                "runtime": time.perf_counter() - start_time,
                "labels": labels
            }

            # Compute Metrics using ORIGINAL Ground Truth (y_orig)
            if y_orig is not None:
                metrics = compute_clustering_metrics(X_reduced, y_orig, labels)
                res.update(metrics)
                del res["labels"]  # Remove labels to save space

            global_results.append(res)
            current_ds_results.append(res)

        except Exception as e:
            pbar.write(f"Clustering Failed: {task} Error: {e}")

        # Partial Save
        if (i + 1) % PARTIAL_SAVE_INTERVAL == 0:
            save_dataframe(global_results, dirs[1], f"partial_{session_id}.csv")

    # Final Saves
    if current_ds_results:
        save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")

    if global_results:
        df_final = pd.DataFrame(global_results)
        df_final.to_csv(os.path.join(base_dir, "session3_final_results.csv"), index=False)

        for algo in df_final['algorithm'].unique():
            safe_name = algo.replace(" ", "_")
            save_dataframe(df_final[df_final['algorithm'] == algo], dirs[3], f"{safe_name}.csv")

        print(f"\nSession 3 Complete. Results in {base_dir}")

if __name__ == "__main__":
    main()
"""

# Assemble the notebook
nb['cells'] = [
    nbf.v4.new_code_cell(source_imports),
    nbf.v4.new_code_cell(source_config),
    nbf.v4.new_code_cell(source_helpers),
    nbf.v4.new_code_cell(source_main)
]

# Write to file
with open('session3_runner.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Success! Created 'session3_runner.ipynb'. Upload this file to Colab.")
"""
Master Execution Script for Work 3.

This script serves as the primary entry point for the project. It orchestrates
the entire experimental pipeline across all three required sessions:
1. Standard Clustering (Agglomerative, GMM) [1].
2. Improved Clustering (K-Means Variants, Fuzzy) [2].
3. Dimensionality Reduction (PCA) + Clustering [3].

It handles dataset loading, parameter grid generation, algorithm execution,
metric computation, and result consolidation.

References
----------
[1] Work 3 Description, UB, 2025, "1.1 Methodology of the clustering analysis".
[2] Work 3 Description, UB, 2025, "Tasks 4, 5, 6".
[3] Work 3 Description, UB, 2025, "1.2 Methodology of the visualization analysis".
"""

import os
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

# Utilities
from utils.parser import preprocess_single_arff
from utils.clustering_metrics import compute_clustering_metrics

# Algorithms
from algorithms.agg_clustering import run_agglomerative_once
from algorithms.gmm_clustering import run_gmm_once
from algorithms.kmeans import KMeans
from algorithms.kmeansfekm import KMeansFEKM
from algorithms.kernel_kmeans import KernelKMeans
from algorithms.fuzzy_c_means import FuzzyCMeans
from algorithms.pca import PCA

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RUN_CONFIG = {
    "datasets": {
        "pen-based": True,
        "adult": True,
        "mushroom": True
    },
    "algorithms": {
        "Agglomerative": True,  # Session 1: Standard Hierarchical
        "GMM": True,  # Session 1: Gaussian Mixtures
        "KMeans_Variants": True,  # Session 2: Standard, FEKM, Kernel
        "Fuzzy_Clustering": True,  # Session 2: Fuzzy C-Means (Std & Suppressed)
        "PCA_Clustering": True  # Session 3: PCA + Improved KMeans
    }
}

DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

# Global Parameters
N_CLUSTERS_LIST = list(range(2, 11))
N_RUNS = 10  # Number of repeats for stochastic algorithms

# Session 1 Params
S1_METRICS = ["euclidean", "manhattan", "cosine"]
GMM_INIT_PARAMS = ["kmeans", "random", "k-means++"]

# Session 2 Params
FUZZY_M = [1.5, 2.0, 2.5]

# Session 3 Params
PCA_COMPONENTS = [2, 3, 5]

PARTIAL_SAVE_INTERVAL = 10


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def generate_task_list() -> List[Dict[str, Any]]:
    """
    Generates the complete list of experimental tasks based on RUN_CONFIG.
    """
    tasks = []
    for ds_name, ds_enabled in RUN_CONFIG["datasets"].items():
        if not ds_enabled: continue

        # --- SESSION 1: Standard Algorithms ---
        if RUN_CONFIG["algorithms"]["Agglomerative"]:
            for k in N_CLUSTERS_LIST:
                for link in ["complete", "average", "single"]:
                    for metric in S1_METRICS:
                        # Validation: Ward requires Euclidean
                        if link == 'ward' and metric != 'euclidean': continue

                        tasks.append({
                            "session": "S1", "type": "agg", "dataset": ds_name,
                            "n_clusters": k, "linkage": link, "metric": metric
                        })

        if RUN_CONFIG["algorithms"]["GMM"]:
            for k in N_CLUSTERS_LIST:
                for init_p in GMM_INIT_PARAMS:
                    for seed in range(N_RUNS):
                        tasks.append({
                            "session": "S1", "type": "gmm", "dataset": ds_name,
                            "n_clusters": k, "init_params": init_p, "run_id": seed
                        })

        # --- SESSION 2: Improved & Fuzzy Algorithms ---
        if RUN_CONFIG["algorithms"]["KMeans_Variants"]:
            km_algos = [
                ("KMeans_Standard", KMeans),
                ("KMeans_FEKM", KMeansFEKM),
                ("Kernel_KMeans", KernelKMeans)
            ]
            for algo_name, AlgoClass in km_algos:
                for k in N_CLUSTERS_LIST:
                    # Logic for Metric/Kernel selection
                    if algo_name == "Kernel_KMeans":
                        current_metrics = ["rbf"]
                    else:
                        current_metrics = ["euclidean", "manhattan"]

                    for metric in current_metrics:
                        # Optimization: Skip redundant runs for deterministic algorithms
                        current_runs = 1 if algo_name in ["KMeans_FEKM", "Kernel_KMeans"] else N_RUNS

                        for seed in range(current_runs):
                            task = {
                                "session": "S2", "type": "kmeans", "class": AlgoClass,
                                "algo_name": algo_name, "dataset": ds_name,
                                "n_clusters": k, "run_id": seed
                            }
                            if algo_name == "Kernel_KMeans":
                                task["kernel"] = metric
                            else:
                                task["metric"] = metric
                            tasks.append(task)

        if RUN_CONFIG["algorithms"]["Fuzzy_Clustering"]:
            alphas = [1.0, 0.75, 0.5]
            for k in N_CLUSTERS_LIST:
                for m in FUZZY_M:
                    for alpha in alphas:
                        algo_name = "FCM_Standard" if alpha == 1.0 else f"FCM_Suppressed_{alpha}"
                        for seed in range(N_RUNS):
                            tasks.append({
                                "session": "S2", "type": "fuzzy", "dataset": ds_name,
                                "algo_name": algo_name, "n_clusters": k, "metric": "euclidean",
                                "m": m, "alpha": alpha, "run_id": seed
                            })

        # --- SESSION 3: PCA + Clustering ---
        if RUN_CONFIG["algorithms"]["PCA_Clustering"]:
            pca_algos = [
                ("KMeans_FEKM", KMeansFEKM),
                ("Kernel_KMeans", KernelKMeans)
            ]
            for n_comp in PCA_COMPONENTS:
                for algo_name, AlgoClass in pca_algos:
                    for k in N_CLUSTERS_LIST:
                        task = {
                            "session": "S3", "type": "kmeans", "class": AlgoClass,
                            "algo_name": algo_name, "dataset": ds_name,
                            "n_clusters": k, "run_id": 0,  # Deterministic
                            "pca_dim": n_comp  # Flag to trigger PCA reduction
                        }
                        if algo_name == "Kernel_KMeans":
                            task["kernel"] = "rbf"
                        else:
                            task["metric"] = "euclidean"
                        tasks.append(task)

    return tasks


def save_dataframe(data: Any, folder: str, filename: str):
    """Safely saves results to CSV."""
    if isinstance(data, pd.DataFrame):
        if data.empty: return
        df_to_save = data
    elif not data:
        return
    else:
        df_to_save = pd.DataFrame(data)

    os.makedirs(folder, exist_ok=True)
    df_to_save.to_csv(os.path.join(folder, filename), index=False)


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results_master/run_{session_id}"
    dirs = [base_dir, os.path.join(base_dir, "partial"),
            os.path.join(base_dir, "by_dataset"), os.path.join(base_dir, "by_algorithm")]
    for d in dirs: os.makedirs(d, exist_ok=True)

    print(f"MASTER Runner Started: {session_id}")
    all_tasks = generate_task_list()

    if not all_tasks:
        print("No tasks configured.")
        return

    global_results = []
    current_ds_results = []
    current_ds_name = None

    # Data Caches
    X_orig, y_orig = None, None
    pca_cache = {}  # { n_dim: X_reduced }

    pbar = tqdm(all_tasks, unit="exp")

    for i, task in enumerate(pbar):
        ds_name = task["dataset"]
        session = task["session"]
        algo = task.get('algo_name', task['type'])
        pbar.set_description(f"[{session}] {ds_name} | {algo} | k={task['n_clusters']}")

        # 1. Load Data
        if ds_name != current_ds_name:
            if current_ds_name and current_ds_results:
                save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")
                current_ds_results = []
                pca_cache = {}  # Clear PCA cache for new dataset
            try:
                X_orig, y_orig, _ = preprocess_single_arff(DATASETS_MAP[ds_name], drop_class=False)
                current_ds_name = ds_name
            except Exception as e:
                pbar.write(f"Error loading {ds_name}: {e}")
                continue

        # 2. Determine Input Matrix X (Original or Reduced)
        X_input = X_orig
        use_pca = "pca_dim" in task

        if use_pca:
            n_dim = task["pca_dim"]
            if n_dim >= X_orig.shape[1]: continue

            if n_dim not in pca_cache:
                try:
                    pca = PCA(n_components=n_dim)
                    pca_cache[n_dim] = pca.fit_transform(X_orig)
                except Exception as e:
                    pbar.write(f"PCA Error: {e}")
                    continue
            X_input = pca_cache[n_dim]

        # 3. Algorithm Execution
        start = time.perf_counter()
        res = {}
        try:
            if task["type"] == "agg":
                res = run_agglomerative_once(X_input, task["n_clusters"], task["metric"], task["linkage"], ds_name)

            elif task["type"] == "gmm":
                res = run_gmm_once(X_input, task["n_clusters"], task["init_params"], ds_name)
                res["run_id"] = task["run_id"]

            elif task["type"] == "kmeans":
                kwargs = {"n_clusters": task["n_clusters"], "random_state": task["run_id"]}
                if "kernel" in task:
                    kwargs["kernel"] = task["kernel"]
                elif "metric" in task:
                    kwargs["metric"] = task["metric"]

                model = task["class"](**kwargs)
                labels = model.fit_predict(X_input)

                res = {
                    "dataset": ds_name, "algorithm": task["algo_name"], "n_clusters": task["n_clusters"],
                    "run_id": task["run_id"], "inertia": getattr(model, 'inertia_', 0), "labels": labels
                }
                if "metric" in task: res["metric"] = task["metric"]
                if "kernel" in task: res["kernel"] = task["kernel"]

            elif task["type"] == "fuzzy":
                fcm = FuzzyCMeans(
                    n_clusters=task["n_clusters"], m=task["m"],
                    alpha=task["alpha"], random_state=task["run_id"]
                )
                labels = fcm.fit_predict(X_input)
                res = {
                    "dataset": ds_name, "algorithm": task["algo_name"], "n_clusters": task["n_clusters"],
                    "metric": "euclidean", "param_m": task["m"], "param_alpha": task["alpha"],
                    "run_id": task["run_id"], "labels": labels
                }

            # 4. Result Consolidation
            res["session"] = session
            if use_pca:
                res["preprocessing"] = f"PCA_{task['pca_dim']}D"
            else:
                res["preprocessing"] = "Original"

            res["runtime"] = time.perf_counter() - start

            # 5. Validation Metrics
            # Note: We always compute metrics against original Y labels
            # DBI is calculated using X_input to reflect cluster quality in the CURRENT space
            if y_orig is not None and "labels" in res:
                res.update(compute_clustering_metrics(X_input, y_orig, res["labels"]))
                del res["labels"]

            global_results.append(res)
            current_ds_results.append(res)

        except Exception as e:
            pbar.write(f"Failed: {task} - {e}")

        # Checkpoint
        if (i + 1) % PARTIAL_SAVE_INTERVAL == 0:
            save_dataframe(global_results, dirs[1], f"partial_{session_id}.csv")

    # Final Saves
    if current_ds_results:
        save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")

    if global_results:
        df = pd.DataFrame(global_results)
        df.to_csv(os.path.join(base_dir, "master_results_final.csv"), index=False)
        print(f"\nMaster Run Complete. Data saved in {base_dir}")


if __name__ == "__main__":
    main()
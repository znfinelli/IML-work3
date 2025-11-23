import os
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

# Import local implementations
from parser import preprocess_single_arff
from clustering_metrics import compute_clustering_metrics

# Session 1 Algorithms
from agg_clustering import run_agglomerative_once
from gmm_clustering import run_gmm_once

# Session 2 Algorithms
from kmeans import KMeans
from kmeanspp import KMeansPP
# from kmeans_improved_2 import KMeansImproved2
from fuzzy_c_means import FuzzyCMeans

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Tip: Edit this dictionary to run in parallel
RUN_CONFIG = {
    "datasets": {
        "pen-based": True,
        "adult": False,
        "mushroom": False
    },
    "algorithms": {
        "Agglomerative": False,
        "GMM": False,
        "KMeans_Variants": True,
        "Fuzzy_Clustering": False
    }
}

DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

N_CLUSTERS_LIST = list(range(2, 11))
METRICS = ["euclidean", "manhattan"]
FUZZY_M = [1.5, 2.0, 2.5]
N_RUNS = 10
PARTIAL_SAVE_INTERVAL = 2


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def generate_task_list():
    """
    Generates a flat list of all experiments to be executed.
    """
    tasks = []

    for ds_name, ds_enabled in RUN_CONFIG["datasets"].items():
        if not ds_enabled: continue

        # 1. Agglomerative Tasks
        if RUN_CONFIG["algorithms"]["Agglomerative"]:
            for k in N_CLUSTERS_LIST:
                for link in ["complete", "average", "single"]:
                    tasks.append({
                        "type": "agg",
                        "dataset": ds_name,
                        "n_clusters": k,
                        "linkage": link,
                        "metric": "euclidean"
                    })

        # 2. GMM Tasks
        if RUN_CONFIG["algorithms"]["GMM"]:
            for k in N_CLUSTERS_LIST:
                for seed in range(N_RUNS):
                    tasks.append({
                        "type": "gmm",
                        "dataset": ds_name,
                        "n_clusters": k,
                        "run_id": seed
                    })

        # 3. K-Means Variants Tasks
        if RUN_CONFIG["algorithms"]["KMeans_Variants"]:
            km_algos = [
                ("KMeans_Standard", KMeans),
                ("KMeans_PP", KMeansPP),
                # ("KMeans_Improved_2", KMeansImproved2)
            ]
            for algo_name, AlgoClass in km_algos:
                for k in N_CLUSTERS_LIST:
                    for metric in METRICS:
                        current_runs = N_RUNS
                        for seed in range(current_runs):
                            tasks.append({
                                "type": "kmeans",
                                "class": AlgoClass,
                                "algo_name": algo_name,
                                "dataset": ds_name,
                                "n_clusters": k,
                                "metric": metric,
                                "run_id": seed
                            })

        # 4. Fuzzy Clustering Tasks
        if RUN_CONFIG["algorithms"]["Fuzzy_Clustering"]:
            alphas = [1.0, 0.7]
            for k in N_CLUSTERS_LIST:
                for m in FUZZY_M:
                    for alpha in alphas:
                        algo_name = "FCM_Standard" if alpha == 1.0 else f"FCM_Suppressed_{alpha}"
                        for seed in range(N_RUNS):
                            tasks.append({
                                "type": "fuzzy",
                                "dataset": ds_name,
                                "algo_name": algo_name,
                                "n_clusters": k,
                                "m": m,
                                "alpha": alpha,
                                "run_id": seed
                            })
    return tasks


def save_dataframe(data, folder, filename):
    """Helper to save list of dicts to CSV."""
    if not data:
        return
    filepath = os.path.join(folder, filename)
    pd.DataFrame(data).to_csv(filepath, index=False)


# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
def main():
    # Create unique session ID based on time to prevent overwrites
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Identify what is running for the folder name (e.g., "adult_only" or "all")
    active_datasets = [d for d, enabled in RUN_CONFIG["datasets"].items() if enabled]
    run_label = "_".join(active_datasets) if len(active_datasets) < 3 else "full_run"

    base_dir = f"results_session2/run_{run_label}_{session_id}"
    dirs = [
        base_dir,
        os.path.join(base_dir, "partial"),
        os.path.join(base_dir, "by_dataset"),
        os.path.join(base_dir, "by_algorithm")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"Starting Session: {session_id}")
    print(f"Results will be saved to: {base_dir}")

    print("Generating task list...")
    all_tasks = generate_task_list()
    total_tasks = len(all_tasks)
    print(f"Total experiments to run: {total_tasks}")

    if total_tasks == 0:
        print("No tasks selected in RUN_CONFIG. Exiting.")
        return

    global_results = []
    current_ds_results = []

    current_ds_name = None
    X, y = None, None

    pbar = tqdm(all_tasks, unit="exp")

    for i, task in enumerate(pbar):
        ds_name = task["dataset"]

        desc_str = f"{ds_name} | {task.get('algo_name', task['type'])} | k={task['n_clusters']}"
        pbar.set_description(f"{desc_str:<45}")

        # Smart Data Loading & Dataset Switching Logic
        if ds_name != current_ds_name:
            if current_ds_name is not None and current_ds_results:
                save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")
                current_ds_results = []

            try:
                X, y, _ = preprocess_single_arff(DATASETS_MAP[ds_name], drop_class=False)
                current_ds_name = ds_name
            except Exception as e:
                pbar.write(f"Error loading {ds_name}: {e}")
                continue

        # Execute Task
        start_time = time.perf_counter()
        res = {}

        try:
            if task["type"] == "agg":
                res = run_agglomerative_once(
                    X, task["n_clusters"], "euclidean", task["linkage"], ds_name
                )

            elif task["type"] == "gmm":
                res = run_gmm_once(X, task["n_clusters"], "kmeans", ds_name)
                res["run_id"] = task["run_id"]

            elif task["type"] == "kmeans":
                model = task["class"](
                    n_clusters=task["n_clusters"],
                    metric=task["metric"],
                    random_state=task["run_id"]
                )
                labels = model.fit_predict(X)
                res = {
                    "dataset": ds_name,
                    "algorithm": task["algo_name"],
                    "n_clusters": task["n_clusters"],
                    "metric": task["metric"],
                    "run_id": task["run_id"],
                    "inertia": getattr(model, 'inertia_', 0),
                    "labels": labels
                }

            elif task["type"] == "fuzzy":
                fcm = FuzzyCMeans(
                    n_clusters=task["n_clusters"],
                    m=task["m"],
                    alpha=task["alpha"],
                    random_state=task["run_id"]
                )
                labels = fcm.fit_predict(X)
                res = {
                    "dataset": ds_name,
                    "algorithm": task["algo_name"],
                    "n_clusters": task["n_clusters"],
                    "metric": "euclidean",
                    "param_m": task["m"],
                    "param_alpha": task["alpha"],
                    "run_id": task["run_id"],
                    "labels": labels
                }

            # Metrics
            res["runtime"] = time.perf_counter() - start_time

            if y is not None and "labels" in res:
                metrics = compute_clustering_metrics(X, y, res["labels"])
                res.update(metrics)
                del res["labels"]

            global_results.append(res)
            current_ds_results.append(res)

        except Exception as e:
            pbar.write(f"Task failed: {task} Error: {e}")

        # Partial Save (Unique filename using timestamp)
        if "run_id" in task and (task["run_id"] + 1) % PARTIAL_SAVE_INTERVAL == 0:
            save_dataframe(global_results, dirs[1], f"partial_results_{session_id}.csv")

    # Final Saves
    if current_ds_results:
        save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")

    if global_results:
        df_final = pd.DataFrame(global_results)

        # Master File
        df_final.to_csv(os.path.join(dirs[0], "clustering_results_final.csv"), index=False)

        # By Algorithm
        algorithms = df_final['algorithm'].unique()
        for algo in algorithms:
            safe_name = algo.replace(" ", "_").replace(".", "-")
            algo_df = df_final[df_final['algorithm'] == algo]
            save_dataframe(algo_df.to_dict('records'), dirs[3], f"{safe_name}.csv")

        print(f"\nExperiment execution complete.")
        print(f"Total experiments: {len(df_final)}")
        print(f"Results saved to: {base_dir}")
    else:
        print("\nNo results generated. Please check configuration.")


if __name__ == "__main__":
    main()
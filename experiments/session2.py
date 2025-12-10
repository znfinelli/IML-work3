"""
Experiment Runner for Session 2 (K-Means Variants & Fuzzy Clustering).

This script orchestrates the experiments for the "Own Code" algorithms:
1. Standard K-Means (Lloyd's Algorithm) [1]
2. Far Efficient K-Means (FEKM) [2]
3. Intelligent Kernel K-Means [3]
4. Fuzzy C-Means (Standard & Suppressed) [4]

It handles parameter grid generation, execution, metric computation, and
result serialization. It includes optimizations to skip redundant runs for
deterministic algorithms (FEKM, Kernel K-Means).

References
----------
[1] Work 3 Description, UB, 2025, "Task 4: Implement your own K-Means".
[2] Work 3 Description, UB, 2025, "Task 5: Improved K-Means (Mishra et al.)".
[3] Work 3 Description, UB, 2025, "Task 5: Improved K-Means (Handhayani et al.)".
[4] Work 3 Description, UB, 2025, "Task 6: Fuzzy Clustering".
"""

import os
import time
import datetime
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Type

# Utilities
from utils.parser import preprocess_single_arff
from utils.clustering_metrics import compute_clustering_metrics

# Session 2 Algorithms (Custom Implementations)
from algorithms.kmeans import KMeans
from algorithms.kmeansfekm import KMeansFEKM
from algorithms.kernel_kmeans import KernelKMeans
from algorithms.fuzzy_c_means import FuzzyCMeans

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
        "KMeans_Variants": True,
        "Fuzzy_Clustering": True
    }
}

DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

# Global Parameters
N_CLUSTERS_LIST = list(range(2, 11))

# Metrics for Standard K-Means
METRICS = ["euclidean", "manhattan"]

# Fuzzy Parameters
FUZZY_M = [1.5, 2.0, 2.5]

N_RUNS = 10  # For stochastic algorithms (Standard KMeans, Fuzzy)
PARTIAL_SAVE_INTERVAL = 10


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def generate_task_list() -> List[Dict[str, Any]]:
    """
    Generates the grid of experiment configurations.
    """
    tasks = []
    for ds_name, ds_enabled in RUN_CONFIG["datasets"].items():
        if not ds_enabled: continue

        # 3. K-Means Variants Tasks [1], [2], [3]
        if RUN_CONFIG["algorithms"]["KMeans_Variants"]:
            km_algos = [
                ("KMeans_Standard", KMeans),
                ("KMeans_FEKM", KMeansFEKM),
                ("Kernel_KMeans", KernelKMeans)
            ]
            for algo_name, AlgoClass in km_algos:
                for k in N_CLUSTERS_LIST:

                    # Logic to handle different parameters for Kernel vs Standard
                    if algo_name == "Kernel_KMeans":
                        # Kernel K-Means uses 'kernel' parameter, not 'metric'
                        current_metrics = ["rbf"]
                    else:
                        # Standard/FEKM use Euclidean/Manhattan
                        current_metrics = METRICS

                    for metric in current_metrics:
                        # Optimization: FEKM and Kernel (Intelligent) are deterministic.
                        # Running them 10 times produces the exact same result, so we run once.
                        if algo_name in ["KMeans_FEKM", "Kernel_KMeans"]:
                            current_runs = 1
                        else:
                            current_runs = N_RUNS

                        for seed in range(current_runs):
                            task = {
                                "type": "kmeans",
                                "class": AlgoClass,
                                "algo_name": algo_name,
                                "dataset": ds_name,
                                "n_clusters": k,
                                "run_id": seed
                            }

                            # Assign the correct parameter name
                            if algo_name == "Kernel_KMeans":
                                task["kernel"] = metric
                            else:
                                task["metric"] = metric

                            tasks.append(task)

        # 4. Fuzzy Clustering Tasks [4]
        if RUN_CONFIG["algorithms"]["Fuzzy_Clustering"]:
            # alpha=1.0 is Standard Bezdek, alpha<1.0 is Suppressed Fan et al.
            alphas = [1.0, 0.75, 0.5]
            for k in N_CLUSTERS_LIST:
                for m in FUZZY_M:
                    for alpha in alphas:
                        algo_name = "FCM_Standard" if alpha == 1.0 else f"FCM_Suppressed_{alpha}"

                        # Fuzzy clustering uses random initialization (U matrix), so we run N_RUNS
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


def save_dataframe(data: Any, folder: str, filename: str):
    """
    Safely saves a list of dicts or DataFrame to CSV.
    """
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
# Main Execution Loop
# ---------------------------------------------------------
def main():
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results_session2/run_{session_id}"
    dirs = [
        base_dir,
        os.path.join(base_dir, "partial"),
        os.path.join(base_dir, "by_dataset"),
        os.path.join(base_dir, "by_algorithm")
    ]
    for d in dirs: os.makedirs(d, exist_ok=True)

    print(f"Session 2 Runner Started: {session_id}")
    print("Generating task list...")
    all_tasks = generate_task_list()

    if not all_tasks:
        print("No tasks configured.")
        return

    print(f"Total tasks scheduled: {len(all_tasks)}")

    global_results = []
    current_ds_results = []
    current_ds_name = None
    X, y = None, None

    pbar = tqdm(all_tasks, unit="exp")

    for i, task in enumerate(pbar):
        ds_name = task["dataset"]
        desc = f"{ds_name} | {task.get('algo_name')} | k={task['n_clusters']}"
        pbar.set_description(f"{desc:<45}")

        # Lazy Loading
        if ds_name != current_ds_name:
            if current_ds_name and current_ds_results:
                save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")
                current_ds_results = []

            try:
                X, y, _ = preprocess_single_arff(DATASETS_MAP[ds_name], drop_class=False)
                current_ds_name = ds_name
            except Exception as e:
                pbar.write(f"Error loading {ds_name}: {e}")
                continue

        start_time = time.perf_counter()
        res = {}

        try:
            if task["type"] == "kmeans":
                # Prepare arguments dynamically based on class signature
                kwargs = {
                    "n_clusters": task["n_clusters"],
                    "random_state": task["run_id"]
                }

                if "kernel" in task:
                    kwargs["kernel"] = task["kernel"]
                elif "metric" in task:
                    kwargs["metric"] = task["metric"]

                # Initialize and run
                model = task["class"](**kwargs)
                labels = model.fit_predict(X)

                res = {
                    "dataset": ds_name,
                    "algorithm": task["algo_name"],
                    "n_clusters": task["n_clusters"],
                    "run_id": task["run_id"],
                    "inertia": getattr(model, 'inertia_', 0),
                    "labels": labels
                }
                # Log specific parameters for analysis
                if "metric" in task: res["metric"] = task["metric"]
                if "kernel" in task: res["kernel"] = task["kernel"]

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
                    "metric": "euclidean",  # FCM standard is L2
                    "param_m": task["m"],
                    "param_alpha": task["alpha"],
                    "run_id": task["run_id"],
                    "labels": labels
                }

            # Metrics Computation
            res["runtime"] = time.perf_counter() - start_time
            if y is not None and "labels" in res:
                res.update(compute_clustering_metrics(X, y, res["labels"]))
                del res["labels"]

            global_results.append(res)
            current_ds_results.append(res)

        except Exception as e:
            pbar.write(f"Task failed: {task} Error: {e}")

        # Checkpoint
        if (i + 1) % PARTIAL_SAVE_INTERVAL == 0:
            save_dataframe(global_results, dirs[1], f"partial_{session_id}.csv")

    # Final Save
    if current_ds_results:
        save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")

    if global_results:
        df_final = pd.DataFrame(global_results)
        df_final.to_csv(os.path.join(base_dir, "session2_final_results_compiled.csv"), index=False)

        for algo in df_final['algorithm'].unique():
            safe_name = algo.replace(" ", "_")
            save_dataframe(df_final[df_final['algorithm'] == algo], dirs[3], f"{safe_name}.csv")

        print(f"\nSession 2 Complete. Results in {base_dir}")


if __name__ == "__main__":
    main()
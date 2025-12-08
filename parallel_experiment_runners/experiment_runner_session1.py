import os
import time
import datetime
import pandas as pd
from tqdm import tqdm

# Utilities
from utils.parser import preprocess_single_arff
from utils.clustering_metrics import compute_clustering_metrics

# Session 1 Algorithms
from algorithms.agg_clustering import run_agglomerative_once
from algorithms.gmm_clustering import run_gmm_once

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
        "Agglomerative": True,
        "GMM": True
    }
}

DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

# Global Parameters
N_CLUSTERS_LIST = list(range(2, 11))

# Distances
METRICS = ["euclidean", "manhattan", "cosine"]

# Initialization methods
GMM_INIT_PARAMS = ["kmeans", "random", "k-means++", "random_from_data"]

N_RUNS = 10
PARTIAL_SAVE_INTERVAL = 2


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def generate_task_list():
    tasks = []
    for ds_name, ds_enabled in RUN_CONFIG["datasets"].items():
        if not ds_enabled: continue

        # 1. Agglomerative Tasks
        if RUN_CONFIG["algorithms"]["Agglomerative"]:
            for k in N_CLUSTERS_LIST:
                for link in ["complete", "average", "single"]:
                    for metric in METRICS:
                        tasks.append({
                            "type": "agg",
                            "dataset": ds_name,
                            "n_clusters": k,
                            "linkage": link,
                            "metric": metric
                        })

        # 2. GMM Tasks
        if RUN_CONFIG["algorithms"]["GMM"]:
            for k in N_CLUSTERS_LIST:
                for init_p in GMM_INIT_PARAMS:
                    # We perform N_RUNS for each configuration
                    for seed in range(N_RUNS):
                        tasks.append({
                            "type": "gmm",
                            "dataset": ds_name,
                            "n_clusters": k,
                            "init_params": init_p,
                            "run_id": seed
                        })
    return tasks


def save_dataframe(data, folder, filename):
    # Check if the input is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return
        df_to_save = data
    # Check if the input is an empty list/dictionary
    elif not data:
        return
    # If it's not a DataFrame and not empty, assume it's a list of records
    else:
        df_to_save = pd.DataFrame(data)

    # Ensure the directory exists before attempting to save
    os.makedirs(folder, exist_ok=True)

    # Save the file
    df_to_save.to_csv(os.path.join(folder, filename), index=False)


# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
def main():
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results_session1/run_{session_id}"
    dirs = [
        base_dir,
        os.path.join(base_dir, "partial"),
        os.path.join(base_dir, "by_dataset"),
        os.path.join(base_dir, "by_algorithm")
    ]
    for d in dirs: os.makedirs(d, exist_ok=True)

    print(f"Session 1 Runner Started: {session_id}")
    print("Generating task list...")
    all_tasks = generate_task_list()

    if not all_tasks:
        print("No tasks configured.")
        return

    global_results = []
    current_ds_results = []
    current_ds_name = None
    X, y = None, None

    pbar = tqdm(all_tasks, unit="exp")

    for i, task in enumerate(pbar):
        ds_name = task["dataset"]
        desc = f"{ds_name} | {task.get('type')} | k={task['n_clusters']}"
        pbar.set_description(f"{desc:<45}")

        # Load Data
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
            if task["type"] == "agg":
                res = run_agglomerative_once(
                    X, task["n_clusters"], task["metric"], task["linkage"], ds_name
                )
            elif task["type"] == "gmm":
                res = run_gmm_once(X, task["n_clusters"], task["init_params"], ds_name)
                res["run_id"] = task["run_id"]

            # Metrics
            res["runtime"] = time.perf_counter() - start_time
            if y is not None and "labels" in res:
                res.update(compute_clustering_metrics(X, y, res["labels"]))
                del res["labels"]

            global_results.append(res)
            current_ds_results.append(res)

        except Exception as e:
            pbar.write(f"Task failed: {task} Error: {e}")

        # Partial Save
        if (i + 1) % PARTIAL_SAVE_INTERVAL == 0:
            save_dataframe(global_results, dirs[1], f"partial_{session_id}.csv")

    # Final Saves
    if current_ds_results:
        save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")

    if global_results:
        df_final = pd.DataFrame(global_results)
        df_final.to_csv(os.path.join(base_dir, "session1_final_results_compiled.csv"), index=False)

        for algo in df_final['algorithm'].unique():
            safe_name = algo.replace(" ", "_")
            save_dataframe(df_final[df_final['algorithm'] == algo], dirs[3], f"{safe_name}.csv")

        print(f"\nSession 1 Complete. Results in {base_dir}")


if __name__ == "__main__":
    main()
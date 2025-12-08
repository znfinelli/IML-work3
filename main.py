import os
import time
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

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
        "Agglomerative": True,  # Session 1
        "GMM": True,  # Session 1
        "KMeans_Variants": True,  # Session 2
        "Fuzzy_Clustering": True,  # Session 2
        "PCA_Clustering": True  # Session 3
    }
}

DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult": "datasets/adult.arff",
    "mushroom": "datasets/mushroom.arff",
}

# Global Parameters
N_CLUSTERS_LIST = list(range(2, 11))
N_RUNS = 10

# Session 1 Params
S1_METRICS = ["euclidean", "manhattan", "cosine"]
GMM_INIT_PARAMS = ["kmeans", "random", "k-means++", "random_from_data"]

# Session 2 Params
FUZZY_M = [1.5, 2.0, 2.5]

# Session 3 Params
PCA_COMPONENTS = [2, 3, 5]

PARTIAL_SAVE_INTERVAL = 10


# ---------------------------------------------------------
# HELPER & MAIN
# ---------------------------------------------------------
def generate_task_list():
    tasks = []
    for ds_name, ds_enabled in RUN_CONFIG["datasets"].items():
        if not ds_enabled: continue

        # --- SESSION 1 ---
        if RUN_CONFIG["algorithms"]["Agglomerative"]:
            for k in N_CLUSTERS_LIST:
                for link in ["complete", "average", "single"]:
                    for metric in S1_METRICS:
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

        # --- SESSION 2 ---
        if RUN_CONFIG["algorithms"]["KMeans_Variants"]:
            km_algos = [
                ("KMeans_Standard", KMeans),
                ("KMeans_FEKM", KMeansFEKM),
                ("Kernel_KMeans", KernelKMeans)
            ]
            for algo_name, AlgoClass in km_algos:
                for k in N_CLUSTERS_LIST:
                    # Determine metric/kernel
                    if algo_name == "Kernel_KMeans":
                        current_metrics = ["rbf"]
                    else:
                        current_metrics = ["euclidean", "manhattan"]

                    for metric in current_metrics:
                        # Deterministic algorithms run once
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

        # --- SESSION 3 (PCA) ---
        if RUN_CONFIG["algorithms"]["PCA_Clustering"]:
            # We run FEKM and KernelKMeans on Reduced Data
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
                            "pca_dim": n_comp  # Flag to trigger PCA
                        }
                        # Set defaults for the algos
                        if algo_name == "Kernel_KMeans":
                            task["kernel"] = "rbf"
                        else:
                            task["metric"] = "euclidean"
                        tasks.append(task)

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


def main():
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results_master/run_{session_id}"
    dirs = [base_dir, os.path.join(base_dir, "partial"),
            os.path.join(base_dir, "by_dataset"), os.path.join(base_dir, "by_algorithm")]
    for d in dirs: os.makedirs(d, exist_ok=True)

    print(f"MASTER Runner Started: {session_id}")
    all_tasks = generate_task_list()

    global_results = []
    current_ds_results = []
    current_ds_name = None

    # Data Caches
    X_orig, y_orig = None, None
    pca_cache = {}  # Stores { n_dim: X_reduced }

    pbar = tqdm(all_tasks, unit="exp")

    for i, task in enumerate(pbar):
        ds_name = task["dataset"]
        session = task["session"]
        algo = task.get('algo_name', task['type'])
        pbar.set_description(f"[{session}] {ds_name} | {algo} | k={task['n_clusters']}")

        # 1. Load Data if Dataset Changed
        if ds_name != current_ds_name:
            if current_ds_name and current_ds_results:
                save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")
                current_ds_results = []
                pca_cache = {}  # Clear cache
            try:
                X_orig, y_orig, _ = preprocess_single_arff(DATASETS_MAP[ds_name], drop_class=False)
                current_ds_name = ds_name
            except Exception as e:
                pbar.write(f"Error loading {ds_name}: {e}")
                continue

        # 2. Determine Input Matrix X (Original or PCA-Reduced)
        X_input = X_orig
        use_pca = "pca_dim" in task

        if use_pca:
            n_dim = task["pca_dim"]
            if n_dim >= X_orig.shape[1]: continue  # Skip invalid reduction

            # Check cache
            if n_dim not in pca_cache:
                try:
                    pca = PCA(n_components=n_dim)
                    pca_cache[n_dim] = pca.fit_transform(X_orig)
                except Exception as e:
                    pbar.write(f"PCA Error: {e}")
                    continue
            X_input = pca_cache[n_dim]

        # 3. Run Algorithms
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
                fcm = FuzzyCMeans(n_clusters=task["n_clusters"], m=task["m"], alpha=task["alpha"],
                                  random_state=task["run_id"])
                labels = fcm.fit_predict(X_input)
                res = {
                    "dataset": ds_name, "algorithm": task["algo_name"], "n_clusters": task["n_clusters"],
                    "metric": "euclidean", "param_m": task["m"], "param_alpha": task["alpha"],
                    "run_id": task["run_id"], "labels": labels
                }

            # Add Metadata
            res["session"] = session
            if use_pca:
                res["preprocessing"] = f"PCA_{task['pca_dim']}D"
            else:
                res["preprocessing"] = "Original"

            res["runtime"] = time.perf_counter() - start

            # Metrics (Always compare against Original Ground Truth y_orig)
            if y_orig is not None and "labels" in res:
                res.update(compute_clustering_metrics(X_input, y_orig, res["labels"]))
                del res["labels"]

            global_results.append(res)
            current_ds_results.append(res)

        except Exception as e:
            pbar.write(f"Failed: {task} - {e}")

        if (i + 1) % PARTIAL_SAVE_INTERVAL == 0:
            save_dataframe(global_results, dirs[1], f"partial_{session_id}.csv")

    if current_ds_results: save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")
    if global_results:
        df = pd.DataFrame(global_results)
        df.to_csv(os.path.join(base_dir, "master_results_final.csv"), index=False)
        print(f"\nMaster Run Complete. Data saved in {base_dir}")


if __name__ == "__main__":
    main()
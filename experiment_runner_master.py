import os
import time
import datetime
import pandas as pd
from tqdm import tqdm

# Utilities
from utils.parser import preprocess_single_arff
from utils.clustering_metrics import compute_clustering_metrics

# Algorithms
from algorithms.agg_clustering import run_agglomerative_once
from algorithms.gmm_clustering import run_gmm_once
from algorithms.kmeans import KMeans
from algorithms.kmeanspp import KMeansPP
# from algorithms.kmeans_mishra import KMeansMishra # (If using Mishra)
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
        "Agglomerative": True,
        "GMM": True,
        "KMeans_Variants": True,
        "Fuzzy_Clustering": True
    }
}

DATASETS_MAP = {
    "pen-based": "datasets/pen-based.arff",
    "adult":     "datasets/adult.arff",
    "mushroom":  "datasets/mushroom.arff",
}

N_CLUSTERS_LIST = list(range(2, 11))
METRICS = ["euclidean", "manhattan"]
FUZZY_M = [1.5, 2.0, 2.5]
N_RUNS = 10
PARTIAL_SAVE_INTERVAL = 2

# ---------------------------------------------------------
# HELPER & MAIN
# ---------------------------------------------------------
def generate_task_list():
    tasks = []
    for ds_name, ds_enabled in RUN_CONFIG["datasets"].items():
        if not ds_enabled: continue
        
        if RUN_CONFIG["algorithms"]["Agglomerative"]:
            for k in N_CLUSTERS_LIST:
                for link in ["complete", "average", "single"]:
                    for metric in METRICS:
                        tasks.append({
                            "type": "agg", "dataset": ds_name, "n_clusters": k,
                            "linkage": link, "metric": metric
                        })

        if RUN_CONFIG["algorithms"]["GMM"]:
            for k in N_CLUSTERS_LIST:
                for seed in range(N_RUNS):
                    tasks.append({
                        "type": "gmm", "dataset": ds_name, "n_clusters": k,
                        "init_params": "kmeans", "run_id": seed
                    })

        if RUN_CONFIG["algorithms"]["KMeans_Variants"]:
            km_algos = [("KMeans_Standard", KMeans), ("KMeans_PP", KMeansPP)]
            # Add partner code here later
            for algo_name, AlgoClass in km_algos:
                for k in N_CLUSTERS_LIST:
                    for metric in METRICS:
                        for seed in range(N_RUNS):
                            tasks.append({
                                "type": "kmeans", "class": AlgoClass, "algo_name": algo_name,
                                "dataset": ds_name, "n_clusters": k, "metric": metric, "run_id": seed
                            })

        if RUN_CONFIG["algorithms"]["Fuzzy_Clustering"]:
            alphas = [1.0, 0.7]
            for k in N_CLUSTERS_LIST:
                for m in FUZZY_M:
                    for alpha in alphas:
                        algo_name = "FCM_Standard" if alpha == 1.0 else f"FCM_Suppressed_{alpha}"
                        for seed in range(N_RUNS):
                            tasks.append({
                                "type": "fuzzy", "dataset": ds_name, "algo_name": algo_name,
                                "n_clusters": k, "metric": "euclidean", "m": m, "alpha": alpha, "run_id": seed
                            })
    return tasks

def save_dataframe(data, folder, filename):
    if not data: return
    pd.DataFrame(data).to_csv(os.path.join(folder, filename), index=False)

def main():
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results_master/run_{session_id}"
    dirs = [base_dir, os.path.join(base_dir, "partial"), os.path.join(base_dir, "by_dataset"), os.path.join(base_dir, "by_algorithm")]
    for d in dirs: os.makedirs(d, exist_ok=True)
    
    print(f"MASTER Runner Started: {session_id}")
    all_tasks = generate_task_list()
    
    global_results = []
    current_ds_results = []
    current_ds_name = None
    X, y = None, None
    
    pbar = tqdm(all_tasks, unit="exp")
    
    for i, task in enumerate(pbar):
        ds_name = task["dataset"]
        algo = task.get('algo_name', task['type'])
        pbar.set_description(f"{ds_name} | {algo} | k={task['n_clusters']}")

        if ds_name != current_ds_name:
            if current_ds_name and current_ds_results:
                save_dataframe(current_ds_results, dirs[2], f"{current_ds_name}_results.csv")
                current_ds_results = []
            try:
                X, y, _ = preprocess_single_arff(DATASETS_MAP[ds_name], drop_class=False)
                current_ds_name = ds_name
            except Exception as e:
                pbar.write(f"Error: {e}")
                continue

        start = time.perf_counter()
        res = {}
        try:
            # Execution Logic
            if task["type"] == "agg":
                res = run_agglomerative_once(X, task["n_clusters"], task["metric"], task["linkage"], ds_name)
            elif task["type"] == "gmm":
                res = run_gmm_once(X, task["n_clusters"], task["init_params"], ds_name)
                res["run_id"] = task["run_id"]
            elif task["type"] == "kmeans":
                model = task["class"](n_clusters=task["n_clusters"], metric=task["metric"], random_state=task["run_id"])
                labels = model.fit_predict(X)
                res = {"dataset": ds_name, "algorithm": task["algo_name"], "n_clusters": task["n_clusters"],
                       "metric": task["metric"], "run_id": task["run_id"], "inertia": getattr(model, 'inertia_', 0), "labels": labels}
            elif task["type"] == "fuzzy":
                fcm = FuzzyCMeans(n_clusters=task["n_clusters"], m=task["m"], alpha=task["alpha"], random_state=task["run_id"])
                labels = fcm.fit_predict(X)
                res = {"dataset": ds_name, "algorithm": task["algo_name"], "n_clusters": task["n_clusters"],
                       "metric": "euclidean", "param_m": task["m"], "param_alpha": task["alpha"], "run_id": task["run_id"], "labels": labels}
            
            res["runtime"] = time.perf_counter() - start
            if y is not None and "labels" in res:
                res.update(compute_clustering_metrics(X, y, res["labels"]))
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
        for algo in df['algorithm'].unique():
            save_dataframe(df[df['algorithm'] == algo], dirs[3], f"{algo}.csv")
        print(f"\nMaster Run Complete. Data saved in {base_dir}")

if __name__ == "__main__":
    main()

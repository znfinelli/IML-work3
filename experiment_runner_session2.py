import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar library
from typing import List, Dict, Optional

# Import your local implementations
from parser import preprocess_single_arff
from clustering_metrics import compute_clustering_metrics

# Session 1 Algorithms (sklearn)
from agg_clustering import run_agglomerative_once
from gmm_clustering import run_gmm_once

# Session 2 Algorithms (Your Code)
from kmeans import KMeans
from kmeanspp import KMeansPP
# from kmeans_improved_2 import KMeansImproved2  # <--- UNCOMMENT THIS WHEN PARTNER IS READY
from fuzzy_c_means import FuzzyCMeans

def save_partial_results(results, filename):
    """Helper to save results during execution"""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    # No print here to avoid cluttering the progress bar output

def main():
    # ---------------------------------------------------------
    # 1. CONFIGURATION
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

    DATASETS = [
        {"name": "pen-based", "path": "datasets/pen-based.arff"},
        {"name": "adult",     "path": "datasets/adult.arff"},
        {"name": "mushroom",  "path": "datasets/mushroom.arff"},
    ]

    N_CLUSTERS_LIST = [2, 3, 4, 5]
    METRICS = ["euclidean", "manhattan"]
    FUZZY_M = [1.5, 2.0, 2.5]
    
    N_RUNS = 10 
    SAVE_INTERVAL = 2 # Save every 2 folds

    os.makedirs("results_session2", exist_ok=True)
    global_results = []
    partial_file = "results_session2/clustering_results_partial.csv"

    for ds_cfg in DATASETS:
        ds_name = ds_cfg["name"]
        
        if not RUN_CONFIG["datasets"].get(ds_name, False):
            continue
            
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {ds_name}")
        print(f"{'='*60}")
        
        # Load Data
        try:
            X, y, _ = preprocess_single_arff(ds_cfg["path"], drop_class=False)
            print(f"Data loaded: {X.shape} samples, {X.shape[1]} features")
        except FileNotFoundError:
            print(f"ERROR: File not found at {ds_cfg['path']}. Skipping.")
            continue

        # -----------------------------------------------------
        # 2. RUN AGGLOMERATIVE
        # -----------------------------------------------------
        if RUN_CONFIG["algorithms"]["Agglomerative"]:
            print("\n--- Algorithm: Agglomerative Clustering ---")
            combinations = [(k, link) for k in N_CLUSTERS_LIST for link in ["complete", "average", "single"]]
            
            # Deterministic = No folds, just iterating configs
            for k, linkage in tqdm(combinations, desc=f"Agglomerative ({ds_name})"):
                res = run_agglomerative_once(X, k, "euclidean", linkage, ds_name)
                if y is not None:
                    res.update(compute_clustering_metrics(X, y, res['labels']))
                del res['labels'] 
                global_results.append(res)
            
            save_partial_results(global_results, partial_file)

        # -----------------------------------------------------
        # 3. RUN GMM
        # -----------------------------------------------------
        if RUN_CONFIG["algorithms"]["GMM"]:
            print("\n--- Algorithm: Gaussian Mixture Models ---")
            
            # Loop over K values
            for k in N_CLUSTERS_LIST:
                # Progress bar for the Folds (Runs)
                pbar = tqdm(range(N_RUNS), desc=f"GMM (k={k})")
                
                for seed in pbar:
                    res = run_gmm_once(X, k, "kmeans", ds_name) 
                    if y is not None:
                        res.update(compute_clustering_metrics(X, y, res['labels']))
                    del res['labels']
                    res['run_id'] = seed
                    global_results.append(res)
                    
                    # Partial Save
                    if (seed + 1) % SAVE_INTERVAL == 0:
                        save_partial_results(global_results, partial_file)

        # -----------------------------------------------------
        # 4. RUN YOUR K-MEANS VARIANTS
        # -----------------------------------------------------
        if RUN_CONFIG["algorithms"]["KMeans_Variants"]:
            print("\n--- Algorithm: K-Means Variants ---")
            
            algorithms = [
                ("KMeans_Standard", KMeans),
                ("KMeans_PP",       KMeansPP),
                # ("KMeans_Improved_2", KMeansImproved2), 
            ]

            for name, AlgoClass in algorithms:
                for k in N_CLUSTERS_LIST:
                    for metric in METRICS:
                        current_runs = N_RUNS 
                        
                        # Progress bar for Folds
                        desc = f"{name} (k={k}, {metric})"
                        pbar = tqdm(range(current_runs), desc=desc)
                        
                        for seed in pbar:
                            start = time.perf_counter()
                            
                            model = AlgoClass(n_clusters=k, metric=metric, random_state=seed)
                            labels = model.fit_predict(X)
                            
                            runtime = time.perf_counter() - start
                            metrics = compute_clustering_metrics(X, y, labels) if y is not None else {}
                            
                            entry = {
                                "dataset": ds_name,
                                "algorithm": name,
                                "n_clusters": k,
                                "metric": metric,
                                "run_id": seed,
                                "runtime": runtime,
                                "inertia": getattr(model, 'inertia_', 0)
                            }
                            entry.update(metrics)
                            global_results.append(entry)
                            
                            if (seed + 1) % SAVE_INTERVAL == 0:
                                save_partial_results(global_results, partial_file)

        # -----------------------------------------------------
        # 5. RUN FUZZY CLUSTERING
        # -----------------------------------------------------
        if RUN_CONFIG["algorithms"]["Fuzzy_Clustering"]:
            print("\n--- Algorithm: Fuzzy C-Means ---")
            alphas = [1.0, 0.7] 
            
            # We flatten the loop slightly to make the progress bar more meaningful
            # Instead of nested loops for m/alpha inside K, we iterate combinations
            
            for k in N_CLUSTERS_LIST:
                for m in FUZZY_M:
                    for alpha in alphas:
                        algo_name = "FCM_Standard" if alpha == 1.0 else f"FCM_Suppressed_{alpha}"
                        desc = f"{algo_name} (k={k}, m={m})"
                        
                        pbar = tqdm(range(N_RUNS), desc=desc)
                        
                        for seed in pbar:
                            start = time.perf_counter()
                            
                            fcm = FuzzyCMeans(n_clusters=k, m=m, alpha=alpha, random_state=seed)
                            labels = fcm.fit_predict(X)
                            
                            runtime = time.perf_counter() - start
                            metrics = compute_clustering_metrics(X, y, labels) if y is not None else {}
                            
                            entry = {
                                "dataset": ds_name,
                                "algorithm": algo_name,
                                "n_clusters": k,
                                "metric": "euclidean",
                                "param_m": m,
                                "param_alpha": alpha,
                                "run_id": seed,
                                "runtime": runtime
                            }
                            entry.update(metrics)
                            global_results.append(entry)
                            
                            if (seed + 1) % SAVE_INTERVAL == 0:
                                save_partial_results(global_results, partial_file)

    # ---------------------------------------------------------
    # 6. FINAL SAVE
    # ---------------------------------------------------------
    if global_results:
        df = pd.DataFrame(global_results)
        final_file = "results_session2/clustering_results_final.csv"
        df.to_csv(final_file, index=False)
        print(f"\n SUCCESS! Final results saved to {final_file}")
        print(f"   Total Experiments Completed: {len(df)}")
    else:
        print("\n No results generated. Check your configuration.")

if __name__ == "__main__":
    main()
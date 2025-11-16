
import os
import time
from typing import List, Dict, Optional

import pandas as pd

from parser import preprocess_single_arff
from agg_clustering import run_agglomerative_once
from gmm_clustering import run_gmm_once
from clustering_metrics import compute_clustering_metrics


def main():
    # 🔧 Adjust these paths / class-column names to your actual files
    DATASETS = [
        {
            "name": "pen-based",
            "path": "datasets/pen-based.arff",
            "class_column": None,  # or e.g. "class". If labels are not in the last clomun, specify here
        },
        {
            "name": "adult",
            "path": "datasets/adult.arff",
            "class_column": None,
        },
        {
            "name": "iris",
            "path": "datasets/iris.arff",
            "class_column": None,
        },
    ]

    # Common configuration for ALL datasets (you can specialize later)
    agg_n_clusters = [2, 3, 4]
    agg_metrics = ["euclidean", "cosine"]
    agg_linkages = ["complete", "average", "single"]

    gmm_n_components = [2, 3, 4]
    gmm_inits = ["kmeans", "random_from_data"]

    jobs_per_dataset = (
        len(agg_n_clusters) * len(agg_metrics) * len(agg_linkages)
        + len(gmm_n_components) * len(gmm_inits)
    )
    total_jobs = jobs_per_dataset * len(DATASETS)

    print(f"Total datasets: {len(DATASETS)}")
    print(f"Experiments per dataset: {jobs_per_dataset}")
    print(f"Total experiments (all datasets): {total_jobs}\n")

    os.makedirs("results_session1", exist_ok=True)

    global_results: List[Dict] = []
    global_start = time.perf_counter()
    global_job_counter = 0

    # ================== MAIN LOOP OVER DATASETS ==================
    for ds_cfg in DATASETS:
        ds_name = ds_cfg["name"]
        arff_path = ds_cfg["path"]
        class_col: Optional[str] = ds_cfg["class_column"]

        print(f"\n=== Dataset: {ds_name} ===")
        print(f"Loading from: {arff_path}")

        # Load + preprocess. IMPORTANT: keep labels for metrics.
        X, y, info = preprocess_single_arff(
            filepath=arff_path,
            class_column=class_col,
            drop_class=False,
        )
        print(f"  -> X shape: {X.shape}, y shape: {None if y is None else y.shape}")

        dataset_results: List[Dict] = []

        # Build job list for THIS dataset
        local_jobs: List[Dict] = []

        for k in agg_n_clusters:
            for metric in agg_metrics:
                for linkage in agg_linkages:
                    local_jobs.append({
                        "type": "agglo",
                        "n_clusters": k,
                        "metric": metric,
                        "linkage": linkage,
                    })

        for k in gmm_n_components:
            for init in gmm_inits:
                local_jobs.append({
                    "type": "gmm",
                    "n_components": k,
                    "init_params": init,
                })

        print(f"  -> Experiments for {ds_name}: {len(local_jobs)}")

        # ---- Run all jobs for this dataset ----
        for local_idx, job in enumerate(local_jobs, start=1):
            global_job_counter += 1
            run_start = time.perf_counter()

            # Run algorithm
            if job["type"] == "agglo":
                res = run_agglomerative_once(
                    X=X,
                    n_clusters=job["n_clusters"],
                    metric=job["metric"],
                    linkage=job["linkage"],
                    dataset_name=ds_name,
                )
            else:  # GMM
                res = run_gmm_once(
                    X=X,
                    n_components=job["n_components"],
                    init_params=job["init_params"],
                    dataset_name=ds_name,
                )

            run_time = time.perf_counter() - run_start

            # Compute metrics if labels exist
            labels = res["labels"]
            if y is not None:
                metric_dict = compute_clustering_metrics(X, y, labels)
            else:
                metric_dict = {
                    "ari": None,
                    "purity": None,
                    "davies_bouldin": None,
                    "f_measure": None,
                }

            # Strip labels but keep everything else
            row = {k: v for k, v in res.items() if k != "labels"}
            row.update(metric_dict)

            dataset_results.append(row)
            global_results.append(row)

            # Global ETA
            elapsed_global = time.perf_counter() - global_start
            avg_per_job = elapsed_global / global_job_counter
            remaining_jobs = total_jobs - global_job_counter
            eta_sec = avg_per_job * remaining_jobs

            print(
                f"[{ds_name}] "
                f"[{local_idx}/{len(local_jobs)}] "
                f"{res['algorithm']} done in {run_time:.2f}s | "
                f"Global progress {global_job_counter}/{total_jobs} | "
                f"Global ETA ~ {eta_sec:.1f}s"
            )

        # ---- Save per-dataset CSV (ALL algorithms together) ----
        ds_df = pd.DataFrame(dataset_results)
        ds_out_path = os.path.join("results_session1", f"{ds_name}_session1.csv")
        ds_df.to_csv(ds_out_path, index=False)
        print(f"  -> Saved results for {ds_name} to: {ds_out_path}")

        # ---- Additionally: split BY ALGORITHM for this dataset ----
        for algo in ds_df["algorithm"].unique():
            algo_df = ds_df[ds_df["algorithm"] == algo].copy()

            # Drop irrelevant columns for readability
            if algo == "Agglomerative":
                drop_cols = ["n_components", "init_params", "bic", "avg_log_likelihood"]
            elif algo == "GaussianMixture":
                drop_cols = ["n_clusters", "metric", "linkage"]
            else:
                drop_cols = []

            for c in drop_cols:
                if c in algo_df.columns:
                    algo_df.drop(columns=c, inplace=True)

            algo_out_path = os.path.join(
                "results_session1",
                f"{ds_name}_{algo}_session1.csv"
            )
            algo_df.to_csv(algo_out_path, index=False)
            print(f"  -> Saved {algo} results for {ds_name} to: {algo_out_path}")


    # ================== AFTER ALL DATASETS ==================
    global_df = pd.DataFrame(global_results)
    global_out_path = os.path.join("results_session1", "ALL_DATASETS_session1.csv")
    global_df.to_csv(global_out_path, index=False)

    total_time = time.perf_counter() - global_start
    print(f"\n=== All Session 1 experiments finished in {total_time:.1f}s ===")
    print(f"Global CSV saved to: {global_out_path}")


if __name__ == "__main__":
    main() 
## Current Directory Map
```text
.
├── algorithms/                     # Core clustering algorithm implementations
│   ├── __init__.py                 # Package initialization
│   ├── agg_clustering.py           # Wrapper for Sklearn Agglomerative Clustering (float32 optimized)
│   ├── fuzzy_c_means.py            # Unified class for Standard (Bezdek) and Suppressed (Fan et al.) FCM
│   ├── gmm_clustering.py           # Wrapper for Sklearn Gaussian Mixture Models (Singularity safe)
│   ├── kernel_kmeans.py            # Improved K-Means: Intelligent Kernel K-Means (Handhayani et al.)
│   ├── kmeans.py                   # Vectorized implementation of Standard K-Means
│   ├── kmeansfekm.py               # Improved K-Means: Far Efficient K-Means (Mishra et al.)
│   └── pca.py                      # Custom Principal Component Analysis implementation
│
├── datasets/                       # Source data files (.arff format)
│   ├── adult.arff
│   ├── mushroom.arff
│   └── pen-based.arff
│
├── plots/                          # (Generated) Output directory for 2D/3D visualization plots
│
├── results_master/                 # (Generated) Combined output directory from main.py master run
│   └── run_[TIMESTAMP]/            # Unique folder per execution
│       ├── by_algorithm/           # Results aggregated by algorithm
│       ├── by_dataset/             # Results aggregated by dataset
│       └── partial/                # Checkpoint files
│
├── results_session1/               # (Generated) Session 1 results (Agglomerative / GMM)
│   └── run_[TIMESTAMP]/            # Unique folder per execution to support parallel runs
│       ├── by_algorithm/           # Results aggregated by algorithm
│       ├── by_dataset/             # Results aggregated by dataset
│       └── partial/                # Checkpoint files saved during execution
│
├── results_session2/               # (Generated) Session 2 results (K-Means Variants / Fuzzy)
│   └── run_[TIMESTAMP]/            # Unique folder per execution to support parallel runs
│       ├── by_algorithm/           # Results aggregated by algorithm
│       ├── by_dataset/             # Results aggregated by dataset
│       └── partial/                # Checkpoint files saved during execution
│
├── results_session3/               # (Generated) Session 3 results (PCA + Reduced Clustering)
│   └── run_[TIMESTAMP]/            # Unique folder per execution to support parallel runs
│       ├── by_algorithm/           # Results aggregated by algorithm
│       ├── by_dataset/             # Results aggregated by dataset
│       └── partial/                # Checkpoint files saved during execution
│
├── utils/                          # Data processing and validation utilities
│   ├── __init__.py                 # Package initialization
│   ├── clustering_metrics.py       # Validation metrics (Purity, F-Measure, ARI, Davies-Bouldin)
│   └── parser.py                   # Data loading, One-Hot Encoding, and memory optimization (float32)
│
├── experiment_runner_session1.py   # Individual runner for Session 1 (Agg/GMM)
├── experiment_runner_session2.py   # Individual runner for Session 2 (KMeans/Fuzzy)
├── experiment_runner_session3.py   # Individual runner for Session 3 (PCA reduction)
├── main.py                         # Master script executing all sessions sequentially
├── results_compiler.py             # Utility to merge distributed result CSVs into a master file
└── visualization.py                # Script for generating PCA vs t-SNE comparison plots
```

```text
.
├── main.py                         # Primary Entry Point (Master execution script)
├── README.md                       # Project documentation and execution instructions
├── requirements.txt                # List of Python dependencies
│
├── algorithms/                     # [SOURCE] Core clustering algorithm implementations
│   ├── __init__.py                 # Package initialization
│   ├── agg_clustering.py           # Wrapper for Sklearn Agglomerative (float32 optimized + caching)
│   ├── fuzzy_c_means.py            # Unified class for Standard (Bezdek) and Suppressed (Fan et al.) FCM
│   ├── gmm_clustering.py           # Wrapper for Sklearn GMM (Singularity safe for categorical data)
│   ├── kernel_kmeans.py            # Improved K-Means: Intelligent Kernel K-Means (float32 memory optimized)
│   ├── kmeans.py                   # Vectorized implementation of Standard K-Means
│   ├── kmeansfekm.py               # Improved K-Means: Far Efficient K-Means (Mishra et al.)
│   └── pca.py                      # Custom Principal Component Analysis implementation
│
├── analysis/                       # [SCRIPTS] Data Processing, Visualization & Reporting
│   ├── __init__.py                 # Package initialization
│   ├── report_generator.py         # Generates tables & graphs for the final report (Elbow, BIC, etc.)
│   ├── results_compiler.py         # Utility to merge distributed result CSVs into a master file
│   └── visualization.py            # Script for generating PCA vs t-SNE comparison plots
│
├── datasets/                       # Source data files (.arff format)
│   ├── adult.arff
│   ├── mushroom.arff
│   └── pen-based.arff
│
├── experiments/                    # [SCRIPTS] Individual Session Runners
│   ├── __init__.py                 # Package initialization
│   ├── session1.py                 # Runner for Session 1 (Agglomerative / GMM)
│   ├── session2.py                 # Runner for Session 2 (K-Means Variants / Fuzzy)
│   └── session3.py                 # Runner for Session 3 (PCA + Reduced Clustering)
│
├── plots/                          # (Generated) Output directory for 2D/3D visualization plots
│
├── report_assets/                  # (Generated) Output directory for report tables and graphs
│
├── results_master/                 # (Generated) Combined output directory from main.py master run
│   └── run_[TIMESTAMP]/            # Unique folder per execution
│       ├── by_algorithm/           # Results aggregated by algorithm
│       ├── by_dataset/             # Results aggregated by dataset
│       └── partial/                # Checkpoint files
│
├── results_session1/               # (Generated) Session 1 outputs (Agglomerative / GMM)
├── results_session2/               # (Generated) Session 2 outputs (K-Means / Fuzzy)
├── results_session3/               # (Generated) Session 3 outputs (PCA experiments)
│
└── utils/                          # [SOURCE] Data processing and validation utilities
    ├── __init__.py                 # Package initialization
    ├── clustering_metrics.py       # Validation metrics (Purity, F-Measure, ARI, Davies-Bouldin)
    └── parser.py                   # Data loading, One-Hot Encoding, and float32 memory optimization
```

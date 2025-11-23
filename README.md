### Current Updates (Zoë):
- Modified parser:
    - For handling '?' instead of only NaN type
    - Uses one-hot encoding instead of label encoder for higher accuracy; however computational time is increased
- Created `fuzzy_c_means.py` with both standard fuzzy and suppressed fuzzy algos, and test more than one alpha
- Renamed `kmeans++.py` -> `kmeanspp.py` for import ability in experiment runner
- Created `experiment_runner_session2.py` with run_configs for running algos/datasets one after another or in parallel in copies of runner
    - Set datasets/algos to true/false to run in parallel
    - Runs k = 2-10 to satisfy complexities of 10 classes in pen-based
- Modifed `experiment_runner_session1.py` logic to support parallel runs and added run_configs
    - Also modified results saving logic
- Added `experiment_runner_master.py` for cohesive project submission that combines all experiment runners into a true main script
- Deleted session 1 & 2 results so no confusion for reruns with one-hot encoder parser; if needed in future, these results are saved in `zoë-work branch`
- Added `results_compiler.py` for handling compiling of results after parallel runs


## Current Directory Map
```
.
├── algorithms/                     # Core clustering algorithm implementations
│   ├── __init__.py                 # Package initialization
│   ├── agg_clustering.py           # Wrapper for Sklearn Agglomerative Clustering
│   ├── fuzzy_c_means.py            # Unified class for Standard (Bezdek) and Suppressed (Fan et al.) FCM
│   ├── gmm_clustering.py           # Wrapper for Sklearn Gaussian Mixture Models
│   ├── kmeans.py                   # Vectorized implementation of Standard K-Means
│   └── kmeanspp.py                 # Improved K-Means (Probabilistic initialization via K-Means++)
│
├── datasets/                       # Source data files (.arff format)
│   ├── adult.arff
│   ├── mushroom.arff
│   └── pen-based.arff
│
├── results_session1/               # (Generated) Output directory for experiment logs and CSVs
│   └── run_[TIMESTAMP]/            # Unique folder per execution to support parallel runs
│       ├── by_algorithm/           # Results aggregated by algorithm
│       ├── by_dataset/             # Results aggregated by dataset
│       └── partial/                # Checkpoint files saved during execution
│
├── results_session2/               # (Generated) Output directory for experiment logs and CSVs
│   └── run_[TIMESTAMP]/            # Unique folder per execution to support parallel runs
│       ├── by_algorithm/           # Results aggregated by algorithm
│       ├── by_dataset/             # Results aggregated by dataset
│       └── partial/                # Checkpoint files saved during execution
│
├── utils/                          # Data processing and validation utilities
│   ├── __init__.py                 # Package initialization
│   ├── clustering_metrics.py       # Validation metrics (Purity, F-Measure, ARI, Davies-Bouldin)
│   └── parser.py                   # Data loading, One-Hot Encoding, and value imputation
│
├── experiment_runner_global.py     # Master script to execute all configured experiments
├── experiment_runner_session1.py   # Execution script for Session 1 algorithms (Agglomerative/GMM)
├── experiment_runner_session2.py   # Execution script for Session 2 algorithms (K-Means/Fuzzy)
└── results_compiler.py             # Utility to merge distributed/parallel-run result CSVs into a master file
```

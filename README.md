current updates:
- modified parser for handing '?' instead of only NaN type
- created fuzzy_c_means.py with both standard fuzzy and suppressed fuzzy algos, and test more than one alpha
- renamed kmeans++.py -> kmeanspp.py for import ability in experiment runner
- created experiment_runner_session2.py with run_configs for running algos/datasets one after another or in parallel in copies of runner
  • set datasets/algos to true/false

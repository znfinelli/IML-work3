"""
Results Compiler Utility.

This script aggregates the distributed results from different experiment sessions
(Session 1: Standard, Session 2: K-Means/Fuzzy, Session 3: PCA) into a single
master CSV file. This unified dataset is required to generate the comparative
tables and graphs for the final report.

References
----------
[1] Work 3 Description, UB, 2025, "1.1.2 Presenting and Interpreting Clustering Results", p. 4.
[2] Work 3 Description, UB, 2025, "1.3 Work to deliver", p. 6.
"""

import os
import pandas as pd
import glob
from typing import Dict, List, Any

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
COMPILATION_CONFIG = {
    # Directories to scan for session outputs
    "input_directories": [
        "results_session1",
        "results_session2",
        "results_session3",
    ],

    # Pattern to identify final result files
    "target_pattern": "*_final_results_compiled.csv",

    # Output destination
    "output_directory": "results_master",
    "output_filename": "master_results_final.csv"
}


# ---------------------------------------------------------
# Compilation Logic
# ---------------------------------------------------------
def compile_results(config: Dict[str, Any]):
    """
    Scans specified directories for result CSV files and merges them.

    This function recursively searches for files matching the target pattern,
    concatenates them into a pandas DataFrame, handles missing columns
    (e.g., 'preprocessing' for non-PCA runs), and saves the master file.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing 'input_directories', 'target_pattern',
        'output_directory', and 'output_filename'.
    """
    input_dirs = config["input_directories"]
    target_pattern = config["target_pattern"]

    # Logic to handle output path
    output_dir = config.get("output_directory")
    filename = config["output_filename"]

    # Determine full output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
    else:
        output_path = filename

    all_files = []
    print("Starting result compilation process...")

    # 1. Collect all matching files recursively
    for directory in input_dirs:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue

        print(f"Scanning directory: {directory}")

        # Use 'recursive=True' to look inside run_[TIMESTAMP] subfolders
        search_path = os.path.join(directory, "**", target_pattern)
        found_files = glob.glob(search_path, recursive=True)

        all_files.extend(found_files)
        print(f"  Found {len(found_files)} files matching '{target_pattern}'")

    if not all_files:
        print("No result files found. Please check your input directories.")
        return

    print(f"\nTotal files identified for merging: {len(all_files)}")

    # 2. Read and Concatenate
    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    if not df_list:
        print("Could not read any valid dataframes. Exiting.")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # 3. Standardization
    # Session 1 & 2 runs might not have 'preprocessing' column (implied 'Original')
    # Session 3 (PCA) will have 'preprocessing' = 'PCA'
    if 'preprocessing' in master_df.columns:
        master_df['preprocessing'] = master_df['preprocessing'].fillna('Original')
    else:
        master_df['preprocessing'] = 'Original'

    # 4. Save Compiled File
    try:
        master_df.to_csv(output_path, index=False)
        print(f"\nCompilation successful.")
        print(f"Merged {len(master_df)} total experiment rows.")
        print(f"Master file saved to: {output_path}")
    except Exception as e:
        print(f"Error saving master file: {e}")
        return

    # 5. Summary
    print("\n--- Dataset and Algorithm Summary ---")
    if 'dataset' in master_df.columns and 'algorithm' in master_df.columns:
        # Group by preprocessing as well to verify PCA vs Original counts
        if 'preprocessing' in master_df.columns:
            summary = master_df.groupby(['dataset', 'algorithm', 'preprocessing']).size()
        else:
            summary = master_df.groupby(['dataset', 'algorithm']).size()
        print(summary)
    else:
        print("Columns 'dataset' or 'algorithm' not found. Skipping summary.")


if __name__ == "__main__":
    compile_results(COMPILATION_CONFIG)
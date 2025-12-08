import os
import pandas as pd
import glob
from typing import List, Dict

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
COMPILATION_CONFIG = {
    # Directories to scan
    "input_directories": [
        "results_session1",
        "results_session2",
        "results_session3",
    ],

    # PATTERN MATCHING:
    "target_pattern": "*_final_results_compiled.csv",

    # --- OUTPUT CONFIGURATION ---
    # Option 1: Set to a folder name (e.g., "results_master") to create a new folder.
    # Option 2: Set to "" or None to save directly in the root (iml_work3 directory).
    "output_directory": "results_master",

    "output_filename": "master_final_results.csv"
}


# ---------------------------------------------------------
# Compilation Logic
# ---------------------------------------------------------
def compile_results(config: Dict):
    """
    Scans specified directories for result CSV files matching a pattern
    and merges them into a single master file.
    """
    input_dirs = config["input_directories"]
    target_pattern = config["target_pattern"]

    # Logic to handle output path
    output_dir = config.get("output_directory")
    filename = config["output_filename"]

    # Determine full output path
    if output_dir:
        # If a folder is specified, create it and join path
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
    else:
        # If empty/None, save to current root directory
        output_path = filename

    all_files = []
    print("Starting result compilation process...")

    # 1. Collect all matching files recursively
    for directory in input_dirs:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue

        print(f"Scanning directory: {directory}")

        # Use 'recursive=True' to look inside run_2025... subfolders
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

    # Fill empty preprocessing slots (from Session 1 & 2) with "Original"
    if 'preprocessing' in master_df.columns:
        master_df['preprocessing'] = master_df['preprocessing'].fillna('Original')
    else:
        # Create column if it doesn't exist at all (e.g. only session 1/2 loaded)
        master_df['preprocessing'] = 'Original'

    # 3. Save Compiled File using the calculated output_path
    try:
        master_df.to_csv(output_path, index=False)
        print(f"\nCompilation successful.")
        print(f"Merged {len(master_df)} total experiment rows.")
        print(f"Master file saved to: {output_path}")
    except Exception as e:
        print(f"Error saving master file: {e}")
        return

    # 4. Summary
    print("\n--- Dataset and Algorithm Summary ---")
    if 'dataset' in master_df.columns and 'algorithm' in master_df.columns:
        summary = master_df.groupby(['dataset', 'algorithm']).size()
        print(summary)
    else:
        print("Columns 'dataset' or 'algorithm' not found. Skipping summary.")


if __name__ == "__main__":
    compile_results(COMPILATION_CONFIG)
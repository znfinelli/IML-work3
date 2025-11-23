import os
import pandas as pd
import glob
from typing import List, Dict

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
COMPILATION_CONFIG = {
    # Directories to scan (add 'results_master' if you use that too)
    "input_directories": [
        "results_session1",
        "results_session2",
        "results_master"
    ],

    # PATTERN MATCHING UPDATE:
    # We use the '*' wildcard to find ANY file ending in "_final_results.csv".
    # This captures "session1_final_results.csv", "session2_...", etc.
    "target_pattern": "*_final_results.csv",

    # The output path for the combined file
    "output_filename": "compiled_results_master.csv"
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
    output_file = config["output_filename"]

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

            # Optional: Add a column to track source session (useful for debugging)
            # df['source_file'] = os.path.basename(filename)

            df_list.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    if not df_list:
        print("Could not read any valid dataframes. Exiting.")
        return

    master_df = pd.concat(df_list, ignore_index=True)

    # 3. Save Compiled File
    try:
        master_df.to_csv(output_file, index=False)
        print(f"\n✅ Compilation successful.")
        print(f"Merged {len(master_df)} total experiment rows.")
        print(f"Master file saved to: {output_file}")
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
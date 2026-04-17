import pandas as pd
from pathlib import Path
import sys
import os

# Ensure parent directory is in sys.path for module imports
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

# Custom preprocessing function that handles None / NaN cleanup
def fix_none(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by removing rows with a target value of -1.

    This function inspects the last column (assumed to be the target variable)
    and drops any rows where the value is -1. This is used to filter out
    samples that were not labeled as 'S' or 'R' in the metadata.

    Args:
        df (pd.DataFrame): The combined DataFrame (features + target).

    Returns:
        pd.DataFrame: The cleaned DataFrame with invalid target rows removed.
    """
    print("    - Applying 'fix_none': Dropping rows where target is -1...")
    
    # Identify the last column, which is our target variable.
    target_column_index = -1
    
    # Count how many rows will be dropped.
    initial_rows = len(df)
    rows_to_drop = df[df.iloc[:, target_column_index] == -1]
    num_to_drop = len(rows_to_drop)
    
    # Filter the DataFrame, keeping only rows where the target is NOT -1.
    cleaned_df = df[df.iloc[:, target_column_index] != -1].copy()
    final_rows = len(cleaned_df)
    
    if num_to_drop > 0:
        print(f"      - Dropped {num_to_drop} rows. Shape changed from {initial_rows} to {final_rows} rows.")
    else:
        print("      - No rows with target = -1 found. Data is clean.")
        
    return cleaned_df


def process_data_folders():
    """
    Main pipeline to:
    1. Iterate over TB drug-resistance data folders
    2. Load genotype feature matrices and metadata
    3. Extract and encode resistance labels
    4. Clean the combined dataset
    5. Save standardized X.csv and Y.csv for benchmarking
    """

    # --------------------------------------------------------
    # Root directory containing multiple subfolders.
    # Each subfolder corresponds to a drug or dataset and
    # is expected to contain:
    #   - <folder>_data_10.csv
    #   - <folder>_metadata.csv
    # --------------------------------------------------------
    data_root = Path('/data/public-data/TB_data_alldrugs/')

    # --------------------------------------------------------
    # Output directory where processed datasets will be saved.
    # One subfolder per drug/dataset will be created.
    # --------------------------------------------------------
    output_root = Path('/data/users/saish/Research/TB/TB-Bench/data/_WholeGenome_Variants')

    print(f"Starting data processing from source: '{data_root.resolve()}'")
    print(f"Output will be saved in: '{output_root.resolve()}'")

    # --------------------------------------------------------
    # Sanity check: ensure the source directory exists
    # --------------------------------------------------------
    if not data_root.exists() or not data_root.is_dir():
        print(f"\n[ERROR] Source directory not found at '{data_root.resolve()}'")
        print("Please ensure the path is correct.")
        sys.exit(1)

    # Ensure output root exists
    output_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Iterate through each dataset folder inside data_root
    # --------------------------------------------------------

    for folder_path in data_root.iterdir():

        # Process only directories (ignore stray files)
        if folder_path.is_dir():
            folder_name = folder_path.name
            print(f"\nProcessing folder: '{folder_name}'")

            try:
                # ------------------------------------------------
                # Create corresponding output directory
                # ------------------------------------------------
                local_output_dir = output_root / folder_name
                local_output_dir.mkdir(exist_ok=True)
                print(f"    - Created/verified local directory: '{local_output_dir}'")

                # ------------------------------------------------
                # Expected input files inside each folder
                # ------------------------------------------------
                data_file = folder_path / f"{folder_name}_data_10.csv"
                meta_file = folder_path / f"{folder_name}_metadata.csv"

                # Skip folder if required files are missing
                if not data_file.exists() or not meta_file.exists():
                    print("    - [WARNING] Required data/metadata files not found. Skipping.")
                    continue

                # ------------------------------------------------
                # Load genotype feature matrix
                #
                # index_col=0:
                #   - Drops the first column (often isolate IDs)
                #   - Prevents it from being treated as a feature
                # ------------------------------------------------
                print(f"    - Loading '{data_file.name}'...")
                X = pd.read_csv(data_file, index_col=0)

                # ------------------------------------------------
                # Load metadata file
                # ------------------------------------------------
                print(f"    - Loading '{meta_file.name}'...")
                metadata = pd.read_csv(meta_file)

                # ------------------------------------------------
                # Ensure resistance column exists
                # ------------------------------------------------
                if 'S/R' not in metadata.columns:
                    print("    - [WARNING] 'S/R' column not found in metadata. Skipping.")
                    continue

                # ------------------------------------------------
                # Process resistance labels:
                #   S (Sensitive) -> 0
                #   R (Resistant) -> 1
                #   Missing/other -> -1
                #
                # NOTE:
                # This assumes 'S/R' has already been numerically
                # encoded upstream or cleaned before this stage.
                # ------------------------------------------------
                target_col = metadata['S/R'].fillna(-1).astype(int)
                target_col.name = 'target'  # Clean header for Y.csv

                # Optional: quick sanity check
                print(metadata.head())

                # ------------------------------------------------
                # Combine features (X) and target (Y)
                #
                # reset_index(drop=True):
                #   - Ensures row-wise alignment
                #   - Avoids accidental joins on index values
                # ------------------------------------------------
                X.reset_index(drop=True, inplace=True)
                X_combined = pd.concat([X, target_col], axis=1)
                print(f"    - Combined data shape: {X_combined.shape}")

                # ------------------------------------------------
                # Custom cleaning step:
                #   - Handles None / NaN / invalid entries
                #   - Applies dataset-specific cleanup logic
                # ------------------------------------------------
                X_prime = fix_none(X_combined)

                # ------------------------------------------------
                # Separate cleaned features and target
                # ------------------------------------------------
                Y_final = X_prime.iloc[:, -1]
                X_final = X_prime.iloc[:, :-1]

                # ------------------------------------------------
                # Save final datasets
                # ------------------------------------------------
                x_output_path = local_output_dir / 'X.csv'
                y_output_path = local_output_dir / 'Y.csv'

                # Save features without index
                X_final.to_csv(x_output_path, index=False)

                # Save target with header "target"
                Y_final.to_csv(y_output_path, index=False, header=True)

                print("    - Successfully saved processed data:")
                print(f"      - Features (X): '{x_output_path}' {X_final.shape}")
                print(f"      - Target (Y):   '{y_output_path}' {Y_final.shape}")

            except Exception as e:
                # ------------------------------------------------
                # Fail gracefully: log error and continue
                # ------------------------------------------------
                print(f"    - [ERROR] An unexpected error occurred: {e}")
                continue

    print("\nProcessing complete.")


# ------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------
if __name__ == '__main__':
    process_data_folders()
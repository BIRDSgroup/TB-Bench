import os
import argparse
import importlib
import importlib.util
import sys
from datetime import datetime
import pandas as pd
import pdb
from models.tb_cnn_codebase import *

# ============================================================
# Attempt to import validation utilities.
#
# Expected functions:
#   - load_validation_function()
#   - test_model()
#
# If import fails, define safe fallbacks so the script
# doesn't crash immediately and provides informative errors.
# ============================================================
try:
    from validation import *
except ImportError:
    print("Error: Could not import from 'validation'.")

# ============================================================
# MODEL LOADING
# ============================================================
def load_models(model_names):
    """
    Dynamically loads model manager classes from ./models.

    For each model name, this function expects:
        - a Python module: models/<model_name>.py
        - a class named: <model_name>Manager inside that module

    Example:
        model_name = "RuleBased"
        module      = models.RuleBased
        class       = RuleBasedManager

    Args:
        model_names (list[str]): List of model module names

    Returns:
        dict:
            {
                model_name: model_manager_class
            }
        or None if no models were loaded successfully.
    """
    loaded_models = {}

    for model_name in model_names:
        # Construct full module path, e.g., "models.RuleBased"
        module_full_name = f"models.{model_name}"

        try:
            # Dynamically import the module
            model_module = importlib.import_module(module_full_name)

            # Retrieve the expected manager class
            model_manager = getattr(model_module, f"{model_name}Manager")

            loaded_models[model_name] = model_manager
            print(f"Successfully loaded '{model_name}Manager' from {module_full_name}")

        except ImportError as e:
            # Module file does not exist or import failed
            print(f"Error: Could not import module {module_full_name}. Reason: {e}")

        except AttributeError:
            # Module exists but class name is missing or mismatched
            print(f"Error: Module {module_full_name} loaded, but '{model_name}Manager' not found.")

        except Exception as e:
            # Catch-all for unexpected import-time errors
            print(f"An unexpected error occurred loading {module_full_name}: {e}")

    if not loaded_models:
        print("Error: No models were successfully loaded.")
        return None

    return loaded_models


# ============================================================
# DATASET PROCESSING AND MODEL EVALUATION
# ============================================================
def process_data_folders(folder_name, loaded_models, run_mode, encoding='standard'):
    """
    Processes dataset subdirectories and evaluates each model.

    Expected directory structure:
        ./data/<folder_name>/<drug_name>/
            ├── X.csv
            ├── Y.csv
            └── (optional) cgr_outputs/   [for FCGR encoding]

    For each model and each drug dataset:
        - Load data
        - Instantiate model manager
        - Run validation/testing
        - Collect and save results

    Args:
        folder_name (str): Dataset folder inside ./data
        loaded_models (dict): Model name -> model manager class
        run_mode (str): 'crossval' or 'test'
        encoding (str): Encoding method:
                        - 'standard' (default)
                        - 'LE'       (Label Encoding)
                        - 'OHE'      (One-Hot Encoding)
                        - 'FCGR'     (Frequency CGR matrices)
    """

    # Base dataset path
    base_path = os.path.join('.', 'data', folder_name)

    print(f"Using encoding: {encoding}")
    print(f"--- Processing base directory: {base_path} ---")

    # Validate dataset folder existence
    if not os.path.isdir(base_path):
        print(f"Error: Base directory not found: {base_path}")
        return

    # Collect all subdirectories (each representing a drug)
    try:
        subdirectories = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing directory {base_path}: {e}")
        return

    if not subdirectories:
        print("No subdirectories found.")
        return

    found_files_count = 0

    # Select the appropriate validation function dynamically
    func_name = f"run_{run_mode}"
    func = globals().get(func_name)

    # Create a filesystem-safe dataset name
    dataset_name = folder_name.rstrip('/')
    safe_dataset_name = dataset_name.replace('/', '_').replace(' ', '_').strip('_')

    # ========================================================
    # Loop over models
    # ========================================================
    for model_name, model_manager in loaded_models.items():
        model_results = []

        # ====================================================
        # Loop over drug subdirectories
        # ====================================================
        for subdir_name in subdirectories:
            subdir_path = os.path.join(base_path, subdir_name)

            x_csv_path = os.path.join(subdir_path, 'X.csv')
            y_csv_path = os.path.join(subdir_path, 'Y.csv')

            # ------------------------------------------------
            # SDCNN-specific logic:
            #   - Requires Y.csv
            #   - Requires Z.pkl (pickled genomic variant data)
            #   - Does NOT require X.csv
            # ------------------------------------------------
            if encoding == 'SDCNN':
                if not os.path.isfile(y_csv_path):
                    print(f"Skipping '{subdir_name}': Missing Y.csv.")
                    continue

                z_pkl_path = os.path.join(subdir_path, 'Z.pkl')
                if not os.path.isfile(z_pkl_path):
                    print(f"Skipping '{subdir_name}': Missing Z.pkl (SD-CNN genomic data).")
                    continue

                found_files_count += 1
                num_features = 0  # Determined internally by SD-CNN loader

            # ------------------------------------------------
            # FCGR-specific logic:
            #   - Requires Y.csv
            #   - Requires cgr_outputs/ directory
            #   - Does NOT require X.csv
            # ------------------------------------------------
            elif encoding == 'FCGR':
                if not os.path.isfile(y_csv_path):
                    print(f"Skipping '{subdir_name}': Missing Y.csv.")
                    continue

                cgr_dir = os.path.join(subdir_path, 'cgr_outputs')
                if not os.path.isdir(cgr_dir):
                    print(f"Skipping '{subdir_name}': Missing cgr_outputs/ directory.")
                    continue

                found_files_count += 1
                num_features = 0  # Determined internally by FCGR

            # ------------------------------------------------
            # Standard / LE / OHE encodings require X.csv + Y.csv
            # ------------------------------------------------
            else:
                if not (os.path.isfile(x_csv_path) and os.path.isfile(y_csv_path)):
                    print(f"Skipping '{subdir_name}': Missing X.csv or Y.csv.")
                    continue

                found_files_count += 1
                df = pd.read_csv(x_csv_path)
                num_features = df.shape[1]

            drug_name = subdir_name
            data_info = f"{dataset_name}_{drug_name}"

            print(f"Found data in '{subdir_name}'. data_info = {data_info}")
            print(f"  -> Running {run_mode} using model: {model_name}")

            # Instantiate model manager with feature dimensionality
            manager_instance = model_manager(num_features)

            try:
                # ------------------------------------------------
                # Run evaluation
                # ------------------------------------------------
                local_result = func(
                    x_csv_path,
                    y_csv_path,
                    manager_instance,
                    data_info,
                    encoding)

                # Attach dataset metadata to result
                push_result = {
                    "Dataset": dataset_name,
                    "Data": drug_name,
                    **local_result
                }

                model_results.append(push_result)
                print(f"  -> Evaluation complete for {model_name}")

            except Exception as e:
                print(f"  -> ERROR running evaluation for {model_name}: {e}")

        # ====================================================
        # Save results for this model
        # ====================================================
        if model_results:
            results_df = pd.DataFrame(model_results)

            # Improve console readability
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 50)
            pd.set_option('display.width', 120)

            print(results_df)

            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = model_name.replace('/', '_').replace(' ', '_')
            output_filename = (
                f'results/output_{safe_dataset_name}_{safe_model_name}_{timestamp}.csv'
            )

            results_df.to_csv(output_filename, index=False)
            print(f"\nResults saved to '{output_filename}'")

    # ========================================================
    # Final sanity message
    # ========================================================
    if found_files_count == 0:
        if encoding == 'SDCNN':
            print(f"\nNo subdirectories contained Y.csv and Z.pkl (SD-CNN data).")
        elif encoding == 'FCGR':
            print(f"\nNo subdirectories contained Y.csv and cgr_outputs/.")
        else:
            print(f"\nNo subdirectories contained both X.csv and Y.csv.")

    print(f"--- Processing for {folder_name} complete ---")


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    """
    Entry point for command-line execution.
    Handles:
        - Argument parsing
        - Model discovery
        - Dataset processing
    """

    # Allow local imports from current directory
    sys.path.insert(0, os.path.abspath('.'))

    parser = argparse.ArgumentParser(
        description="Process data folders and evaluate models."
    )

    # Dataset source folders
    parser.add_argument(
        "-s", "--source",
        dest="folder_names",
        required=True,
        nargs='+',
        help="One or more folder names inside ./data"
    )

    # Models to load
    parser.add_argument(
        "-m", "--models",
        dest="model_names",
        required=True,
        nargs='+',
        help="Model names OR the single keyword 'all'"
    )

    # Evaluation mode
    parser.add_argument(
        "-r", "--runmode",
        dest="run_mode",
        required=True,
        choices=["crossval", "test"],
        help="Choose 'crossval' or 'test'"
    )

    # Encoding method
    parser.add_argument(
        "-e", "--encoding",
        dest="encoding",
        choices=["LE", "OHE", "FCGR", "SDCNN", "standard"],
        default="standard",
        help=(
            "Encoding method (default: standard). "
            "Options: 'standard' (feature-based), 'LE' (Label Encoding), "
            "'OHE' (One-Hot Encoding), 'FCGR' (Frequency CGR), 'SDCNN' (Spatial Dropout CNN genomic variants)"
        )
    )

    args = parser.parse_args()

    # ========================================================
    # Handle 'all' models option
    # ========================================================
    if args.model_names == ['all']:
        print("Loading all models from ./models")

        models_dir = os.path.join('.', 'models')
        if not os.path.isdir(models_dir):
            print(f"Error: models directory not found: {models_dir}")
            return

        model_files = [
            f for f in os.listdir(models_dir)
            if f.endswith('.py') and f != '__init__.py'
        ]
        model_names_to_load = [os.path.splitext(f)[0] for f in model_files]

        if not model_names_to_load:
            print("Error: No model files found.")
            return

        print(f"Discovered models: {', '.join(model_names_to_load)}")
    else:
        model_names_to_load = args.model_names

    # ========================================================
    # Load models
    # ========================================================
    loaded_models = load_models(model_names_to_load)
    if loaded_models is None:
        print("Exiting: No models loaded.")
        return

    # Create directory for saving trained models (if needed)
    os.makedirs("saved_models", exist_ok=True)

    # ========================================================
    # Process datasets
    # ========================================================
    print(f"\nProcessing datasets: {', '.join(args.folder_names)}")
    print(f"Using models: {', '.join(loaded_models.keys())}\n")

    for folder in args.folder_names:
        process_data_folders(
            folder,
            loaded_models,
            args.run_mode,
            args.encoding
        )


if __name__ == "__main__":
    main()
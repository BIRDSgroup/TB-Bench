import os
import subprocess
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Map of drug short codes to loci
drug_loci_map = {
    "AMK": ["rrs-rrl", "eis"],
    "CAP": ["rrs-rrl", "tlyA"],
    "CIP": ["gyrBA"],
    "ETO": ["ethAR", "FabG1-inhA"],
    "KAN": ["rrs-rrl", "eis"],
    "LFX": ["gyrBA"],
    "MFX": ["gyrBA"],
    "OFX": ["gyrBA"],
    # New drugs
    "CYC": ["alr", "ald", "ddlA", "cycA"],
    "MB": ["rpoBC"],
    "PTO": ["FabG1-inhA", "ethAR"],
    "PAS": ["thyA"],
    "BDQ": ["mmpL5", "mmpS5", "Rv0678", "atpE", "pepQ"],
    "LZD": ["rplC", "rrl"]
}

# Map short codes to full names
drug_fullname_map = {
    "AMK": "AMIKACIN",
    "BDQ": "BEDAQUILINE",
    "CAP": "CAPREOMYCIN",
    "CIP": "CIPROFLOXACIN",
    "CYC": "CYCLOSERINE",
    "ETO": "ETHIONAMIDE",
    "KAN": "KANAMYCIN",
    "LFX": "LEVOFLOXACIN",
    "LZD": "LINEZOLID",
    "MB": "RIFABUTIN",
    "MFX": "MOXIFLOXACIN",
    "OFX": "OFLOXACIN",
    "PAS": "PARA-AMINOSALICYLIC_ACID",
    "PTO": "PROTHIONAMIDE"
}

# Directory for generated parameter files
param_dir = Path("generated_params")
param_dir.mkdir(exist_ok=True)

# Directory for generated phenotype files
phenotype_dir = Path("generated_phenotypes")
phenotype_dir.mkdir(exist_ok=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate parameter files (and optionally run preprocessing).")
parser.add_argument(
    "--preprocess",
    action="store_true",
    help="If set, run preprocessing after generating parameter files."
)
parser.add_argument(
    "--master-phenotype",
    type=str,
    default="/data/users/be21b039/all_drugs_sdcnn_k_fold/master_phenotype.csv",
    help="Path to master phenotype CSV file"
)
parser.add_argument(
    "--metadata-base",
    type=str,
    default="/data/public-data/TB_data_alldrugs",
    help="Base directory for drug metadata files"
)
parser.add_argument(
    "--random-seed",
    type=int,
    default=42,
    help="Random seed for reproducible train/test splits"
)
args = parser.parse_args()

RUN_PREPROCESSING = args.preprocess
MASTER_PHENOTYPE_PATH = args.master_phenotype
METADATA_BASE_DIR = args.metadata_base
RANDOM_SEED = args.random_seed

# Template for parameter file
template = """## Parameter file for pre_processing_script.py
# Run parameters
filter_size: 12
N_epochs: 100
weight_of_sensitive_class: 1
drug: {drug_full}
locus_list:
{loci_block}
## output paths specific to antibiotic
output_path:  output_optimal_epochs/{drug_short}_ccp_80_XVal_20210219/{drug_short}
threshold_file:  output_optimal_epochs/{drug_short}_ccp_80_XVal_20210219/{drug_short}_thresholds.csv
pkl_file_sparse_train:   output_optimal_epochs/{drug_short}_ccp_80_XVal_20210219/{drug_short}_X_sparse_train.npy.npz
pkl_file_sparse_test:   output_optimal_epochs/{drug_short}_ccp_80_XVal_20210219/{drug_short}_X_sparse_test.npy.npz
alpha_file:   output_optimal_epochs/{drug_short}_ccp_80_XVal_20210219/{drug_short}_alpha_sensitive_weight_1.0.csv
saved_model_path:  saved_models/{drug_short}_tf1_model
## Invariant to antibiotic
phenotype_file:  {phenotype_file_path}
genotype_input_directory: /data/users/be21b039/all_drugs_sdcnn_k_fold/fasta_files_aligned_final_20_not_in_master/{drug_short}/cleaned/
genotype_df_file: multitask_geno_train_test_{drug_short}.pkl 
pkl_file: multitask_geno_pheno_train_test_{drug_short}.pkl
"""

def load_metadata(drug_short, metadata_base_dir):
    """Load metadata file for a specific drug."""
    metadata_path = Path(metadata_base_dir) / drug_short / f"{drug_short}_metadata.txt"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Read metadata (space-separated: isolate_id label)
    metadata = pd.read_csv(metadata_path, sep=r'\s+', header=None, names=['isolate_id', 'label'])
    
    # Convert 'I' (Intermediate) labels to 'R' (Resistant) for stratification
    metadata['label'] = metadata['label'].replace('I', 'R')
    
    return metadata

def create_drug_phenotype_file(drug_short, master_phenotype_path, metadata_base_dir, output_dir, random_seed=42):
    """
    Create drug-specific phenotype file with stratified train/test split.
    
    Args:
        drug_short: Drug short code (e.g., 'PTO')
        master_phenotype_path: Path to master phenotype CSV
        metadata_base_dir: Base directory containing drug metadata
        output_dir: Directory to save generated phenotype file
        random_seed: Random seed for reproducibility
    
    Returns:
        Path to generated phenotype file
    """
    # Load master phenotype
    master_df = pd.read_csv(master_phenotype_path)
    
    # Load metadata for this drug
    metadata = load_metadata(drug_short, metadata_base_dir)
    
    # Check if we have enough samples for stratified split
    label_counts = metadata['label'].value_counts()
    print(f"     Label distribution: {dict(label_counts)}")
    
    if len(label_counts) < 2 or label_counts.min() < 2:
        raise ValueError(f"Insufficient samples for stratified split. Need at least 2 samples per class. Got: {dict(label_counts)}")
    
    # Perform stratified 80/20 split
    train_ids, test_ids = train_test_split(
        metadata['isolate_id'].values,
        test_size=0.20,
        random_state=random_seed,
        stratify=metadata['label'].values
    )
    
    # Create a copy of master dataframe
    drug_phenotype = master_df.copy()
    
    # Initialize category column if it doesn't exist
    if 'category' not in drug_phenotype.columns:
        drug_phenotype['category'] = ''
    
    # Set category to 'set1_original_10202' for isolates in this drug's metadata
    # Assuming the master phenotype has an 'isolate_id' column or similar identifier
    # Adjust the column name based on your actual data structure
    isolate_col = 'isolate_id'  # Change this if your column name is different
    
    if isolate_col not in drug_phenotype.columns:
        # Try to find the isolate identifier column
        possible_cols = ['isolate_id', 'sample_id', 'id', 'Isolate', 'Sample']
        for col in possible_cols:
            if col in drug_phenotype.columns:
                isolate_col = col
                break
        else:
            raise ValueError(f"Could not find isolate identifier column in master phenotype. Available columns: {drug_phenotype.columns.tolist()}")
    
    # Set category to 'set1_original_10202' ONLY for TRAIN isolates
    train_isolates_set = set(train_ids)
    train_mask = drug_phenotype[isolate_col].isin(train_isolates_set)
    drug_phenotype.loc[train_mask, 'category'] = 'set1_original_10202'
    
    # Save drug-specific phenotype file
    output_path = output_dir / f"{drug_short}_phenotype.csv"
    drug_phenotype.to_csv(output_path, index=False)
    
    print(f"[OK] Created phenotype file for {drug_short}: {output_path}")
    print(f"     Train samples: {len(train_ids)}, Test samples: {len(test_ids)}")
    print(f"     Train isolates marked with 'set1_original_10202': {train_mask.sum()}")
    
    return output_path

# Main processing loop
for drug_short, loci in drug_loci_map.items():
    print(f"\n{'='*60}")
    print(f"Processing drug: {drug_short}")
    print(f"{'='*60}")
    
    drug_full = drug_fullname_map[drug_short]
    loci_block = "\n".join([f"- {l}" for l in loci])
    
    # --- Ensure output directory exists ---
    output_dir = Path(f"output_optimal_epochs/{drug_short}_ccp_80_XVal_20210219/{drug_short}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Generate drug-specific phenotype file ---
    try:
        phenotype_file_path = create_drug_phenotype_file(
            drug_short=drug_short,
            master_phenotype_path=MASTER_PHENOTYPE_PATH,
            metadata_base_dir=METADATA_BASE_DIR,
            output_dir=phenotype_dir,
            random_seed=RANDOM_SEED
        )
    except FileNotFoundError as e:
        print(f"[WARNING] Skipping {drug_short}: {e}")
        continue
    except Exception as e:
        print(f"[ERROR] Failed to create phenotype file for {drug_short}: {e}")
        continue
    
    # Fill in the template
    content = template.format(
        drug_full=drug_full,
        drug_short=drug_short,
        loci_block=loci_block,
        phenotype_file_path=str(phenotype_file_path)
    )
    
    # Save param file
    param_file = param_dir / f"{drug_short}_parameter.txt"
    with open(param_file, "w") as f:
        f.write(content)
    print(f"[OK] Generated parameter file: {param_file}")
    
    # Optionally run preprocessing
    if RUN_PREPROCESSING:
        cmd = ["python", "pre_processing_script.py", str(param_file)]
        print(f"[INFO] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[OK] Preprocessing completed for {drug_short}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Preprocessing failed for {drug_short}: {e}")

print(f"\n{'='*60}")
print("All parameter files generated successfully!")
print(f"Parameter files saved in: {param_dir}")
print(f"Phenotype files saved in: {phenotype_dir}")
print(f"{'='*60}")
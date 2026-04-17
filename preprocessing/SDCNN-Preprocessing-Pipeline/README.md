# Pre-Processing Pipeline for SD-CNN Antibiotic Resistance Prediction

## Overview

This repository contains the full pre-processing and model-training pipeline for the **SD-CNN** that predicts antibiotic resistance phenotypes in *Mycobacterium tuberculosis* (MTB) from whole-genome sequencing (WGS) data.

The pipeline takes raw per-sample variant call files (VCF, produced by [Pilon](https://github.com/broadinstitute/pilon)) and transforms them into one-hot-encoded multiple-sequence alignment (MSA) tensors that are fed into the CNN. Resistance phenotypes (Resistant/Sensitive) are sourced from a master phenotype CSV and matched to each isolate.

### Supported Drugs (14)

| Code | Full Name | Code | Full Name |
|------|-----------|------|-----------|
| AMK | Amikacin | LFX | Levofloxacin |
| BDQ | Bedaquiline | LZD | Linezolid |
| CAP | Capreomycin | MB  | Rifabutin |
| CIP | Ciprofloxacin | MFX | Moxifloxacin |
| CYC | Cycloserine | OFX | Ofloxacin |
| ETO | Ethionamide | PAS | Para-aminosalicylic acid |
| KAN | Kanamycin | PTO | Prothionamide |

### Genomic Loci Analysed (30)

`acpM-kasA`, `gid`, `rpsA`, `clpC`, `embCAB`, `aftB-ubiA`, `rrs-rrl`, `ethAR`, `oxyR-ahpC`, `tlyA`, `KatG`, `rpsL`, `rpoBC`, `FabG1-inhA`, `eis`, `gyrBA`, `panD`, `pncA`, `alr`, `ald`, `ddlA`, `cycA`, `thyA`, `mmpL5`, `mmpS5`, `Rv0678`, `atpE`, `pepQ`, `rplC`, `rrl`

---

## Project Structure

```
Pre_processing_pipeline_SDCNN/
├── vcf_processor_cli_flag.py               # Step 1 – VCF → per-locus aligned FASTA
├── snpConcatenater_w_exclusion_frompilonvcf_2.9.pl  # Core Perl: VCF → MSA FASTA
├── get_seq_coord.pl                        # Helper Perl: extract reference subsequence
├── h37rv.fasta                             # M. tuberculosis H37Rv reference genome
├── IDfail.tab                              # QC-failed sample IDs to exclude (currently empty)
├── run_preprocess_modified_pheno.py        # Step 2 – generate configs & phenotype splits
├── pre_processing_script.py               # Step 3 – build data tensors + define/train SD-CNN
└── tb_cnn_codebase.py                     # Shared library: encoding, losses, metrics
```

---

## Pipeline Architecture

```
WGS data (Pilon VCF.gz)
        │
        ▼
┌───────────────────────────────────────┐
│  vcf_processor_cli_flag.py            │  Step 1
│  • Unzip VCF.gz files                 │
│  • Call Perl: extract locus FASTAs    │
│  • Align with MAFFT                   │
└───────────────┬───────────────────────┘
                │  aligned FASTA files per locus
                ▼
┌───────────────────────────────────────┐
│  run_preprocess_modified_pheno.py     │  Step 2
│  • Stratified 80/20 train/test split  │
│  • Generate drug-specific YAML config │
│  • Generate phenotype CSV per drug    │
└───────────────┬───────────────────────┘
                │  YAML param files + phenotype CSVs
                ▼
┌───────────────────────────────────────┐
│  pre_processing_script.py             │  Step 3
│  • Build genotype–phenotype DataFrame │
│  • One-hot encode sequences           │
│  • Construct sparse 4D input tensor X │
│  • Compute class-weight alpha matrix  │
│  • Define & train SD-CNN (Keras)      │
└───────────────────────────────────────┘
```

---

## Dependencies & Requirements

### System Requirements
- Python ≥ 3.8
- Perl ≥ 5.10
- [MAFFT](https://mafft.cbrc.jp/alignment/software/) — must be on `PATH`

### Python Packages

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
biopython
sparse
pyyaml
```

Install all Python dependencies with:

```bash
pip install tensorflow numpy pandas scikit-learn biopython sparse pyyaml
```

### Reference Data (included)
- `h37rv.fasta` — *M. tuberculosis* H37Rv complete genome (NC_000962.3, ~4.4 MB)
- `IDfail.tab` — placeholder file for QC-failed sample IDs; **must exist** even if empty

---

## Installation / Setup

```bash
# 1. Clone or copy this directory to your server
git clone <repo_url> Pre_processing_pipeline_SDCNN
cd Pre_processing_pipeline_SDCNN

# 2. Install Python dependencies
pip install tensorflow numpy pandas scikit-learn biopython sparse pyyaml

# 3. Install MAFFT (example for Ubuntu/Debian)
sudo apt-get install mafft

# 4. Configure paths in vcf_processor_cli_flag.py (see "Configuration" below)
```

### Configuration

Before running **Step 1**, open `vcf_processor_cli_flag.py` and update the module-level path variables at the top of the file to match your environment:

| Variable | Description |
|----------|-------------|
| `vcf_source_dir` | Directory containing per-sample `*.vcf.gz` files |
| `perl_script` | Absolute path to `snpConcatenater_w_exclusion_frompilonvcf_2.9.pl` |
| `reference_dir` | Directory containing one reference FASTA per locus (for MAFFT `--add`) |

---

## Usage

### Step 1 — Process VCF files to aligned FASTAs

`vcf_processor_cli_flag.py` supports three methods for specifying drug metadata files (two-column TSV: `sample_id  R/S/I`):

```bash
# Method A: point to individual drug metadata files
python vcf_processor_cli_flag.py \
    --LFX /data/LFX/LFX_metadata.txt \
    --AMK /data/AMK/AMK_metadata.txt

# Method B: directory containing drug subdirectories
python vcf_processor_cli_flag.py \
    --meta-dir /data/public-data/TB_data_alldrugs

# Method C: config file (drug=path pairs, one per line)
python vcf_processor_cli_flag.py --config drugs.conf

# Skip MAFFT alignment (Perl FASTA extraction only)
python vcf_processor_cli_flag.py --meta-dir /path/to/meta --perl-only

# Control parallelism
python vcf_processor_cli_flag.py --meta-dir /path/to/meta --max-workers 4
```

**Config file format (`drugs.conf`):**
```
LFX=/path/to/LFX_metadata.txt
AMK=/path/to/AMK_metadata.txt
# Lines starting with # are comments
```

**Outputs** (written relative to `perl_dir`):
- `perl_output_fastas/<DRUG>/<locus>.fasta` — raw per-locus FASTAs
- `fasta_files_aligned_final_20_not_in_master/<DRUG>/<locus>_aligned.fasta` — MAFFT-aligned FASTAs

---

### Step 2 — Generate parameter files and phenotype splits

```bash
# Generate YAML configs and phenotype files only:
python run_preprocess_modified_pheno.py \
    --master-phenotype /path/to/master_phenotype.csv \
    --metadata-base    /data/public-data/TB_data_alldrugs \
    --random-seed      42

# Also launch pre_processing_script.py for each drug:
python run_preprocess_modified_pheno.py \
    --master-phenotype /path/to/master_phenotype.csv \
    --metadata-base    /data/public-data/TB_data_alldrugs \
    --preprocess
```

**Outputs:**
- `generated_params/<DRUG>_parameter.txt` — YAML config for each drug
- `generated_phenotypes/<DRUG>_phenotype.csv` — stratified phenotype CSV (80 % train marked as `set1_original_10202`)

---

### Step 3 — Build data matrices and train the SD-CNN

```bash
python pre_processing_script.py generated_params/LFX_parameter.txt
```

This script:
1. Reads the YAML parameter file.
2. Creates (or loads from cache) the genotype–phenotype pickle.
3. One-hot encodes all locus sequences and builds the 4D sparse input tensors `X_train` and `X_test`.
4. Computes the class-weight alpha matrix (accounts for R/S class imbalance).
5. Defines the SD-CNN architecture and trains it.

**Intermediate outputs** (all paths set in the YAML config):
- `*_df_geno_pheno.csv` — filtered genotype–phenotype table
- `*_X_sparse_train.npy.npz` / `*_X_sparse_test.npy.npz` — sparse input tensors
- `*_alpha_*.csv` — class-weight matrix

---

## Key Modules & Scripts

### `tb_cnn_codebase.py`
Shared utility library imported by `pre_processing_script.py`.

| Function / Class | Purpose |
|-----------------|---------|
| `get_one_hot(sequence)` | Encodes a DNA string as a (L×5) one-hot array (A/C/T/G/gap) |
| `sequence_dictionary(filename)` | Reads a multi-sample FASTA into a keyed DataFrame |
| `make_genotype_df(directory)` | Loads all locus FASTAs and joins into one DataFrame |
| `rs_encoding_to_numeric(df, drugs)` | Converts R/S labels to 0/1 (missing → −1) |
| `alpha_mat(y, df, weight)` | Builds the class-weight matrix that up/down-weights S/R classes |
| `make_geno_pheno_pkl(**kwargs)` | End-to-end: read phenotypes + genotypes, encode, save pickle |
| `create_X(df)` | Assembles the 4D numpy tensor (N, 5, L_max, N_loci) with zero-padding |
| `masked_multi_weighted_bce(α, ŷ)` | Custom Keras loss: weighted BCE ignoring missing phenotype entries |
| `masked_weighted_accuracy(α, ŷ)` | Custom Keras metric corresponding to the above loss |
| `get_threshold_val(y_true, y_pred)` | Finds the optimal classification threshold (max sens + spec) |
| `split_into_traintest(X, df, cat)` | Splits sparse array into train/test using category label |

### `vcf_processor_cli_flag.py`
CLI driver for VCF → aligned FASTA conversion. Uses `ProcessPoolExecutor` to process multiple drugs in parallel. Calls the Perl script for each locus, then aligns outputs with MAFFT.

### `run_preprocess_modified_pheno.py`
Orchestration script that loops over all 14 drugs, creates output directories, generates YAML parameter files from a template, performs stratified 80/20 train/test splits, writes per-drug phenotype CSV files, and optionally launches `pre_processing_script.py` for each drug.

### `snpConcatenater_w_exclusion_frompilonvcf_2.9.pl`
Core Perl script (author: Maha Farhat) that:
- Reads a list of Pilon VCF files (`input_vcf_files.txt`).
- Applies quality filters (QUAL ≥ 10, heterozygosity threshold 10 %).
- Handles SNPs, MNPs (multi-base substitutions), and indels.
- Skips positions listed in the exclusion BED file and samples in `IDfail.tab`.
- Outputs a multiple-sequence alignment FASTA for the specified genomic region.

**Usage (called automatically by `vcf_processor_cli_flag.py`):**
```
perl snpConcatenater_w_exclusion_frompilonvcf_2.9.pl \
    <exclude.BED> <IDfail.tab> INDEL REGION <start>-<end> pos \
    > output.fasta
```

### `get_seq_coord.pl`
Helper Perl script that extracts a sub-sequence from a FASTA file by 1-based coordinate range, with support for reverse-complement and multi-exon extraction. Called internally by `snpConcatenater_w_exclusion_frompilonvcf_2.9.pl` to fetch the H37Rv reference sequence for a given locus window.

### `h37rv.fasta`
*Mycobacterium tuberculosis* H37Rv complete genome (accession NC_000962.3, length 4,411,532 bp). Used as the reference sequence for:
- Region extraction (by `get_seq_coord.pl`)
- MAFFT profile alignment (`--add --keeplength`)

### `IDfail.tab`
Tab-delimited file listing sample IDs that failed QC and should be excluded from the Perl MSA construction. **Currently empty** (no samples excluded). Add one sample ID per line to exclude samples.

---

## Data Flow Summary

```
Input:
  master_phenotype.csv  ─────────────────────────────────────┐
  per-drug metadata TSVs (sample_id  R/S/I label)            │
  per-sample *.vcf.gz (Pilon variant calls)                  │
                                                             │
Step 1 (vcf_processor_cli_flag.py)                          │
  └─ per-locus aligned FASTA files                          │
        │                                                   │
Step 2 (run_preprocess_modified_pheno.py) ◄─────────────────┘
  └─ YAML param file per drug
  └─ phenotype CSV per drug (with train/test category)
        │
Step 3 (pre_processing_script.py)
  └─ genotype–phenotype pickle
  └─ sparse X tensors (train + test)
  └─ alpha class-weight matrix
  └─ trained SD-CNN model
```

---

## Authors

- Michael Chen (original SD-CNN codebase)
- Anna G. Green
- Chang-ho Yoon
- Maha Farhat (`snpConcatenater_w_exclusion_frompilonvcf_2.9.pl`)

---

## Notes

- The `pre_processing_script.py` file contains TODO markers for replacing hard-coded `X.shape` references with explicit passed arguments in the CNN definition — these are known areas awaiting refactoring.
- Hardcoded server paths in `vcf_processor_cli_flag.py` (e.g., `/data/users/be21b039/...`) must be updated before running on a new system.
- `IDfail.tab` **must exist** in the working directory when running the Perl script, even when empty. Do not delete it.

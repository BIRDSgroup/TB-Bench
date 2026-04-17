# TB-Bench

A benchmarking framework for evaluating machine (ML) learning and deep learning (DL) models that predict **antibiotic resistance in _Mycobacterium tuberculosis_ (MTB)** from whole-genome sequencing (WGS) data.

---

## License

Copyright 2026 BIRDS Group, IIT Madras

TB-Bench is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

TB-Bench is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with TB-Bench. If not, see https://www.gnu.org/licenses/.

## Overview

TB-Bench trains and evaluates a diverse set of classical ML and DL models on genotype–phenotype datasets, one dataset per antibiotic drug. It supports two evaluation modes:

- **`crossval`** — stratified 80/20 split, 4-fold cross-validation for threshold selection, final evaluation on the held-out 20%.
- **`test`** — load a previously trained model and evaluate on new external data.

Results are saved as timestamped CSVs under `results/`, and trained models are persisted under `saved_models/`.

---

## Project Structure

```
TB-Bench/
├── main.py                          # Entry point
├── validation.py                    # Training, CV, threshold selection, evaluation
├── model_hyperparams.tsv            # Persisted Youden-optimised thresholds
├── requirements.txt
├── run_batch_yangml_models.sh       # Batch runner: Yang et al. 2018 classical ML models
├── run_batch_mliamr_models.sh       # Batch runner: ML-iAMR sequence models
├── run_batch_external_validation.sh # Batch runner: external (cross-dataset) validation
├── models/                          # One file per model, each exposing a *Manager class
├── preprocessing/
│   ├── Cutoff/                      # Converts raw variant CSVs to X.csv / Y.csv pairs
│   ├── SDCNN-Preprocessing-Pipeline/  # VCF → MSA → SD-CNN tensors
│   └── TB-WGS-Preprocessing-Pipeline/ # Raw WGS download, alignment, variant calling
├── data/                            # Input datasets (one subfolder per dataset)
├── results/                         # Output CSVs (auto-generated)
├── saved_models/                    # Persisted model files (auto-generated)
└── logs/                            # Log files from batch runs
```

---

## Installation

### Prerequisites

- Python 3.9+
- [MAFFT](https://mafft.cbrc.jp/alignment/software/) (required for the SD-CNN preprocessing pipeline)
- Perl (required for SD-CNN VCF processing scripts)

### Python dependencies

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---|---|
| `scikit-learn` | Classical ML models and cross-validation |
| `tensorflow` / `keras` | Deep learning models |
| `scikeras` | Keras ↔ scikit-learn wrapper |
| `xgboost` | Gradient boosting |
| `sparse` | Sparse 4D tensors for SD-CNN |
| `numpy`, `pandas`, `scipy` | Core data processing |

---

## Usage

### Basic syntax

```bash
python main.py -s <dataset_folder> -m <ModelName> -r <crossval|test> [-e <encoding>]
```

| Argument | Description | Default |
|---|---|---|
| `-s` / `--source` | One or more dataset folder names inside `./data/` | required |
| `-m` / `--models` | Model module name(s), or `all` to run every available model | required |
| `-r` / `--runmode` | `crossval` or `test` | required |
| `-e` / `--encoding` | `standard`, `LE`, `OHE`, `FCGR`, or `SDCNN` | `standard` |

### Examples

```bash
# Cross-validate Bernoulli NB on Tier-1/2 gene variant dataset
python main.py -s _Tier12_Genes -m BernoulliNB_Yang2018 -r crossval

# Cross-validate 1D CNN with label encoding on BDQ sequence data
python main.py -s BDQ-Only-SeqData -m CNN_1D_MLiAMR -r crossval -e LE

# Cross-validate 2D CNN using FCGR matrix encoding
python main.py -s BDQ-Only-SeqData -m CNN_2D_MLiAMR -r crossval -e FCGR

# Run all models on a dataset
python main.py -s _All_SNPs -m all -r crossval

# Run test-mode inference using a previously trained model
python main.py -s _Chinese_CodingOnly_test -m BernoulliNB_Yang2018 -r test
```

### Batch runners

Three shell scripts are provided for running multiple models and datasets in parallel. Each script supports configuration via environment variables.

#### Yang et al. 2018 models

```bash
bash run_batch_yangml_models.sh

# Override run mode:
RUN_MODE=test bash run_batch_yangml_models.sh
```

Runs all 6 Yang models × 4 datasets (`_All_SNPs`, `_CodingRegion_SNPs`, `_Tier12_Genes`, `_Tier12_Genes_without10`) in parallel. Logs are written to `logs_local/`.

#### ML-iAMR sequence models

```bash
bash run_batch_mliamr_models.sh
```

Runs SVC, RF, LR, and CNN-1D models on sequence-encoded data with label encoding. Logs are written to `logs_mliamr/`.

#### External validation

```bash
bash run_batch_external_validation.sh
```

Cross-applies models trained on one dataset to three external test datasets (72 total jobs). Useful for evaluating cross-population generalisation.

---

## Input Data Format

Data is organised as `data/<dataset_name>/<drug_name>/`. The required files depend on the encoding used.

### `standard` / `LE` encoding (binary or integer variant features)

```
data/<dataset_name>/<drug_name>/
├── X.csv    # (n_samples × n_features); no row index; numeric values
└── Y.csv    # Single column named 'target'; 0 = drug-susceptible, 1 = drug-resistant
```

### `OHE` encoding

Same file layout as `standard`. Raw categorical or integer allele values in `X.csv`; one-hot encoding is fitted and applied internally, with the fitted encoder saved to `saved_models/`.

### `FCGR` encoding (Frequency Chaos Game Representation matrices)

```
data/<dataset_name>/<drug_name>/
├── Y.csv
└── cgr_outputs/
    ├── cgr_output_0.csv    # Flattened FCGR matrix for sample 0
    ├── cgr_output_1.csv
    └── ...
```

Files must be named `cgr_output_<index>.csv` and be sortable in sample order.

### `SDCNN` encoding (SD-CNN one-hot tensors)

```
data/<dataset_name>/<drug_name>/
├── Y.csv
└── Z.pkl    # Pickled pandas DataFrame with columns:
             #   'category', '<DRUG>' (R/S phenotype),
             #   and '<locus>_one_hot' for each resistance-associated locus
```

See the [SD-CNN Preprocessing Pipeline](#sd-cnn-preprocessing-pipeline) section for how to generate `Z.pkl`.

---

## Models

All models live in `models/` and implement the `AbstractModel` interface defined in `models/Model.py`. Each exposes a `*Manager` class with a `name`, `model`, `param_grid`, and `static_params`.

### Yang et al. 2018 — classical ML (standard encoding)

| Model file | Algorithm |
|---|---|
| `BernoulliNB_Yang2018.py` | Bernoulli Naïve Bayes with Beta-prior class weighting |
| `LogisticRegressionL1_Yang2018.py` | Logistic Regression (L1, liblinear solver) |
| `LogisticRegressionL2_Yang2018.py` | Logistic Regression (L2, lbfgs solver) |
| `RandomForest_Yang2018.py` | Random Forest |
| `SVCLinear_Yang2018.py` | Support Vector Classifier (linear kernel) |
| `SVCRBF_Yang2018.py` | Support Vector Classifier (RBF kernel) |

### ML-iAMR — sequence models (LE / OHE / FCGR encoding)

| Model file | Algorithm |
|---|---|
| `LR_MLiAMR.py` | Logistic Regression |
| `RF_MLiAMR.py` | Random Forest |
| `SVC_MLiAMR.py` | Support Vector Classifier |
| `CNN_1D_MLiAMR.py` | 1D CNN (Conv1D × 4 → Dense(128) → Dense(2, softmax)) |
| `CNN_2D_MLiAMR.py` | 2D CNN (Conv2D × 4 → Dense(128) → Dense(2, softmax)); expects FCGR 200×200 matrices |

### Deep learning models

| Model file | Algorithm |
|---|---|
| `ANN_Ankita.py` | Shallow ANN (Dense(8, relu) → Dense(1, sigmoid)) |
| `XGBoost_Ankita.py` | XGBoost gradient-boosted trees |
| `DeepAMR.py` | Autoencoder + multi-task classifier; trained with cyclic learning rate (base 0.0001, max 0.003); saves weights to `DeepAMR_weights/` |
| `WDNN.py` | Wide & Deep Neural Network (3 × Dense(256)+BN+Dropout concatenated with raw input → Dense(1, sigmoid)) |
| `Deep.py` | Fully-connected deep network |
| `MTB_SD_CNN.py` | Spatial Dropout CNN from Chen, Green & Yoon; input: 4D sparse one-hot MSA tensors (SDCNN encoding) |
| `DecisionTree.py` | CART Decision Tree |
| `Treeresist.py` | Treesist-TB custom decision tree with genetic-variant-aware splitting |

---

## Evaluation Pipeline

### Cross-validation (`crossval` mode)

1. Stratified 80/20 train/test split.
2. Hyperparameter tuning via grid search on the 80% split (if `param_grid` is provided).
3. 4-fold stratified cross-validation: for each fold, compute Youden's J statistic across 101 evenly-spaced thresholds (0–1).
4. Select the threshold that maximises average Youden's J across folds; persist it to `model_hyperparams.tsv`.
5. Retrain on the full 80% with best hyperparameters.
6. Save model to `saved_models/`.
7. Evaluate on both the training set and the held-out 20%.

### Test mode (`test` mode)

Loads the saved model and the Youden-optimised threshold from `model_hyperparams.tsv`, then evaluates on new data.

### Reported metrics

| Metric | Description |
|---|---|
| ACC | Accuracy |
| Sen | Sensitivity (recall) |
| Spe | Specificity |
| F1 | F1 score |
| ROC-AUC | Area under the ROC curve |
| PR-AUC | Area under the precision-recall curve |

Results are written to `results/output_<dataset>_<model>_<timestamp>.csv`.

---

## Preprocessing Pipelines

### Cutoff10

`preprocessing/Cutoff/Cutoff10.py` converts raw TB WGS variant/metadata files (applying a 10% minor-allele-frequency cutoff) into benchmarking-ready `X.csv` / `Y.csv` pairs.

**Expected inputs (per drug subfolder):**
- `<folder>_data_10.csv` — genotype feature matrix (rows = isolates, columns = variants)
- `<folder>_metadata.csv` — must include an `S/R` susceptibility column

Labels are encoded as `S→0`, `R→1`; rows with missing or ambiguous labels are removed.

### SD-CNN Preprocessing Pipeline

`preprocessing/SDCNN-Preprocessing-Pipeline/` converts per-sample VCF files (Pilon output) into the `Z.pkl` tensor format required by `MTB_SD_CNN.py`. See the pipeline's own [README](preprocessing/SDCNN-Preprocessing-Pipeline/README.md) for full usage instructions.

**Three-step workflow:**

1. **`vcf_processor_cli_flag.py`** — extracts per-locus FASTAs from VCF files and aligns them with MAFFT.
2. **`run_preprocess_modified_pheno.py`** — performs stratified 80/20 splitting and generates per-drug phenotype CSVs and YAML configs.
3. **`pre_processing_script.py`** — one-hot encodes aligned sequences into 4D sparse tensors and outputs `Z.pkl`.

**Supported drugs (14):** AMK, BDQ, CAP, CIP, CYC, ETO, KAN, LFX, LZD, MB, MFX, OFX, PAS, PTO

**Analysed genomic loci (30):** `acpM-kasA`, `gid`, `rpsA`, `clpC`, `embCAB`, `aftB-ubiA`, `rrs-rrl`, `ethAR`, `oxyR-ahpC`, `tlyA`, `KatG`, `rpsL`, `rpoBC`, `FabG1-inhA`, `eis`, `gyrBA`, `panD`, `pncA`, `alr`, `ald`, `ddlA`, `cycA`, `thyA`, `mmpL5`, `mmpS5`, `Rv0678`, `atpE`, `pepQ`, `rplC`, `rrl`

### TB-WGS-Preprocessing-Pipeline

`preprocessing/TB-WGS-Preprocessing-Pipeline/` handles raw WGS data acquisition and variant calling:

- **`scripts/fastq_download.sh`** — downloads FASTQ files via the bundled SRA Toolkit (`sratoolkit.3.3.0-ubuntu64/`).
- **`scripts/annot_variants.sh`** — aligns reads to the H37Rv reference genome (`NC_000962.3.fasta`), calls variants, and annotates them.
- **`WHO/resolved_1720.tsv`** — WHO phenotype catalogue (1,720 resolved entries).

---

## Output Files

| Path | Description |
|---|---|
| `results/output_<dataset>_<model>_<timestamp>.csv` | Per-drug metrics for each run |
| `saved_models/<key>_model.pkl` | Serialised sklearn or Keras model |
| `saved_models/<key>_OHE_encoder.pkl` | Fitted one-hot encoder (OHE encoding only) |
| `saved_models/DeepAMR_weights/` | DeepAMR network weights |
| `model_hyperparams.tsv` | Youden-optimised thresholds, keyed by `<Model>_<Dataset>_<Encoding>` |

---

## Adding a New Model

1. Create `models/MyModel.py`.
2. Define a class `MyModelManager` that inherits from `AbstractModel` (`models/Model.py`).
3. Implement the four required properties: `name`, `model`, `param_grid`, `static_params`.
4. The framework will automatically discover and run your model when `-m MyModel` is passed.

```python
from models.Model import AbstractModel
from sklearn.ensemble import GradientBoostingClassifier

class MyModelManager(AbstractModel):
    @property
    def name(self):
        return "MyModel"

    @property
    def model(self):
        return GradientBoostingClassifier()

    @property
    def param_grid(self):
        return {}  # or a dict for hyperparameter tuning

    @property
    def static_params(self):
        return {"n_estimators": 100}
```

---

## Data Availability

All additional supplementary files and source data associated with this study have been deposited in the following repository:  
[Google Drive](https://drive.google.com/drive/folders/17DBf0hSLbbl0xsRwQb_7DQhSCP1qok6r?usp=drive_link)

---

## Citation

If you use TB-Bench in your research, please cite "TB-Bench: A Systematic Benchmark of Machine Learning and Deep Learning Methods for Second-Line TB Drug Resistance Prediction. Brintha VP, Saish Jaiswal, Ansh Meshram, Deepti PVS, Sidharthan S C, Manikandan Narayanan."
[bioRxiv link](https://doi.org/10.64898/2026.04.08.717138)

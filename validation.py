import pandas as pd
import numpy as np
import os
import sys
from sklearn import metrics
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import pickle
import json
import ast
import pdb
import matplotlib.pyplot as plt
import tensorflow as tf
import sparse
from models.tb_cnn_codebase import create_X, rs_encoding_to_numeric, alpha_mat
#import shap
#from tf_explain.core.gradients_inputs import GradientsInputs

df_geno_pheno = pd.DataFrame()  # Global variable to hold the genotype-phenotype data for SDCNN processing.

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

PARAM_FILE = 'model_hyperparams.tsv'

import tensorflow as tf

def load_cgr_data(data_dir):
    cgr_files = sorted([f for f in os.listdir(data_dir) if f.startswith("cgr_output_") and f.endswith(".csv")],
                    key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0])) # maintains the right order!!!!!!!
    cgr_matrices = []
    for file in cgr_files:
        filepath = os.path.join(data_dir, file)
        matrix = pd.read_csv(filepath, header=None).values.flatten()
        cgr_matrices.append(matrix)
    print(f"Loaded {len(cgr_matrices)} FCGR files.")
    return np.array(cgr_matrices)

def X_input_processing(X_path_str : str | None, Y_path_str : str, model, encoding_method : str, drug_name):
    
    global df_geno_pheno

    # Load X and Y based on the specified encoding method.
    # This function handles different data loading and preprocessing steps
    # depending on the encoding type (e.g., SDCNN, FCGR, OHE, LE).
    X_base_path = Path(X_path_str) if X_path_str else None
    Y = pd.read_csv(Y_path_str).squeeze("columns").values

    drug_locus_dict = {"AMK": {"drug_name": "AMIKACIN","locus_list": ["rrs-rrl", "eis"]},
    "BDQ": {"drug_name": "BEDAQUILINE","locus_list": ["mmpL5", "mmpS5", "Rv0678", "atpE", "pepQ"]},
    "CAP": {"drug_name": "CAPREOMYCIN","locus_list": ["rrs-rrl", "tlyA"]},
    "CIP": {"drug_name": "CIPROFLOXACIN","locus_list": ["gyrBA"]},
    "CYC": {"drug_name": "CYCLOSERINE","locus_list": ["alr", "ald", "ddlA", "cycA"]},
    "ETO": {"drug_name": "ETHIONAMIDE","locus_list": ["ethAR", "FabG1-inhA"]},
    "KAN": {"drug_name": "KANAMYCIN","locus_list": ["rrs-rrl", "eis"]},
    "LFX": {"drug_name": "LEVOFLOXACIN","locus_list": ["gyrBA"]},
    "LZD": {"drug_name": "LINEZOLID","locus_list": ["rplC", "rrl"]},
    "MB": {"drug_name": "RIFABUTIN","locus_list": ["rpoBC"]},
    "MFX": {"drug_name": "MOXIFLOXACIN","locus_list": ["gyrBA"]},
    "OFX": {"drug_name": "OFLOXACIN","locus_list": ["gyrBA"]},
    "PAS": {"drug_name": "PARA-AMINOSALICYLIC_ACID","locus_list": ["thyA"]},
    "PTO": {"drug_name": "PROTHIONAMIDE","locus_list": ["FabG1-inhA", "ethAR"]}}

    if encoding_method.upper() == 'SDCNN':
        print("Using SDCNN data loader...")
        data_folder_dir = X_base_path.parent
        drug_name = drug_name.split('_')[-1]
        DRUG = drug_locus_dict[drug_name]["drug_name"]
        locus_list = drug_locus_dict[drug_name]["locus_list"]
        df_geno_pheno = pd.read_pickle(data_folder_dir / "Z.pkl")
        
        columns_to_keep = ["category", DRUG] + [x+"_one_hot" for x in locus_list]
        df_geno_pheno_subset = df_geno_pheno[columns_to_keep]
        del df_geno_pheno
        
        df_geno_pheno_subset = df_geno_pheno_subset.loc[
            np.logical_or(df_geno_pheno_subset[DRUG] == 'R', df_geno_pheno_subset[DRUG] == "S")
        ]

        print(df_geno_pheno_subset.shape)
        df_geno_pheno = df_geno_pheno_subset.reset_index(drop=True)
    
        X_all = create_X(df_geno_pheno_subset)
        X_sparse = sparse.COO(X_all)
        print('making it sparse')
        X = X_sparse
        print('making it dense')
        X = X_sparse.todense()
        print('X_sparse created will probably face error in train test split')
        y_all, y_array = rs_encoding_to_numeric(df_geno_pheno, DRUG)
        y_all = y_all.values.astype(np.int32)
        model._n_features = X.shape[1:]
        y_all_2 = y_all.reshape(-1,1)
        Y = y_all_2
    
    elif encoding_method.upper() == 'FCGR':
        print("Using FCGR data loader...")
        data_folder_dir = X_base_path.parent
        cgr_data_dir = data_folder_dir / "cgr_outputs"
        
        if not cgr_data_dir.is_dir():
            print(f"[ERROR] FCGR directory not found at: {cgr_data_dir}")
            return {"Error": f"FCGR directory not found at {cgr_data_dir}"}
        
        X_data_np = load_cgr_data(cgr_data_dir)
        X = pd.DataFrame(X_data_np).values
        print(f"Loaded FCGR data with shape: {X.shape}")
        
        ### NEW: ensure CNN_2D uses the correct FCGR image size ---
        if '2D' in model.name or '2d' in model.name.lower():
            n_features = X.shape[1]
            matrix_size = int(np.sqrt(n_features))
            if matrix_size * matrix_size != n_features:
                raise ValueError(
                    f"FCGR features ({n_features}) are not a perfect square; "
                    f"cannot infer a square image size. Got sqrt={matrix_size}."
                )
            
            # Update the internal input shape for both preprocessing and model creation
            model._n_features = (matrix_size, matrix_size)
            print(matrix_size)
            model.best_params = model.static_params  # refresh static_params with new shape
            
            print(f"Updated CNN_2D_MLiAMR input shape to: {model._n_features}")
        ### end NEW
            
    
    elif encoding_method.upper() in ['OHE', 'LE']:
        print(f"Loading base data from: {X_base_path}")
        X = pd.read_csv(X_base_path, header=0, index_col=None).values
        print(f"Loaded base data with shape: {X.shape}")
    else:
        # Standard feature-based data (no special encoding)
        print("Loading standard feature-based data...")
        X = pd.read_csv(X_base_path).values
        print(f"Loaded data with shape: {X.shape}")
    return X,Y

def gen_ohe(X_train_val,X_held_out_test,folder_name,drug,model):
    
        print("Applying One-Hot Encoding...")
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(X_train_val)
        
        # Save the encoder for 'run_test'
        encoder_filename = f'saved_models/{folder_name}_{drug}_OHE_encoder.pkl'
        os.makedirs('saved_models', exist_ok=True)
        with open(encoder_filename, 'wb') as f:
            pickle.dump(encoder, f)
        print(f"Saved OHE encoder to {encoder_filename}")

        feature_names = encoder.get_feature_names_out()
        
        X_train_val_np = encoder.transform(X_train_val)
        X_train_val = pd.DataFrame(X_train_val_np, columns=feature_names, index=range(X_train_val_np.shape[0]))
        
        X_held_out_test_np = encoder.transform(X_held_out_test)
        #X_held_out_test = pd.DataFrame(X_held_out_test_np, columns=feature_names, index=range(X_held_out_test_np.shape[0]))
        
        print(f"Data transformed. New feature count: {X_train_val.shape[1]}")

        # Update model's feature count for CNN models after OHE transformation
        if 'CNN' in model.name or 'cnn' in model.name.lower():
            model._n_features = X_train_val.shape[1]
            # Recreate static_params with updated feature count
            model.best_params = model.static_params
            print(f"Updated CNN model input shape to: {model._n_features}")

        return X_train_val_np, X_held_out_test_np

def _youden_threshold(model, val_X, val_y):
    """
    Selects an optimal classification threshold for
    a TRAINED model based on Youden’s J statistic.
    """
    # Use predict_proba if available (for models like Logistic Regression, RF),
    # otherwise fall back to decision_function (for models like SVM).
    
    try:
        # For sklearn and keras-based models with predict_proba
        val_scores = model.predict_proba(val_X)[:, 1]
    except Exception:
        try:
            # SVM and some other models use decision_function to get scores
            val_scores = model.decision_function(val_X)
        except Exception:
            try:
                # For SDCNN, we have a custom predict method that returns probabilities/scores
                val_scores = model.model.predict(val_X)
            except Exception:
                print("Using deepamr_model.predict instead...")
                val_scores = model.deepamr_model.predict(val_X,batch_size = model.batch_size)[0].flatten()
    
    y_pred = val_scores
    print(type(val_y))
    y_true = val_y

    num_samples = y_pred.shape[0]
    fpr_ = []
    tpr_ = []
    thresholds = np.linspace(0, 1, 101)
    num_resistant = np.sum(y_true == 1)
    num_sensitive = np.sum(y_true == 0)

    fpr_, tpr_ = [], []

    for thr in thresholds:
        tp = np.sum((y_pred >= thr) & (y_true == 1))
        fp = np.sum((y_pred >= thr) & (y_true == 0))

        tpr_.append(tp / float(num_resistant) if num_resistant > 0 else 0.0)
        fpr_.append(fp / float(num_sensitive) if num_sensitive > 0 else 0.0)

    fpr_ = np.array(fpr_)
    tpr_ = np.array(tpr_)

    J = tpr_ - fpr_

    return J

def _compute_metrics(model, X, y_true, thr):
    """
    Calculates a set of performance metrics for a TRAINED model using a given threshold.
    """
    
    try:
        # For sklearn and keras-based models with predict_proba
        scores = model.predict_proba(X)[:, 1]
    except Exception:
        try:
            # SVM and some other models use decision_function to get scores
            scores = model.decision_function(X)
        except Exception:
            try:
                # For SDCNN, we have a custom predict method that returns probabilities/scores
                scores = model.model.predict(X)
            except Exception:
                print("Using deepamr_model.predict instead...")
                scores = model.deepamr_model.predict(X, batch_size = model.batch_size)[0].flatten()

    # Apply the chosen threshold to the continuous scores to get binary predictions.
    preds = (scores > thr).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    # Handle cases where predictions are all positive or all negative,
    # which can result in a non-2x2 confusion matrix.
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if y_true.sum() == 0: # All true labels are negative.
            tn = cm[0,0]
        else: # All true labels are positive.
            tp = cm[0,0]

    # Calculate standard classification metrics.
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0.0        # Accuracy
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0       # Sensitivity (Recall)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0       # Specificity
    
    # ROC AUC and PR AUC are calculated on the continuous scores, not the binary predictions.
    # Use try-except blocks as these can fail if only one class is present in y_true.
    try:
        fpr, tpr, _ = metrics.roc_curve(y_true, scores)
        roc_auc = metrics.auc(fpr, tpr)
    except Exception: roc_auc = float('nan')
    try:
        pr, rc, _ = metrics.precision_recall_curve(y_true, scores)
        pr_auc = metrics.auc(rc, pr)
    except Exception: pr_auc = float('nan')
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0       # Precision
    f1 = (2 * prec * sen) / (prec + sen) if (prec + sen) > 0 else 0.0 # F1-Score
    # A list of metric names to be calculated and used as headers in the output file.
    return {"ACC" : acc, "Sen" : sen, "Spe": spe, "F1":f1, "ROC_AUC" : roc_auc, "PR_AUC":pr_auc}


def run_crossval(X : str , Y : str, model, dataset_name : str, encoding_method: str):
 
    # --- Load Data for the current drug ---
    drug = dataset_name
    X, Y = X_input_processing(X, Y, model, encoding_method, dataset_name)
    
    # --- Step 1: Initial 80/20 Split ---
    # The data is split ONCE into a main training/validation set (80%) and a
    # final held-out test set (20%). The test set is not used until the very
    # end to provide an unbiased estimate of the final model's performance.
  
    X_train_val, X_held_out_test, y_train_val, y_held_out_test = train_test_split(
        X, Y, test_size = 0.20, random_state = 42, stratify = Y
    )
    print(f"Initial data split: {len(y_train_val)} for CV, {len(y_held_out_test)} for final hold-out test.")

    if encoding_method.upper() == 'SDCNN':
        try:
            alpha_matrix = alpha_mat(y_train_val, df_geno_pheno)
        except Exception as e:
            print("Error occurred while calling alpha_mat:")

    if encoding_method.upper() == 'OHE':
        # need to check
        X_train_val, X_held_out_test = gen_ohe(X_train_val,X_held_out_test,drug,model)
         
    print(f"\n--- Model: {model.name} ---")
    # ======================================================================
    # This block determines whether to run hyperparameter tuning or use fixed parameters.
    # ======================================================================

    number_folds = 4
    
    outer_cv = StratifiedKFold(n_splits=number_folds, shuffle=True, random_state=42)
    best_hyperparams ={}
    
   
    if model.param_grid:
        # Path B: Run tuning if a parameter grid is provided.
        print("Hyperparameter tuning enabled. Running 4-fold cross-validation...")
        model.best_params = model.tune_hyperparams(X_train_val, y_train_val, outer_cv)
        
    elif model.static_params:
        # Path A: Skip tuning and use the pre-configured model directly.
        print("Using fixed hyperparameters. Skipping tuning.")

        ##### To be reviewed...
        if(model.name == "Treeresist"):
            model.gene_array=pd.read_csv("/data/public-data/TB_data_alldrugs/"+dataset_name+"/"+dataset_name+"_variants_with_genes.csv")["Gene"].tolist()
        model.best_params = model.static_params

    #Determine the best threshols
    candidate_threshold = {}
    Youden_statistic = np.zeros((101,number_folds))
    fold_metrics=[] 
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_train_val, y_train_val)):
        print(f"\n--- Processing Fold {fold+1}/4 ---")
 
        idx = np.concatenate((train_idx, test_idx))
        X_train_inner, y_train_inner = X_train_val[train_idx], y_train_val[train_idx]
        X_val_inner, y_val_inner = X_train_val[test_idx], y_train_val[test_idx]

        # SDCNN
        if(model.name == "SD_CNN"):
            model.reset_data(X_train_inner)
            trained_model = model
            history = trained_model.train(X_train_inner, alpha_matrix[train_idx, :], X_val_inner, alpha_matrix[test_idx, :])
        
        # DeepAMR
        elif(model.name == "DeepAMR"):
            trained_model = model
            # Convert to numpy array if needed (handles both DataFrame and ndarray)
            X_data = X_train_val.values if hasattr(X_train_val, 'values') else X_train_val
            y_data = y_train_val.values if hasattr(y_train_val, 'values') else y_train_val
            trained_model.reset_data(X_data, y_data)
            trained_model.data_prep(train_idx,test_idx)
            trained_model.train(Epochs = 20, Callbacks = None, lr = model.best_params['Learning rate'])

        # Other models (keras-based or sklearn-based)
        else:   
            trained_model = model.model.__class__(**model.best_params)
            trained_model.fit(X_train_inner, y_train_inner)
                
        # After training the model on the inner training set, we evaluate it
        # on the inner validation set to compute the Youden statistic across
        # a range of thresholds. This helps us identify the optimal threshold
        # for classification based on the trade-off between sensitivity and specificity.
        Youden_statistic[:,fold] = _youden_threshold(trained_model, X_val_inner, y_val_inner)

    # After processing all folds, we average the Youden statistic across folds
    # for each threshold and select the threshold that maximizes this average statistic.
    # This chosen threshold will be used for the final model evaluation on the held-out test set.
    avg_youden = np.mean(Youden_statistic, axis=1)
    best_index = np.argmax(avg_youden)
    thresholds = np.linspace(0, 1, 101)
    chosen_threshold = thresholds[best_index]

    print(f"Chosen threshold for final model: {chosen_threshold:.2f}")

    best_hyperparams["Threshold"] = chosen_threshold

    data_key = f"{model.name}_{dataset_name}_{encoding_method}"

    params_str = json.dumps(best_hyperparams, default=lambda x: x.item() if hasattr(x, "item") else x)
    
    if os.path.exists(PARAM_FILE):
        try:
            params_df = pd.read_csv(PARAM_FILE, sep = '\t')
        except pd.errors.EmptyDataError:
            params_df = pd.DataFrame(columns = ['key', 'params'])
    else:
        params_df = pd.DataFrame(columns = ['key', 'params'])

    if data_key in params_df['key'].values:
        print(f"Overwriting existing params for key: {data_key}")
        params_df.loc[params_df['key'] == data_key, 'params'] = params_str
    else:
        print(f"Adding new params for key: {data_key}")
        new_row = pd.DataFrame([{'key': data_key, 'params': params_str}])
        params_df = pd.concat([params_df, new_row], ignore_index = True)
        
    params_df.to_csv(PARAM_FILE, sep = '\t', index = False)
    print(f"Parameters saved to {PARAM_FILE}")

    # ======================================================================
    # The following steps are common to both paths (Tuning and Fixed).
    # ======================================================================

    # --- Step 4: Train the final model and select a threshold ---
    print("Training final model on a subset of the 80% data...")

    if(model.name == "SD_CNN"):
        model.reset_data(X_train_val)
        final_model=model
        history=final_model.train(X_train_val, alpha_matrix) #CNN
        cnn = final_model.model
        cnn.model.save('saved_models/'+ data_key + '_model.h5')
    else:
        try:
            final_model = model.model.__class__(**model.best_params)
            final_model.fit(X_train_val,y_train_val)
            model.save(final_model,data_key)

        except TypeError as e:
            final_model = model.fit(X_train_val,y_train_val)
            final_model.class_threshold = chosen_threshold
            final_model.deepamr_model.save_weights('./DeepAMR_weights/best_final_'+data_key+'_.weights.h5') 
            final_model.deepamr_model.save("./DeepAMR_weights/best_"+data_key+".keras")
    
    train_metrics = _compute_metrics(final_model, X_train_val, y_train_val, chosen_threshold)
    final_metrics = _compute_metrics(final_model, X_held_out_test, y_held_out_test,chosen_threshold)

    # Store the complete results for this model run in a dictionary.
    results_row = {
        "Model": data_key,
        "Best_Params": model.best_params,
        "Final_Threshold": chosen_threshold,
    }
    for metric_name , metric in train_metrics.items():
        results_row[f"Train_{metric_name}"] = metric
        print(f"  - Train {metric_name}: {metric:.4f}")

    for metric_name , metric in final_metrics.items():
        results_row[f"Final_{metric_name}"] = metric
        print(f"  - Final {metric_name}: {metric:.4f}")
    
    # Add the results of this run to the main list.
    return results_row

def run_test(X : str , Y : str, model, dataset_name : str, encoding_method: str):
 
    # --- Load Data for the current drug ---
    drug = dataset_name
    X,Y = X_input_processing(X, Y, model, encoding_method, dataset_name)


    if encoding_method.upper() == 'OHE':
        print("Applying One-Hot Encoding for test set...")
        encoder_filename = f'saved_models/{drug}_OHE_encoder.pkl'
        try:
            with open(encoder_filename, 'rb') as f:
                encoder = pickle.load(f)
            print(f"Loaded OHE encoder from {encoder_filename}")
        except FileNotFoundError:
            print(f"[ERROR] Encoder file not found: {encoder_filename}")
            return {"Error": "OHE Encoder not found."}
        
        X = encoder.transform(X)

    data_key = f"{model.name}_{dataset_name}_{encoding_method}"
    
    if not os.path.exists(PARAM_FILE):
        raise FileNotFoundError(f"{PARAM_FILE} not found")

    params_df = pd.read_csv(PARAM_FILE, sep='\t')
    row = params_df[params_df['key'] == data_key]

    if row.empty:
        raise ValueError(f"No parameters found for key: {data_key}")

    params = json.loads(row.iloc[0]['params'])
    
    threshold = params.get('Threshold')
    print(f"Loaded threshold: {threshold}")

    if(model.name == "SD_CNN"):  
        model.X = X

    trained_model = model.load(data_key)
    
    test_metrics = _compute_metrics(trained_model,X, Y, threshold)

    results_row = {
        "Model": data_key,
        "Dataset": dataset_name,
        "Threshold": threshold,
    }
    for metric_name , metric in test_metrics.items():
        results_row[f"Test_{metric_name}"] = metric
        print(f"  - Test {metric_name}: {metric:.4f}")

    return results_row
#!/usr/bin/env python
# coding: utf-8
"""
Runs SD-CNN training on cross-validation set
- assesses accuracy for each CV split
- saves table of loss per epoch

Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang-ho Yoon
"""

import sys
import glob
import os
import yaml
import sparse
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import random
from typing import Dict, Any, List
from sklearn.model_selection import GridSearchCV
from .Model import AbstractModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared')))
from models.tb_cnn_codebase import *

num_drugs = 1


os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def get_conv_nn(X):
    """
    Define convolutional neural network architecture.

    NB: filter_size is a global variable (int) given by the kwargs
    """

    model = models.Sequential()
    model.add(layers.Conv2D(
        64, (5, filter_size),
        data_format='channels_last',
        activation='relu',
        input_shape=X.shape[1:]
    ))
    model.add(layers.Conv2D(64, (1, 12), activation='relu', name='conv1d'))
    model.add(layers.MaxPooling2D((1, 3), name='max_pooling1d'))
    model.add(layers.Conv2D(32, (1, 3), activation='relu', name='conv1d_1'))
    model.add(layers.Conv2D(32, (1, 3), activation='relu', name='conv1d_2'))
    model.add(layers.MaxPooling2D((1, 3), name='max_pooling1d_1'))
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(256, activation='relu', name='d1'))
    model.add(layers.Dense(256, activation='relu', name='d2'))
    model.add(layers.Dense(1, activation='sigmoid', name='d4'))

    print(model.summary())

    opt = Adam(learning_rate=np.exp(-1.0 * 9))

    model.compile(
        optimizer=opt,
        loss=masked_multi_weighted_bce,
        metrics=[masked_weighted_accuracy]
    )

    return model


class myCNN:
        """
        Class for handling CNN functionality

        """
        def __init__(self,N_epochs,X):
            self.model = get_conv_nn(X)
            self.epochs = N_epochs
        

    

        def fit_model(self, X_train, y_train, X_val=None, y_val=None):
            """
            X_train: np.ndarray
                n_strains x 5 (one-hot) x longest locus length x no. of loci
                Genotypes of isolates used for training
            y_train: np.ndarray
                Labels for isolates used for training

            X_val: np.ndarray (optional, default=None)
                Optional genotypes of isolates in validation set

            y_val: np.ndarray (optional, default=None)
                Optional labels for isolates in validation set

            Returns
            -------
            pd.DataFrame:
                training history (accuracy, loss, validation accuracy, and validation loss) per epoch

            """
            if X_val is not None and y_val is not None:
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    validation_data=(X_val, y_val),
                    batch_size=128 #og 128 change it back for other drugs this  32 only for bdq
                )
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)
            else:
                
                history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=128)
                print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)

        def predict(self, X_val):
            """
            Returns
            -------
            predicted labels for given X
            """           
            return np.squeeze(self.model.predict(X_val))

'''
_, input_file = sys.argv

# load kwargs from config file (input_file)
kwargs = yaml.safe_load(open(input_file, "r"))
print(kwargs)
output_path = kwargs["output_path"]
N_epochs = kwargs["N_epochs"]
#filter_size = kwargs["filter_size"]
pkl_file = kwargs["pkl_file"]
DRUG = kwargs["drug"]
locus_list = kwargs["locus_list"]
'''

filter_size = 12
N_epochs=10

# --- MTB_SD_CNNManager Class ---



class MTB_SD_CNNManager(AbstractModel):
    """
    Manager for the SD-CNN model.
    Handles model instantiation and configuration consistent with AbstractModel.
    """

    def __init__(self, n ):
        """
        Parameters
        ----------
        X : np.ndarray
            Input tensor used to define CNN input shape.
        n_epochs : int
            Number of training epochs.
        """
        super().__init__()
        self.X = None
        self.N_epochs = N_epochs
        print("MTB_SD_CNN: Configuration initialized.")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "SD_CNN"

    @property
    #def model(self) -> myCNN:
    #    """
    #    Returns an instance of myCNN initialized with static parameters.
    #    """
    #    return myCNN(**self.static_params)

    def model(self) -> myCNN:
        if not hasattr(self, '_model') or self._model is None:
            self._model = myCNN(**self.static_params)
        return self._model

    @property
    def param_grid(self) -> Dict[str, Any]:
        """No hyperparameter tuning for SD-CNN."""
        return {}

    @property
    def static_params(self) -> Dict[str, Any]:
        """Fixed model parameters."""
        return {
            'N_epochs': self.N_epochs,
            'X': self.X
        }
    def reset_data(self,X):
        self.X=X
        print(self.X.shape)

    def load(self, dataset_name: str, model_name: str):
        """Load a trained model from disk."""
        import pickle
        filename = f'saved_models/{dataset_name}{model_name}_model.pkl'
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None) -> pd.DataFrame:
        """
        Convenience wrapper around fit_model for training.
        """
        cnn = self.model
        history = cnn.fit_model(X_train, y_train, X_val, y_val)
        #cnn.model.save("trained_model.h5") 
        return history
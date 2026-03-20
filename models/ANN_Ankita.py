# --- ANNManager Class ---

from sklearn.model_selection import GridSearchCV
import pickle
from .Model import AbstractModel
from scikeras.wrappers import KerasClassifier
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

def create_model(input_shape):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(input_shape,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

class ANN_AnkitaManager(AbstractModel):
    def __init__(self,n_features: int):
        self._n_features = n_features
        super().__init__()
        print("ANN_ankitaManager: Configuration initialized.")
        print("-" * 60)
        
    @property
    def name(self) -> str:
        return "ANN"

    @property
    def model(self) -> KerasClassifier:
        return KerasClassifier(
            **self.static_params  # Unpack the static params here
        )

    @property
    def param_grid(self) -> dict[str, any]:
        return {
        }

    @property
    def static_params(self) -> dict[str, any]:
        return {
            # KerasClassifier build-time parameters
            'model' : create_model,
            'model__input_shape': self._n_features,
            'optimizer': Adam,
            'loss': "binary_crossentropy",
            'metrics': ["accuracy"],
            'classes_': [0, 1],
            'batch_size': 32,
            'epochs': 50,
            'verbose': 0,
            'random_state': 42
        }

    def load(self,data_key):
        filename='saved_models/'+ data_key +'_model.pkl'
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model

    def tune_hyperparams(self, X, y, outer_cv):
        print(f"Tuning {self.name} using GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=outer_cv,
            scoring='average_precision',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_params = {**grid_search.best_params_, **self.static_params}
        print(f"{self.name} best hyperparams: {self.best_params}")
        return self.best_params

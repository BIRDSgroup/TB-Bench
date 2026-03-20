from keras.layers import Dense, Dropout, Input, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras import regularizers
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from keras.utils import to_categorical
import numpy as np
import os
import pdb
import pickle
import random


os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Configure TensorFlow to use CPU if GPU is not available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
import tensorflow as tf
try:
    # Try to use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        # Force CPU usage if no GPU found
        tf.config.set_visible_devices([], 'GPU')
        print("GPU not available. Using CPU for CNN_2D_MLiAMR.")
except Exception as e:
    print(f"GPU configuration error. Defaulting to CPU: {e}")


from .Model import AbstractModel 


# --- 1. Keras Model Definition ---
def get_cnn_2d_model(input_shape_tuple, **kwargs):
    """
    2D CNN architecture from ML-iAMR study.
    Used for FCGR (Frequency Chaos Game Representation) matrices.
    """
    n_classes = 2

    model = Sequential()
    model.add(Input(shape=(*input_shape_tuple, 1)))

    model.add(Conv2D(8, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(8, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))

    return model

class CustomKerasClassifier(KerasClassifier):
    def fit(self, X, y, **kwargs):
        # ---- custom logic BEFORE training ----
        X=X.reshape(X.shape[0], 200, 200, 1)
        y = to_categorical(y)

        return super().fit(X,y,**kwargs)

    def predict_proba(self, X):
        X=X.reshape(X.shape[0],200, 200, 1)
        probs = self.model_.predict(X, verbose=0)

        return probs


# --- 2. Manager Class ---
class CNN_2D_MLiAMRManager(AbstractModel):
    """
    2D CNN model manager for ML-iAMR method.
    Works with FCGR (Frequency Chaos Game Representation) encoded data.
    """

    def __init__(self, n_features):
        super().__init__()
        
        # For FCGR, n_features is the flattened size (e.g., 40000 for 200x200)
        # Infer the matrix dimensio0ns (assuming square matrix)
        if isinstance(n_features, int) and n_features > 0:
            matrix_size = int(np.sqrt(n_features))
            self._n_features = (matrix_size, matrix_size)
        elif isinstance(n_features, tuple):
            self._n_features = n_features
        else:
            # Default to 200x200 for FCGR
            self._n_features = (200, 200)
        self._model = None
        print("CNN_2D_MLiAMR: Configuration initialized.")
        print(f"  Input shape (FCGR matrix): {self._n_features}")
        print("-" * 60)
    

    @property
    def name(self) -> str:
        return "CNN_2D_MLiAMR"

    @property
    def model(self) -> KerasClassifier:
        if self._model is not None:
            return self._model
        
        self._model = CustomKerasClassifier(**self.static_params)
        return self._model

    @property
    def param_grid(self) -> dict[str, any]:
        # No hyperparameter tuning - use static params from ML-iAMR study
        return {}

    @property
    def static_params(self):
        # Architecture and parameters from ML-iAMR study
        return {
            'model': get_cnn_2d_model,
            'model__input_shape_tuple': self._n_features,
            'optimizer': 'adam',
            'loss': "categorical_crossentropy",
            'metrics': ["accuracy"],
            'batch_size': 32,
            'epochs': 50,
            'verbose': 0,
        }
    def load(self,data_key):
        filename='saved_models/'+data_key +'_model.pkl'
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model
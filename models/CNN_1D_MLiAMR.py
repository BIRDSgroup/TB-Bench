from keras.layers import Dense, Dropout, Input, BatchNormalization, Flatten, Conv1D, MaxPooling1D, Reshape
from keras.models import Model, Sequential
from keras import regularizers
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from keras.utils import to_categorical
import numpy as np
import pdb
import pickle
import os
import random

# Configure TensorFlow to use CPU if GPU is not available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
import tensorflow as tf


os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

try:
    # Try to use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        # Force CPU usage if no GPU found
        tf.config.set_visible_devices([], 'GPU')
        print("GPU not available. Using CPU for CNN_1D_MLiAMR.")
except Exception as e:
    print(f"GPU configuration error. Defaulting to CPU: {e}")

#from keras import backend as K
import tensorflow.keras.backend as K

from .Model import AbstractModel 

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class CustomKerasClassifier(KerasClassifier):
    def fit(self, X, y, **kwargs):
        # ---- custom logic BEFORE training ----
        X=X.reshape(X.shape[0], X.shape[1], 1)
        y = to_categorical(y)

        return super().fit(X,y,**kwargs)

    def predict_proba(self, X):
        X=X.reshape(X.shape[0], X.shape[1], 1)
        probs = self.model_.predict(X, verbose=0)

        return probs

# --- 1. Keras Model Definition ---
def get_cnn_1d_model(input_shape, **kwargs):
    """
    1D CNN architecture from ML-iAMR study.
    Used for LE/OHE encoded sequences.
    """
    n_classes = 2
    
    model = Sequential()
    # Use Input layer instead of input_shape parameter for proper shape handling
    model.add(Input(shape=(input_shape, 1)))
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(8, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(16, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    
    return model






class CNN_1D_MLiAMRManager(AbstractModel):
    """
    1D CNN model manager for ML-iAMR method.
    Works with LE (Label Encoding) or OHE (One-Hot Encoding) sequence data.
    """

    def __init__(self, n_features: int):
        super().__init__()
        
        self._n_features = n_features
        self._model = None
        
        print("CNN_1D_MLiAMR: Configuration initialized.")
        print("-" * 60)
    
    @property
    def name(self) -> str:
        return "CNN_1D_MLiAMR"

    @property
    def model(self) -> KerasClassifier:
        if self._model is not None:
            return self._model
        
        self._model = CustomKerasClassifier(**self.static_params)
        return self._model
    
    def create_model(self):
        """Create a new model instance with proper preprocessing bound."""
        return KerasClassifier(**self.static_params)

    @property
    def param_grid(self) -> dict[str, any]:
        # No hyperparameter tuning - use static params from ML-iAMR study
        return {}

    def load(self, data_key):
        filename = f"saved_models/{data_key}_model.pkl"
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model
    
    @property
    def static_params(self):
        # Architecture and parameters from ML-iAMR study
        return {
            'model': get_cnn_1d_model,
            'model__input_shape': self._n_features,
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'batch_size': 32,
            'epochs': 50,
            'verbose': 0,
        }
    def load(self,data_key):
        filename='saved_models/'+data_key +'_model.pkl'
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model
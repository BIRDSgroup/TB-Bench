from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.models import Model
from keras.layers import *
import keras.backend as K
from sklearn.model_selection import GridSearchCV
from keras import regularizers
from keras.layers import concatenate
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
import numpy as np
import pickle
from .Model import AbstractModel


def get_wdnn_model(input_shape):
    lambda_2=1e-8
    #lambda_2=1e-4
    input = Input(shape=(input_shape,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lambda_2))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lambda_2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(lambda_2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    wide_deep = concatenate([input, x])
    preds = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_2))(wide_deep)
    model = Model(inputs=input, outputs=preds)
    opt = Adam(learning_rate=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


class WDNNManager(AbstractModel):

    def __init__(self, n_features: int):
        super().__init__()
        self._n_features = n_features
        self._model_instance = None 
        print("WDNNManager: Configuration initialized.")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "WideAndDeepNN"

    @property

        
    def model(self) -> KerasClassifier:

        if self._model_instance is not None:
            return self._model_instance
        
        # Otherwise, create and store it
        self._model_instance = KerasClassifier(
            **self.static_params
        )
        return self._model_instance
    
        # Instantiate the model using the static parameters
        #model_instance = KerasClassifier(
        #    **self.static_params
        #)

        #model_instance._estimator_type = "classifier"
        
        #return model_instance

    @property
    def param_grid(self) -> dict:
        return {
        }

    
    @property
    def static_params(self) -> dict:
        return {
            # KerasClassifier build-time parameters
            'epochs': 100,
            'model' : get_wdnn_model,
            'model__input_shape': self._n_features,
            'optimizer': Adam,
            'loss': "binary_crossentropy",
            'metrics': ["accuracy"],
            'random_state':42,
            # KerasClassifier fit-time parameters
            'verbose': 0,
            #'class_weight': 'balanced'
        }
        
    def load(self,data_key):
        filename='saved_models/'+data_key +'_model.pkl'
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model
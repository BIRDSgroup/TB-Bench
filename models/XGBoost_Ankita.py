# --- XGBoostManager Class ---
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from .Model import AbstractModel
from scikeras.wrappers import KerasClassifier
import numpy as np

class XGBoost_AnkitaManager(AbstractModel):
    def __init__(self,n_features: int):
        self._n_features = n_features
        super().__init__()
        print("XGBoost_ankitaManager: Configuration initialized.")
        print("-" * 60)
        
    @property
    def name(self) -> str:
        return "XGBoost_ankita"

    @property
    def model(self) -> XGBClassifier:
        return XGBClassifier(**self.static_params)

    @property
    def param_grid(self) -> dict[str, any]:
        return {
        }

    @property
    def static_params(self) -> dict[str, any]:
        return {
            'learning_rate': 0.1,
            'tree_method': 'hist',
            'device': 'cpu',
            'random_state' : 42
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

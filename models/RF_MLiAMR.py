from sklearn.ensemble import RandomForestClassifier
import numpy as np

from .Model import AbstractModel


class RF_MLiAMRManager(AbstractModel):
    """
    ML-iAMR Random Forest model for sequence-encoded data (LE, OHE, or FCGR).
    """

    def __init__(self, n_features):
        super().__init__()
        self._n_features = n_features
        print(f"RF_MLiAMR: Initialized with {n_features} features")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "RF_MLiAMR"

    @property
    def model(self):
        return RandomForestClassifier(random_state=42, n_jobs=-1)

    @property
    def param_grid(self) -> dict:
        # No hyperparameter tuning - use static params from ML-iAMR study
        return {}

    @property
    def static_params(self):
        # Best parameters from ML-iAMR study (https://github.com/YunxiaoRen/ML-iAMR)
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }

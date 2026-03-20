from sklearn.linear_model import LogisticRegression
import numpy as np

from .Model import AbstractModel


class LR_MLiAMRManager(AbstractModel):
    """
    ML-iAMR Logistic Regression model for sequence-encoded data (LE, OHE, or FCGR).
    """

    def __init__(self, n_features):
        super().__init__()
        self._n_features = n_features
        print(f"LR_MLiAMR: Initialized with {n_features} features")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "LR_MLiAMR"

    @property
    def model(self):
        return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

    @property
    def param_grid(self) -> dict:
        # No hyperparameter tuning - use static params from ML-iAMR study
        return {}

    @property
    def static_params(self):
        # Best parameters from ML-iAMR study (https://github.com/YunxiaoRen/ML-iAMR)
        return {
            'C': 1,
            'penalty': 'l1',
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': 42
        }

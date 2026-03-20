from sklearn.svm import SVC
import numpy as np

from .Model import AbstractModel


class SVC_MLiAMRManager(AbstractModel):
    """
    ML-iAMR SVC model for sequence-encoded data (LE, OHE, or FCGR).
    """

    def __init__(self, n_features):
        super().__init__()
        self._n_features = n_features
        print(f"SVC_MLiAMR: Initialized with {n_features} features")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "SVC_MLiAMR"

    @property
    def model(self):
        return SVC(probability=True, random_state=42)

    @property
    def param_grid(self) -> dict:
        # No hyperparameter tuning - use static params from ML-iAMR study
        return {}

    @property
    def static_params(self):
        # Best parameters from ML-iAMR study (https://github.com/YunxiaoRen/ML-iAMR)
        return {
            'C': 1,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }

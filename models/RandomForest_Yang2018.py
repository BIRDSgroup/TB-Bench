# Random Forest — Yang et al. 2018

from __future__ import annotations
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from .Model import AbstractModel


class RandomForest_Yang2018Manager(AbstractModel):
    """
    The paper code calls RF via an ensemble wrapper; exact params not shown.
    This mirrors a standard RF setup; tune externally if needed.
    """
    def __init__(self, n: int):
        super().__init__()
        print("RandomForestManager: Configuration initialized.")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "RandomForest"

    @property
    def model(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.static_params)

    @property
    def param_grid(self) -> Dict[str, Any]:
        # Authors' grid not specified; keep empty.
        return {}

    @property
    def static_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "n_jobs": -1,
            "random_state": 42,
        }

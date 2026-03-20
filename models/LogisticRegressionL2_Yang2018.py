# Logistic Regression L2 — Yang et al. 2018

from __future__ import annotations
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from .Model import AbstractModel


class LogisticRegressionL2_Yang2018Manager(AbstractModel):
    def __init__(self, n: int):
        super().__init__()
        print("LogisticRegressionL2Manager: Configuration initialized.")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "LogisticRegression-L2"

    @property
    def model(self) -> LogisticRegression:
        return LogisticRegression(**self.static_params)

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {}

    @property
    def static_params(self) -> Dict[str, Any]:
        # penalty='l2' (default); pick stable solver and iterations.
        return {
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        }

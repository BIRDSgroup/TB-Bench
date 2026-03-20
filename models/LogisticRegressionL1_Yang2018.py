# Logistic Regression L1 — Yang et al. 2018

from __future__ import annotations
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from .Model import AbstractModel


class LogisticRegressionL1_Yang2018Manager(AbstractModel):
    def __init__(self, n: int):
        super().__init__()
        print("LogisticRegressionL1Manager: Configuration initialized.")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "LogisticRegression-L1"

    @property
    def model(self) -> LogisticRegression:
        return LogisticRegression(**self.static_params)

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {}

    @property
    def static_params(self) -> Dict[str, Any]:
        # Authors used penalty='l1'. We choose a compatible solver and bump max_iter.
        return {
            "penalty": "l1",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": 42,
        }

# SVC Linear Kernel — Yang et al. 2018

from __future__ import annotations
from typing import Dict, Any
from sklearn.svm import SVC
from .Model import AbstractModel


class SVCLinear_Yang2018Manager(AbstractModel):
    def __init__(self, n: int):
        super().__init__()
        print("SVCLinearManager: Configuration initialized.")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "SVC-Linear"

    @property
    def model(self) -> SVC:
        return SVC(**self.static_params)

    @property
    def param_grid(self) -> Dict[str, Any]:
        # Authors tuned C externally; we keep grid empty here.
        return {}

    @property
    def static_params(self) -> Dict[str, Any]:
        # As used by the authors: linear kernel + probability=True
        return {
            "kernel": "linear",
            "probability": True,
            "random_state": 42,
        }

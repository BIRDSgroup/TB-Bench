# SVC RBF Kernel — Yang et al. 2018

from __future__ import annotations
from typing import Dict, Any
from sklearn.svm import SVC
from .Model import AbstractModel


class SVCRBF_Yang2018Manager(AbstractModel):
    def __init__(self, n: int):
        super().__init__()
        print("SVCRBFManager: Configuration initialized.")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "SVC-RBF"

    @property
    def model(self) -> SVC:
        return SVC(**self.static_params)

    @property
    def param_grid(self) -> Dict[str, Any]:
        # Authors optimized C and gamma externally (optunity)
        return {}

    @property
    def static_params(self) -> Dict[str, Any]:
        # As used by the authors: rbf kernel + probability=True
        return {
            "kernel": "rbf",
            "probability": True,
            "random_state": 42,
        }

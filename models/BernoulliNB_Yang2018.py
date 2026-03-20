# Bernoulli NB — Yang et al. 2018

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from .Model import AbstractModel


def _beta_prior(
    y: np.ndarray,
    beta0: Tuple[float, float] = (0.25, 1.0),
    beta1: Tuple[float, float] = (1.0, 0.5),
) -> List[float]:
    """
    Mirror the author's Beta-prior construction:
      pw[yi] = (beta0[yi] + n_yi) / (beta0[yi] + beta1[yi] + len(y))
    Assumes binary labels {0,1}. If your labels are not {0,1}, map them first.
    """
    y = np.asarray(y).ravel()
    priors: List[float] = []
    for yi in (0, 1):
        n_yi = int((y == yi).sum())
        p = (beta0[yi] + n_yi) / (beta0[yi] + beta1[yi] + len(y))
        priors.append(p)
    return priors


class BernoulliNB_Yang2018Manager(AbstractModel):
    """
    Matches the behavior in Yang et al.-style code:
    - You can pass precomputed `class_prior` at construction time (dataset-level).
    - Or call `set_class_prior_from_labels(y_train)` per fold to mirror the paper exactly.
    """

    def __init__(
        self,
        n: int,
        class_prior: Optional[List[float]] = None,
        beta0: Tuple[float, float] = (0.25, 1.0),
        beta1: Tuple[float, float] = (1.0, 0.5),
    ):
        super().__init__()
        self._class_prior = class_prior
        self._beta0 = beta0
        self._beta1 = beta1
        print("BernoulliNB_Yang2018Manager: Configuration initialized.")
        if class_prior is not None:
            print(f"  Using externally provided class_prior: {class_prior}")
        print("-" * 60)

    @property
    def name(self) -> str:
        return "BernoulliNB"

    @property
    def model(self) -> BernoulliNB:
        return BernoulliNB(**self.static_params)

    @property
    def param_grid(self) -> Dict[str, Any]:
        # No grid in the author's path; tuning (if any) happens outside.
        return {}

    @property
    def static_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"alpha": 1.0}
        if self._class_prior is not None:
            params["class_prior"] = self._class_prior
            # When class_prior is provided, sklearn ignores fit_prior.
        return params

    # ---------- Optional: per-fold hook ----------
    def set_class_prior_from_labels(self, y_train: np.ndarray) -> None:
        """
        Call this at each CV fold with the **training** labels to mirror the paper exactly.
        """
        self._class_prior = _beta_prior(y_train, self._beta0, self._beta1)
        print(f"[{self.name}] Updated class_prior from fold: {self._class_prior}")

    # Convenience if you want to compute once from a full Y:
    def set_class_prior(self, class_prior: List[float]) -> None:
        self._class_prior = list(class_prior)

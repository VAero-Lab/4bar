"""
Normalized cross-configuration performance residual  ℛ(x₁, xᵢ).

This module computes the weighted, dimensionless measure of performance
distance between two designs from (possibly different) parameterization
families, evaluated across a grid of operating conditions.
"""

from __future__ import annotations

import numpy as np

from biopt.families import ParameterizationFamily


class IsoperformanceResidual:
    """Compute the isoperformance residual  ℛ(x₁, x₂).

    Parameters
    ----------
    family_ref : ParameterizationFamily
        Reference family (Family 1).
    family_target : ParameterizationFamily
        Target family (Family i).
    eval_params : list[ndarray]
        List of P evaluation-parameter vectors  [α⁽¹⁾, ..., α⁽ᴾ⁾].
    omega : ndarray of shape (P,), optional
        Evaluation-parameter weights (uniform if not given).
    w : ndarray of shape (m,), optional
        Performance-metric weights (uniform if not given).
    f_ref : ndarray of shape (m,), optional
        Reference performance values for normalization.  If ``None``,
        they are computed from ``family_ref`` at its bounds midpoint
        and the first evaluation parameter.
    """

    def __init__(
        self,
        family_ref: ParameterizationFamily,
        family_target: ParameterizationFamily,
        eval_params: list[np.ndarray],
        omega: np.ndarray | None = None,
        w: np.ndarray | None = None,
        f_ref: np.ndarray | None = None,
    ) -> None:
        self.family_ref = family_ref
        self.family_target = family_target
        self.eval_params = list(eval_params)
        self.P = len(self.eval_params)

        # Infer m from a probe evaluation
        x0 = family_ref.bounds.mean(axis=1)
        self._m = len(family_ref.performance(x0, self.eval_params[0]))

        # Weights
        if omega is None:
            self.omega = np.ones(self.P) / self.P
        else:
            omega = np.asarray(omega, dtype=float)
            self.omega = omega / omega.sum()

        if w is None:
            self.w = np.ones(self._m) / self._m
        else:
            w = np.asarray(w, dtype=float)
            self.w = w / w.sum()

        # Reference values for normalization
        if f_ref is None:
            f0 = family_ref.performance(x0, self.eval_params[0])
            self.f_ref = np.where(np.abs(f0) > 1e-30, np.abs(f0), 1.0)
        else:
            self.f_ref = np.asarray(f_ref, dtype=float)

    @property
    def m(self) -> int:
        """Number of performance metrics."""
        return self._m

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate  ℛ(x₁, x₂).

        Parameters
        ----------
        x1 : ndarray, shape (n1,)
            Reference design.
        x2 : ndarray, shape (n2,)
            Target design.

        Returns
        -------
        float
            Non-negative residual value.
        """
        total = 0.0
        for j, alpha in enumerate(self.eval_params):
            f1 = self.family_ref.performance(x1, alpha)
            f2 = self.family_target.performance(x2, alpha)
            diff = (f1 - f2) / self.f_ref
            total += self.omega[j] * np.sum(self.w * diff**2)
        return float(total)

    def with_eval_params(self, eval_params: list[np.ndarray]) -> "IsoperformanceResidual":
        """Return a new residual with a different evaluation grid.

        Preserves families, metric weights, and reference values.
        Recomputes evaluation-parameter weights as uniform.
        """
        return IsoperformanceResidual(
            family_ref=self.family_ref,
            family_target=self.family_target,
            eval_params=eval_params,
            omega=None,
            w=self.w,
            f_ref=self.f_ref,
        )

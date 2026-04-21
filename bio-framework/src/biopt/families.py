"""
Abstract base class for design parameterization families.

Each application case defines concrete subclasses that provide:
  - Parameter bounds and dimensionality.
  - A realization map φᵢ : Xᵢ → S  (parameters → shape space).
  - A performance map Fᵢ : Xᵢ × A → ℝᵐ  (parameters × eval params → performance).
  - Optionally, the Jacobian ∂Fᵢ/∂xᵢ for dimensional pre-analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DimensionalAnalysis:
    """Result of the dimensional pre-analysis for a family.

    Attributes
    ----------
    n : int
        Number of design parameters.
    m : int
        Number of independent performance constraints.
    predicted_dim : int
        Predicted manifold dimension (n - m).  Negative → over-determined.
    jacobian_rank : int | None
        Numerically evaluated rank of ∂F/∂x at a test point (None if not computed).
    """
    n: int
    m: int
    predicted_dim: int
    jacobian_rank: int | None = None


class ParameterizationFamily(ABC):
    """Abstract base class for a single parameterization family.

    Subclasses must implement :meth:`realize` and :meth:`performance`.
    """

    # ------------------------------------------------------------------
    # Properties that subclasses must define
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def n_params(self) -> int:
        """Dimensionality nᵢ of the parameter space Xᵢ."""

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """Parameter bounds, shape (n_params, 2).

        Each row is ``[lower, upper]`` for the corresponding parameter.
        """

    @property
    def name(self) -> str:
        """Human-readable name (defaults to class name)."""
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Maps that subclasses must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def realize(self, x: np.ndarray) -> np.ndarray:
        """Realization map  φᵢ(x) → s ∈ S.

        Parameters
        ----------
        x : ndarray, shape (n_params,)
            Design parameter vector.

        Returns
        -------
        s : ndarray
            Shape-space representation (discretized boundary, frequency
            response, coupler curve, etc.).
        """

    @abstractmethod
    def performance(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Parametric performance map  Fᵢ(x, α) → ℝᵐ.

        Parameters
        ----------
        x : ndarray, shape (n_params,)
            Design parameter vector.
        alpha : ndarray
            Single evaluation parameter (scalar or vector, domain-dependent).

        Returns
        -------
        f : ndarray, shape (m,)
            Performance vector.
        """

    # ------------------------------------------------------------------
    # Optional: Jacobian for dimensional pre-analysis
    # ------------------------------------------------------------------
    def jacobian(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray | None:
        """Jacobian  ∂Fᵢ/∂xᵢ  at (x, α), shape (m, n_params).

        Returns ``None`` if not implemented (will be estimated numerically).
        """
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def random_sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw a uniformly random feasible parameter vector."""
        rng = rng or np.random.default_rng()
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return rng.uniform(lo, hi)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check whether *x* lies within the parameter bounds."""
        return bool(np.all(x >= self.bounds[:, 0]) and np.all(x <= self.bounds[:, 1]))

    def dimensional_pre_analysis(
        self,
        m: int,
        x_test: np.ndarray | None = None,
        alpha_test: np.ndarray | None = None,
        eps: float = 1e-7,
    ) -> DimensionalAnalysis:
        """Perform the dimensional pre-analysis.

        Parameters
        ----------
        m : int
            Number of performance metrics being matched.
        x_test : ndarray, optional
            Point at which to evaluate the Jacobian rank.
        alpha_test : ndarray, optional
            Evaluation parameter for the Jacobian.
        eps : float
            Step size for finite-difference Jacobian (used only when
            :meth:`jacobian` returns ``None``).

        Returns
        -------
        DimensionalAnalysis
        """
        predicted_dim = self.n_params - m
        jac_rank = None

        if x_test is not None and alpha_test is not None:
            J = self.jacobian(x_test, alpha_test)
            if J is None:
                # Numerical finite-difference Jacobian
                f0 = self.performance(x_test, alpha_test)
                J = np.zeros((len(f0), self.n_params))
                for k in range(self.n_params):
                    xp = x_test.copy()
                    xp[k] += eps
                    J[:, k] = (self.performance(xp, alpha_test) - f0) / eps
            jac_rank = int(np.linalg.matrix_rank(J))

        return DimensionalAnalysis(
            n=self.n_params,
            m=m,
            predicted_dim=predicted_dim,
            jacobian_rank=jac_rank,
        )

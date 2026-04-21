"""
Pluggable optimizer interface.

The BIOpt framework is optimizer-agnostic.  This module defines the
abstract interface and provides two concrete implementations:

- **DEOptimizer**: Differential Evolution via ``scipy.optimize``.
- **CMAESOptimizer**: CMA-ES via the ``cma`` package (optional dependency).

All optimizers solve::

    min f(x)   subject to   x ∈ [lb, ub]

with an optional penalty callback for non-box constraints (exclusion zones).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution


@dataclass
class OptResult:
    """Result of an optimization run.

    Attributes
    ----------
    x : ndarray
        Best solution found.
    fun : float
        Objective value at ``x``.
    n_evals : int
        Total number of objective evaluations.
    success : bool
        Whether the optimizer converged.
    message : str
        Solver status message.
    """
    x: np.ndarray
    fun: float
    n_evals: int = 0
    success: bool = True
    message: str = ""


class Optimizer(ABC):
    """Abstract optimizer interface."""

    @abstractmethod
    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        penalty: Callable[[np.ndarray], float] | None = None,
    ) -> OptResult:
        """Minimize ``objective(x)`` subject to box bounds.

        Parameters
        ----------
        objective : callable
            Scalar objective  f(x) → float.
        bounds : ndarray, shape (n, 2)
            Box bounds  [lower, upper] per parameter.
        penalty : callable, optional
            Returns 0.0 if feasible, +∞ (or a large value) otherwise.
            Used for non-box constraints (exclusion zones).

        Returns
        -------
        OptResult
        """


class DEOptimizer(Optimizer):
    """Differential Evolution optimizer (SciPy).

    Parameters
    ----------
    maxiter : int
        Maximum number of generations.
    popsize : int
        Population size multiplier (actual population = popsize × n).
    tol : float
        Convergence tolerance on the objective.
    seed : int | None
        Random seed for reproducibility.
    strategy : str
        DE strategy string (e.g., ``'best1bin'``, ``'rand1bin'``).
    """

    def __init__(
        self,
        maxiter: int = 300,
        popsize: int = 15,
        tol: float = 1e-8,
        seed: int | None = None,
        strategy: str = "best1bin",
    ) -> None:
        self.maxiter = maxiter
        self.popsize = popsize
        self.tol = tol
        self.seed = seed
        self.strategy = strategy

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        penalty: Callable[[np.ndarray], float] | None = None,
    ) -> OptResult:
        n_evals = 0

        def _wrapped(x: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            if penalty is not None:
                p = penalty(x)
                if p > 0:
                    return 1e30  # death penalty
            return objective(x)

        scipy_bounds = list(zip(bounds[:, 0], bounds[:, 1]))
        res = differential_evolution(
            _wrapped,
            bounds=scipy_bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tol,
            seed=self.seed,
            strategy=self.strategy,
        )
        return OptResult(
            x=res.x,
            fun=res.fun if res.fun < 1e29 else np.inf,
            n_evals=n_evals,
            success=bool(res.success),
            message=res.message,
        )


class CMAESOptimizer(Optimizer):
    """CMA-ES optimizer (requires ``pip install cma``).

    Parameters
    ----------
    sigma0 : float
        Initial step size (fraction of the bound range).
    maxfevals : int
        Maximum number of function evaluations.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        sigma0: float = 0.3,
        maxfevals: int = 5000,
        seed: int | None = None,
    ) -> None:
        self.sigma0 = sigma0
        self.maxfevals = maxfevals
        self.seed = seed

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        penalty: Callable[[np.ndarray], float] | None = None,
    ) -> OptResult:
        try:
            import cma
        except ImportError as exc:
            raise ImportError(
                "CMA-ES requires the 'cma' package: pip install cma"
            ) from exc

        n_evals = 0

        def _wrapped(x: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            if penalty is not None:
                p = penalty(x)
                if p > 0:
                    return 1e30
            return objective(x)

        lb = bounds[:, 0]
        ub = bounds[:, 1]
        x0 = 0.5 * (lb + ub)
        sigma = self.sigma0 * np.mean(ub - lb)

        opts = {
            "bounds": [lb.tolist(), ub.tolist()],
            "maxfevals": self.maxfevals,
            "verbose": -9,  # silent
        }
        if self.seed is not None:
            opts["seed"] = self.seed

        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma, opts)
        es.optimize(_wrapped)
        res = es.result

        return OptResult(
            x=np.asarray(res.xbest),
            fun=float(res.fbest) if res.fbest < 1e29 else np.inf,
            n_evals=n_evals,
            success=not es.stop(),
            message=str(es.stop()),
        )

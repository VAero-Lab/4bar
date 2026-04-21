"""
Stringency analysis and δ-sensitivity sweeps.

- **Stringency**: How does ε* degrade as more operating conditions
  are enforced?  Produces the stringency curve ε*(P).
- **δ-sensitivity**: How does ε* change as the required geometric
  separation increases?  Produces the feasibility curve ε*(δ).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biopt.families import ParameterizationFamily
from biopt.residual import IsoperformanceResidual
from biopt.pair import PairProblem, PairResult
from biopt.optimizers import Optimizer, DEOptimizer


@dataclass
class StringencyResult:
    """Result of a stringency sweep.

    Attributes
    ----------
    P_values : list[int]
        Number of evaluation conditions at each sweep point.
    epsilon_star : list[float]
        Optimal residual at each P.
    critical_P : int | None
        Largest P for which ε* ≤ ε_tol.
    pair_results : list[PairResult]
        Full pair-problem results at each P.
    """
    P_values: list[int] = field(default_factory=list)
    epsilon_star: list[float] = field(default_factory=list)
    critical_P: int | None = None
    pair_results: list[PairResult] = field(default_factory=list)


@dataclass
class DeltaResult:
    """Result of a δ-sensitivity sweep.

    Attributes
    ----------
    delta_values : list[float]
        Separation thresholds tested.
    epsilon_star : list[float]
        Optimal residual at each δ.
    delta_max : float | None
        Largest δ for which ε* ≤ ε_tol.
    pair_results : list[PairResult]
        Full pair-problem results at each δ.
    """
    delta_values: list[float] = field(default_factory=list)
    epsilon_star: list[float] = field(default_factory=list)
    delta_max: float | None = None
    pair_results: list[PairResult] = field(default_factory=list)


def stringency_sweep(
    x1: np.ndarray,
    family_ref: ParameterizationFamily,
    family_target: ParameterizationFamily,
    eval_param_grids: list[list[np.ndarray]],
    delta: float,
    epsilon_tol: float,
    optimizer: Optimizer | None = None,
    **residual_kwargs,
) -> StringencyResult:
    """Run the stringency analysis.

    Parameters
    ----------
    x1 : ndarray
        Fixed reference design.
    family_ref, family_target : ParameterizationFamily
        Reference and target families.
    eval_param_grids : list of lists
        Nested evaluation grids  [A₁, A₂, ..., A_Pmax] where each Aₗ
        is a list of evaluation-parameter vectors of length Pₗ.
        Grids should be nested: A₁ ⊂ A₂ ⊂ ...
    delta : float
        Shape-space separation threshold.
    epsilon_tol : float
        Isoperformance tolerance.
    optimizer : Optimizer, optional
        Optimizer for the inner problem.
    **residual_kwargs
        Extra keyword arguments passed to ``IsoperformanceResidual``
        (e.g., ``w``, ``f_ref``).

    Returns
    -------
    StringencyResult
    """
    opt = optimizer or DEOptimizer()
    result = StringencyResult()
    critical = None

    for grid in eval_param_grids:
        P = len(grid)
        residual = IsoperformanceResidual(
            family_ref=family_ref,
            family_target=family_target,
            eval_params=grid,
            **residual_kwargs,
        )
        pair = PairProblem(
            family_ref=family_ref,
            family_target=family_target,
            residual=residual,
            delta=delta,
            epsilon_tol=epsilon_tol,
            optimizer=opt,
        )
        pr = pair.solve(x1)
        result.P_values.append(P)
        result.epsilon_star.append(pr.epsilon_star)
        result.pair_results.append(pr)
        if pr.feasible:
            critical = P

    result.critical_P = critical
    return result


def delta_sweep(
    x1: np.ndarray,
    family_ref: ParameterizationFamily,
    family_target: ParameterizationFamily,
    residual: IsoperformanceResidual,
    delta_values: np.ndarray | list[float],
    epsilon_tol: float,
    optimizer: Optimizer | None = None,
) -> DeltaResult:
    """Run the δ-sensitivity analysis.

    Parameters
    ----------
    x1 : ndarray
        Fixed reference design.
    family_ref, family_target : ParameterizationFamily
        Reference and target families.
    residual : IsoperformanceResidual
        Pre-configured residual.
    delta_values : array-like
        Sequence of δ values to test (should be increasing).
    epsilon_tol : float
        Isoperformance tolerance.
    optimizer : Optimizer, optional
        Optimizer for the inner problem.

    Returns
    -------
    DeltaResult
    """
    opt = optimizer or DEOptimizer()
    result = DeltaResult()
    d_max = None

    for d in delta_values:
        pair = PairProblem(
            family_ref=family_ref,
            family_target=family_target,
            residual=residual,
            delta=float(d),
            epsilon_tol=epsilon_tol,
            optimizer=opt,
        )
        pr = pair.solve(x1)
        result.delta_values.append(float(d))
        result.epsilon_star.append(pr.epsilon_star)
        result.pair_results.append(pr)
        if pr.feasible:
            d_max = float(d)

    result.delta_max = d_max
    return result

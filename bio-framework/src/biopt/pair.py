"""
Level A — The pair problem (fixed-reference isoperformance search).

Given a fixed reference design x₁ ∈ X₁, find the design x₂ ∈ X₂
that minimizes the performance residual ℛ(x₁, x₂) subject to
the shape-space exclusion constraint  dS(x₁, x₂) ≥ δ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biopt.families import ParameterizationFamily
from biopt.residual import IsoperformanceResidual
from biopt.distance import shape_distance, check_exclusion_zones
from biopt.optimizers import DEOptimizer, Optimizer, OptResult


@dataclass
class PairResult:
    """Result of a pair-problem solve.

    Attributes
    ----------
    x1 : ndarray
        Reference design (fixed input).
    x2 : ndarray
        Best target design found.
    epsilon_star : float
        Optimal residual  ε* = ℛ(x₁, x₂*).
    distance : float
        Shape-space distance  dS(x₁, x₂*).
    feasible : bool
        Whether  ε* ≤ ε_tol  (equivalence exists).
    opt_result : OptResult
        Raw optimizer output.
    """
    x1: np.ndarray
    x2: np.ndarray
    epsilon_star: float
    distance: float
    feasible: bool
    opt_result: OptResult


class PairProblem:
    """Level A solver: fixed-reference pair search.

    Parameters
    ----------
    family_ref : ParameterizationFamily
        Reference family (Family 1).
    family_target : ParameterizationFamily
        Target family (Family 2).
    residual : IsoperformanceResidual
        Pre-configured residual (carries eval params, weights, f_ref).
    delta : float
        Minimum shape-space separation.
    epsilon_tol : float
        Isoperformance tolerance.
    optimizer : Optimizer, optional
        Optimizer instance (defaults to DE).
    """

    def __init__(
        self,
        family_ref: ParameterizationFamily,
        family_target: ParameterizationFamily,
        residual: IsoperformanceResidual,
        delta: float,
        epsilon_tol: float,
        optimizer: Optimizer | None = None,
    ) -> None:
        self.family_ref = family_ref
        self.family_target = family_target
        self.residual = residual
        self.delta = delta
        self.epsilon_tol = epsilon_tol
        self.optimizer = optimizer or DEOptimizer()

    def solve(
        self,
        x1: np.ndarray,
        extra_exclusions: list[tuple[np.ndarray, ParameterizationFamily]] | None = None,
    ) -> PairResult:
        """Solve the pair problem for fixed reference x₁.

        Parameters
        ----------
        x1 : ndarray
            Fixed reference design in Family 1.
        extra_exclusions : list of (x, family) tuples, optional
            Additional exclusion-zone centers (for sequential sampling).

        Returns
        -------
        PairResult
        """
        # Build exclusion list: always exclude x1, plus any extras
        exc_centers = [x1]
        exc_families: list[ParameterizationFamily] = [self.family_ref]
        if extra_exclusions:
            for xc, fc in extra_exclusions:
                exc_centers.append(xc)
                exc_families.append(fc)

        def objective(x2: np.ndarray) -> float:
            return self.residual(x1, x2)

        def penalty(x2: np.ndarray) -> float:
            if not check_exclusion_zones(
                x2, self.family_target, exc_centers, exc_families, self.delta
            ):
                return 1.0  # triggers death penalty in optimizer
            return 0.0

        opt = self.optimizer.minimize(
            objective=objective,
            bounds=self.family_target.bounds,
            penalty=penalty,
        )

        dist = shape_distance(x1, opt.x, self.family_ref, self.family_target)
        eps = opt.fun if opt.fun < 1e29 else np.inf

        return PairResult(
            x1=x1,
            x2=opt.x,
            epsilon_star=eps,
            distance=dist,
            feasible=eps <= self.epsilon_tol,
            opt_result=opt,
        )

"""
Sequential manifold sampling (Algorithm 1 in the paper).

After the pair problem confirms existence of at least one equivalent,
this module explores the isoperformance set by repeatedly solving the
pair problem with cumulative exclusion zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biopt.families import ParameterizationFamily
from biopt.residual import IsoperformanceResidual
from biopt.pair import PairProblem, PairResult
from biopt.optimizers import Optimizer, DEOptimizer


@dataclass
class SamplingResult:
    """Result of sequential manifold sampling.

    Attributes
    ----------
    x1_star : ndarray
        Reference design (fixed).
    samples : list[ndarray]
        Accepted equivalent designs  Qᵢ = {x₂⁽¹⁾, ..., x₂⁽ᴿ'⁾}.
    residuals : list[float]
        Residual ℛ at each accepted sample.
    distances_from_ref : list[float]
        Shape-space distance from reference for each sample.
    n_accepted : int
        Number of accepted samples (= len(samples)).
    terminated_early : bool
        Whether sampling terminated before reaching R (infeasibility).
    pair_results : list[PairResult]
        Full pair-problem results for each iteration.
    """
    x1_star: np.ndarray
    samples: list[np.ndarray] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    distances_from_ref: list[float] = field(default_factory=list)
    n_accepted: int = 0
    terminated_early: bool = False
    pair_results: list[PairResult] = field(default_factory=list)

    def realized_shapes(self, family: ParameterizationFamily) -> np.ndarray:
        """Map all accepted samples to shape space for PCA.

        Returns
        -------
        ndarray, shape (n_accepted, dim_S)
        """
        return np.array([family.realize(x) for x in self.samples])


class ManifoldSampler:
    """Sequential manifold sampling within a target family.

    Parameters
    ----------
    family_ref : ParameterizationFamily
        Reference family.
    family_target : ParameterizationFamily
        Target family (searched for equivalents).
    residual : IsoperformanceResidual
        Pre-configured residual.
    delta : float
        Minimum shape-space separation (used for both reference
        exclusion and inter-sample exclusion).
    epsilon_tol : float
        Isoperformance tolerance.
    optimizer : Optimizer, optional
        Optimizer for each sub-problem.
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

        self._pair = PairProblem(
            family_ref=family_ref,
            family_target=family_target,
            residual=residual,
            delta=delta,
            epsilon_tol=epsilon_tol,
            optimizer=self.optimizer,
        )

    def sample(self, x1_star: np.ndarray, R: int = 20) -> SamplingResult:
        """Run sequential manifold sampling.

        Parameters
        ----------
        x1_star : ndarray
            Fixed reference design.
        R : int
            Maximum number of samples to attempt.

        Returns
        -------
        SamplingResult
        """
        result = SamplingResult(x1_star=x1_star)

        # Cumulative exclusion zones: start with reference
        extra_exclusions: list[tuple[np.ndarray, ParameterizationFamily]] = []

        for r in range(R):
            pair_res = self._pair.solve(x1_star, extra_exclusions=extra_exclusions)
            result.pair_results.append(pair_res)

            if pair_res.epsilon_star > self.epsilon_tol:
                result.terminated_early = True
                break

            result.samples.append(pair_res.x2.copy())
            result.residuals.append(pair_res.epsilon_star)
            result.distances_from_ref.append(pair_res.distance)
            result.n_accepted += 1

            # Add this sample to exclusion zones for subsequent iterations
            extra_exclusions.append((pair_res.x2.copy(), self.family_target))

        return result

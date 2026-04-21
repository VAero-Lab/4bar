"""
Level C — The bilevel isoperformance problem.

Upper level: optimize x₁ for performance subject to equivalence constraint.
Lower level: for each outer candidate x₁, find the best equivalent x₂.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from biopt.families import ParameterizationFamily
from biopt.residual import IsoperformanceResidual
from biopt.pair import PairProblem, PairResult
from biopt.optimizers import DEOptimizer, Optimizer, OptResult


@dataclass
class BilevelResult:
    """Result of the bilevel optimization.

    Attributes
    ----------
    x1_star : ndarray
        Optimal reference design (bilevel solution).
    x2_star : ndarray
        Corresponding equivalent in the target family.
    J_bio : float
        Outer objective at the bilevel optimum.
    J_unconstrained : float | None
        Outer objective without the equivalence constraint (for Δ_sub).
    delta_sub : float | None
        Substitutability cost  (J_bio - J_unc) / |J_unc|.
    epsilon_star : float
        Optimal residual at the bilevel solution.
    pair_result : PairResult
        Inner-problem result at the bilevel optimum.
    outer_opt : OptResult
        Raw outer optimizer output.
    n_inner_solves : int
        Total number of inner optimizations performed.
    """
    x1_star: np.ndarray
    x2_star: np.ndarray
    J_bio: float
    J_unconstrained: float | None
    delta_sub: float | None
    epsilon_star: float
    pair_result: PairResult
    outer_opt: OptResult
    n_inner_solves: int


class BilevelProblem:
    """Level C solver: bilevel isoperformance optimization.

    Parameters
    ----------
    family_ref : ParameterizationFamily
        Reference family.
    family_target : ParameterizationFamily
        Target family.
    residual : IsoperformanceResidual
        Pre-configured residual.
    outer_objective : callable
        J(x₁) → float.  Takes a reference parameter vector, returns
        the scalar objective to minimize.
    delta : float
        Minimum shape-space separation.
    epsilon_tol : float
        Isoperformance tolerance.
    outer_constraints : callable, optional
        G(x₁) → ndarray.  Returns a vector of constraint values;
        feasible when all components ≤ 0.
    outer_optimizer : Optimizer, optional
        Optimizer for the upper level.
    inner_optimizer : Optimizer, optional
        Optimizer for the lower level (pair problem).
    """

    def __init__(
        self,
        family_ref: ParameterizationFamily,
        family_target: ParameterizationFamily,
        residual: IsoperformanceResidual,
        outer_objective: Callable[[np.ndarray], float],
        delta: float,
        epsilon_tol: float,
        outer_constraints: Callable[[np.ndarray], np.ndarray] | None = None,
        outer_optimizer: Optimizer | None = None,
        inner_optimizer: Optimizer | None = None,
    ) -> None:
        self.family_ref = family_ref
        self.family_target = family_target
        self.residual = residual
        self.outer_objective = outer_objective
        self.delta = delta
        self.epsilon_tol = epsilon_tol
        self.outer_constraints = outer_constraints
        self.outer_optimizer = outer_optimizer or DEOptimizer()
        self.inner_optimizer = inner_optimizer or DEOptimizer()

        # Inner solver reused for every outer evaluation
        self._pair = PairProblem(
            family_ref=family_ref,
            family_target=family_target,
            residual=residual,
            delta=delta,
            epsilon_tol=epsilon_tol,
            optimizer=self.inner_optimizer,
        )

    def solve(self, compute_unconstrained: bool = True) -> BilevelResult:
        """Solve the bilevel problem.

        Parameters
        ----------
        compute_unconstrained : bool
            If ``True``, also solve the outer problem without the
            equivalence constraint to compute Δ_sub.

        Returns
        -------
        BilevelResult
        """
        n_inner = 0
        best_pair: PairResult | None = None

        def outer_obj_with_coupling(x1: np.ndarray) -> float:
            nonlocal n_inner, best_pair
            # Solve inner problem
            pair_res = self._pair.solve(x1)
            n_inner += 1

            # Death penalty if inner problem infeasible
            if pair_res.epsilon_star > self.epsilon_tol:
                return 1e30

            # Outer objective
            J = self.outer_objective(x1)

            # Cache the best pair result (minimum J among feasible)
            if best_pair is None or J < self.outer_objective(best_pair.x1):
                best_pair = pair_res

            return J

        def outer_penalty(x1: np.ndarray) -> float:
            if self.outer_constraints is not None:
                g = self.outer_constraints(x1)
                if np.any(g > 0):
                    return 1.0
            return 0.0

        # --- Bilevel solve ---
        outer_res = self.outer_optimizer.minimize(
            objective=outer_obj_with_coupling,
            bounds=self.family_ref.bounds,
            penalty=outer_penalty,
        )

        # Re-solve inner at the outer optimum to get the final pair
        final_pair = self._pair.solve(outer_res.x)
        J_bio = self.outer_objective(outer_res.x)

        # --- Unconstrained solve (without equivalence) ---
        J_unc = None
        delta_sub = None
        if compute_unconstrained:
            unc_res = self.outer_optimizer.minimize(
                objective=self.outer_objective,
                bounds=self.family_ref.bounds,
                penalty=outer_penalty,
            )
            J_unc = float(unc_res.fun)
            if abs(J_unc) > 1e-30:
                delta_sub = (J_bio - J_unc) / abs(J_unc)

        return BilevelResult(
            x1_star=outer_res.x,
            x2_star=final_pair.x2,
            J_bio=J_bio,
            J_unconstrained=J_unc,
            delta_sub=delta_sub,
            epsilon_star=final_pair.epsilon_star,
            pair_result=final_pair,
            outer_opt=outer_res,
            n_inner_solves=n_inner,
        )

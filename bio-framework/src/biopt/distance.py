"""
Shape-space distance  dS(x₁, x₂) and exclusion-zone checking.

The distance operates through the realization maps φᵢ, comparing
designs in the common shape space S regardless of their
parameterization families.
"""

from __future__ import annotations

import numpy as np

from biopt.families import ParameterizationFamily


def shape_distance(
    x1: np.ndarray,
    x2: np.ndarray,
    family1: ParameterizationFamily,
    family2: ParameterizationFamily,
) -> float:
    """Compute  dS(x₁, x₂) = ‖φ₁(x₁) − φ₂(x₂)‖ / √dim(S).

    The L² norm is root-mean-squared over all shape-space components,
    making it independent of the discretization resolution Ns.

    Parameters
    ----------
    x1, x2 : ndarray
        Parameter vectors in their respective families.
    family1, family2 : ParameterizationFamily
        Families providing the realization maps.

    Returns
    -------
    float
        Non-negative distance in shape space.
    """
    s1 = family1.realize(x1)
    s2 = family2.realize(x2)
    return float(np.sqrt(np.mean((s1 - s2) ** 2)))


def check_exclusion_zones(
    x_candidate: np.ndarray,
    family_candidate: ParameterizationFamily,
    exclusion_centers: list[np.ndarray],
    exclusion_families: list[ParameterizationFamily],
    delta: float,
) -> bool:
    """Check whether a candidate satisfies all exclusion constraints.

    Parameters
    ----------
    x_candidate : ndarray
        Candidate parameter vector.
    family_candidate : ParameterizationFamily
        Family of the candidate.
    exclusion_centers : list[ndarray]
        Parameter vectors defining exclusion-zone centers.
    exclusion_families : list[ParameterizationFamily]
        Families corresponding to each exclusion center.
    delta : float
        Minimum shape-space separation.

    Returns
    -------
    bool
        ``True`` if all exclusion constraints  dS ≥ δ  are satisfied.
    """
    for xc, fc in zip(exclusion_centers, exclusion_families):
        if shape_distance(x_candidate, xc, family_candidate, fc) < delta:
            return False
    return True

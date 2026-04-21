"""
BIOpt: Bilevel Isoperformance Optimization
==========================================

A field-agnostic framework for discovering equivalent designs
across parameterization families.

Modules
-------
families   : Abstract base class for parameterization families.
residual   : Isoperformance residual computation.
distance   : Shape-space distance and exclusion zones.
optimizers : Pluggable optimizer interface (DE, CMA-ES).
pair       : Level A pair problem solver.
bilevel    : Level C bilevel optimization.
sampling   : Sequential manifold sampling.
"""

from biopt.families import ParameterizationFamily
from biopt.residual import IsoperformanceResidual
from biopt.pair import PairProblem, PairResult
from biopt.bilevel import BilevelProblem, BilevelResult
from biopt.sampling import ManifoldSampler, SamplingResult
from biopt.optimizers import DEOptimizer, Optimizer

__version__ = "0.1.0"

__all__ = [
    "ParameterizationFamily",
    "IsoperformanceResidual",
    "PairProblem", "PairResult",
    "BilevelProblem", "BilevelResult",
    "ManifoldSampler", "SamplingResult",
    "DEOptimizer", "Optimizer",
]

"""
Roberts-Chebyshev Cognate Linkage Finder Package
=================================================

Package name: robchev  (short for Roberts-Chebyshev)

Main classes:
    FourBarLinkage   – kinematics
    CognateAnalyzer  – analytical cognate derivation
    CognateOptimizer – numerical optimization-based cognate finding
    Plotter          – static plots (snapshots, curves, parameters, opt-history)
    CognateAnimator  – animated all-three-cognates visualization
    OptimizationAnimator – animated optimization convergence + linkage evolution
"""

__version__ = "0.1.0"
__author__  = "Victor Alulema"

from .kinematics   import FourBarLinkage
from .analytical   import CognateAnalyzer
from .optimization import CognateOptimizer
from .visualization import (
    Plotter,
    CognateAnimator,
    OptimizationAnimator,
)

__all__ = [
    "FourBarLinkage",
    "CognateAnalyzer",
    "CognateOptimizer",
    "Plotter",
    "CognateAnimator",
    "OptimizationAnimator",
]

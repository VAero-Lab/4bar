# Roberts–Chebyshev Cognate Linkage Finder — Walkthrough

## Overview

Implemented a Python framework to identify the three cognate four-bar linkages for a given mechanism using the Roberts–Chebyshev theorem. The system combines **analytical focal-triangle construction** with **numerical global optimization** and produces publication-ready visualizations.

## Files Created

### [fourbar_kinematics.py](file:///home/victor-alulema/PhD%20Program/Paper%20Iso-performance/fourbar_kinematics.py)

Reusable `FourBarLinkage` class with:
- **Three vectorized sweep parameterizations** — crank (θ₂), coupler (θ₃), and rocker (θ₄) — each solving the loop-closure equation independently
- Scalar position solver (`solve_position`) and joint-position extractor
- Robust coupler-curve generator that combines all three sweeps
- Grashof classification utility

### [cognate_finder.py](file:///home/victor-alulema/PhD%20Program/Paper%20Iso-performance/cognate_finder.py)

Main script with four sections:
1. **Analytical cognate computation** — focal triangle similarity + Cayley diagram construction
2. **Numerical optimization** — `differential_evolution` with multi-seed restarts
3. **Visualization** — multi-snapshot linkage plots + coupler-curve overlays
4. **Main** — reference crank-rocker (a=2, b=5, c=4, d=6, p=3, α=30°)

## Key Design Decisions & Debugging

### The Assembly-Branch Problem

> [!IMPORTANT]
> The most critical discovery: **different sweep parameterizations select different assembly branches independently**. A single four-bar can produce TWO distinct coupler curves (open vs. crossed assembly). Each {crank, coupler, rocker} × {mode=+1, mode=−1} combination can land on a different branch.

**Naive approach (broken):** Combine all sweeps into one point cloud → contaminates the curve with points from both branches → optimizer stalls at 0.12–0.28 error.

**Solution:** Evaluate each of the **6 sweep variants** independently, compute bidirectional nearest-neighbor error for each, and return the minimum. This is implemented in `_best_curve_for_linkage()`.

### Why Three Sweeps Are Needed

| Cognate Type | Shortest Link | Full-rotation link | Required sweep |
|---|---|---|---|
| Cog 1 (crank-rocker) | L2 (crank) | L2 | **crank** sweep |
| Cog 2 (rocker-crank) | L4 (rocker) | L4 | **rocker** sweep |
| Cog 3 (double-rocker) | L3 (coupler) | L3 | **coupler** sweep |

Without the rocker sweep, Cognate 2's curve was always incomplete (only 94/360 valid angles from crank sweep).

### Multi-Seed Restarts

With a fixed seed, DE can get trapped in local minima (especially for Cognate 3, a Grashof double-rocker). The optimizer uses 5 restarts with diverse seeds (7, 24, 41, 58, 75) and early-stops when error < 1e-3.

## Results

### Parameter Comparison (Analytical vs. Optimised)

| Parameter | Cog2 Analytical | Cog2 Optimised | Cog3 Analytical | Cog3 Optimised |
|---|---|---|---|---|
| L2 (crank) | 2.8318 | 2.8234 | 3.0000 | 2.9997 |
| L3 (coupler) | 2.2655 | 2.2704 | 1.2000 | 1.2000 |
| L4 (rocker) | 1.1327 | 1.1333 | 2.4000 | 2.4003 |
| d_cp | 4.0000 | 4.0087 | 2.0000 | 1.9992 |
| α_cp (deg) | 31.98° | 31.81° | −30.00° | −30.10° |

### Curve-Matching Quality

| Linkage | RMSE | Hausdorff |
|---|---|---|
| Cognate 2 (analytical) | 4.48e-04 | 6.85e-04 |
| Cognate 2 (optimised) | 3.20e-03 | 5.42e-03 |
| Cognate 3 (analytical) | 1.47e-15 | 4.97e-15 |
| Cognate 3 (optimised) | 3.41e-03 | 4.01e-03 |

### Multi-Snapshot Visualization

![Cognate linkage snapshots at 8 crank angles](/home/victor-alulema/.gemini/antigravity/brain/f1c331a8-f190-4dd5-99a8-a1c6b6c26e7c/cognate_snapshots.png)

### Coupler-Curve Comparison

![Analytical vs optimised coupler curves for all three cognates](/home/victor-alulema/.gemini/antigravity/brain/f1c331a8-f190-4dd5-99a8-a1c6b6c26e7c/cognate_curves.png)

## Verification

- ✅ Both optimised cognates converge to error ~8.5e-05 (near discretization floor)
- ✅ Optimised link lengths match analytical values within 0.3%
- ✅ Coupler curves visually overlay for all three cognates
- ✅ All three cognate points meet at the same P at every crank angle (verified in snapshots)

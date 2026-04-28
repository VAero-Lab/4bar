#!/usr/bin/env python3
"""
Optimization Example 3 – Self-Verification: Recovering Cognate 1 (A₀, B₀)
===========================================================================

This is a **ground-truth verification** run. The optimizer is given the
same ground pivots (A₀, B₀) as the reference linkage and is asked to
find a linkage that traces the reference coupler curve.  A successful
run should recover the original link lengths  (a=2, b=5, c=4, p=3, α=30°)
up to numerical precision.

Reference linkage:
    O_A=(0,0), O_B=(6,0),  a=2, b=5, c=4, p=3, α=30°

Outputs (saved to  output/optimization_example_3/):
  - opt_cog1_final.png    – 3-panel: convergence | parameters | final geometry
  - opt_cog1_history.gif  – animated optimization history

Run
---
    python examples/optimization_example_3.py
"""

import numpy as np
import pathlib
import matplotlib; matplotlib.use('TkAgg')

from robchev import FourBarLinkage, CognateAnalyzer, CognateOptimizer
from robchev import Plotter, OptimizationAnimator

# ── Output directory ──────────────────────────────────────────────────────────
_SCRIPT = pathlib.Path(__file__).stem         # "optimization_example_3"
_ROOT   = pathlib.Path(__file__).parent.parent
OUT     = _ROOT / "output" / _SCRIPT
OUT.mkdir(parents=True, exist_ok=True)
def P(name): return str(OUT / name)

# ── 1. Reference linkage ──────────────────────────────────────────────────────
O_A = np.array([0.0, 0.0])
O_B = np.array([6.0, 0.0])

ref = FourBarLinkage(O1=O_A, O2=O_B, L2=2.0, L3=5.0, L4=4.0,
                     d_cp=3.0, alpha_cp=np.radians(30))

print("Reference linkage (ground truth):")
print(f"  {ref}")
print(f"  Grashof: {ref.grashof_info()}\n")

ref_curve = ref.generate_coupler_curve(720, mode=1)
print(f"Reference curve: {len(ref_curve)} points\n")

# ── 2. Optimize Cognate 1 – same pivots (A₀, B₀) ────────────────────────────
print("=" * 65)
print("  Optimising Cognate 1  (A₀, B₀) — self-recovery verification")
print("=" * 65)

opt1 = CognateOptimizer(
    ref_curve, O_A, O_B,
    grashof_target=None,        # crank-rocker search
    n_restarts=10,
    n_points_coarse=180,
    n_points_fine=1440,
    verbose=True,
)
opt1.run()

print(f"\n  Found linkage:\n    {opt1.linkage}")
print(f"  Best sweep : {opt1.best_sweep}")
print(f"  Final error: {opt1.error:.3e}\n")

# ── 3. Comparison against ground truth ───────────────────────────────────────
print("Parameter recovery check:")
params_gt  = dict(L2=2.0, L3=5.0, L4=4.0, d_cp=3.0, alpha_deg=30.0)
params_opt = dict(
    L2=opt1.linkage.L2, L3=opt1.linkage.L3, L4=opt1.linkage.L4,
    d_cp=opt1.linkage.d_cp,
    alpha_deg=np.degrees(opt1.linkage.alpha_cp),
)
sep = "─" * 52
print(sep)
print(f"  {'Parameter':<12}  {'Ground truth':>13}  {'Recovered':>10}  {'|Err|':>8}")
print(sep)
for key in params_gt:
    gt  = params_gt[key]
    opt = params_opt[key]
    pct = abs(gt - opt) / abs(gt) * 100 if abs(gt) > 1e-12 else 0.0
    print(f"  {key:<12}  {gt:>13.4f}  {opt:>10.4f}  {pct:>7.4f}%")
print(sep + "\n")

# ── 4. Static comparison plot ─────────────────────────────────────────────────
analyzer = CognateAnalyzer(ref)
O_C, cog2_ana, cog3_ana = analyzer.get_cognates()

Plotter.coupler_curves(analyzer, opt_cog2=cog2_ana, opt_cog3=cog3_ana,
                        save_path=P("cog1_coupler_curves.png"))

# ── 5. Optimization animation + PNG ──────────────────────────────────────────
print("Building optimization animation …")
oa = OptimizationAnimator(opt1, ref_curve=ref_curve,
                           title='Cognate 1 — Self-Recovery (A₀, B₀)', interval=50)
oa.build()
oa.save_final_png(P("opt_cog1_final.png"))
oa.save(P("opt_cog1_history.gif"), fps=15)
# oa.save(P("opt_cog1_history.mp4"), fps=20)   # uncomment for MP4

oa.show()

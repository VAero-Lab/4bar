#!/usr/bin/env python3
"""
Optimization Example 2 – Numerically Recovering Cognate 3  (A₀, C₀)
=====================================================================

Same reference linkage as Example 1.  Cognate 3 uses ground pivots
(A₀, C₀) and is a **Grashof double-rocker** (the coupler is the
shortest link), so `grashof_target='double-rocker'` is used.

Outputs (saved to  ./optimization_example_2/):
  - opt_coupler_curves.png
  - opt_parameter_comparison.png
  - opt_cog3_history.gif / .mp4
  - opt_cog3_final.png

Run
---
    python examples/optimization_example_2.py
"""

import numpy as np
import pathlib
import matplotlib; matplotlib.use('TkAgg')

from robchev import FourBarLinkage, CognateAnalyzer, CognateOptimizer
from robchev import Plotter, OptimizationAnimator

# ── Output directory ──────────────────────────────────────────────────────────
_SCRIPT = pathlib.Path(__file__).stem
_ROOT   = pathlib.Path(__file__).parent.parent
OUT     = _ROOT / "output" / _SCRIPT
OUT.mkdir(parents=True, exist_ok=True)
def P(name): return str(OUT / name)

# ── 1. Reference linkage + analytical cognates ───────────────────────────────
O_A = np.array([0.0, 0.0])
O_B = np.array([6.0, 0.0])

ref = FourBarLinkage(O1=O_A, O2=O_B, L2=2.0, L3=5.0, L4=4.0,
                     d_cp=3.0, alpha_cp=np.radians(30))
analyzer  = CognateAnalyzer(ref)
O_C, cog2_ana, cog3_ana = analyzer.get_cognates()

print(f"Reference linkage:\n  {ref}")
print(f"  Third pivot O_C = ({O_C[0]:.4f}, {O_C[1]:.4f})\n")
print(f"Analytical Cognate 3 (target):\n  {cog3_ana}\n")

ref_curve = ref.generate_coupler_curve(720, mode=1)

# ── 2. Optimize Cognate 3  (A₀, C₀) — double-rocker parameterisation ─────────
print("=" * 60)
print("  Optimising Cognate 3  (A₀, C₀)  [double-rocker]")
print("=" * 60)

opt3 = CognateOptimizer(ref_curve, O_A, O_C,
                         grashof_target='double-rocker',
                         n_restarts=10, n_points_coarse=180, n_points_fine=1440,
                         verbose=True)
opt3.run()

print(f"\n  Found linkage:\n    {opt3.linkage}")
print(f"  Best sweep : {opt3.best_sweep}")
print(f"  Final error: {opt3.error:.3e}\n")

# ── 3. Static comparison plots ────────────────────────────────────────────────
Plotter.coupler_curves(analyzer, opt_cog3=opt3.linkage,
                        save_path=P("opt_coupler_curves.png"))

Plotter.parameter_comparison(
    cognates_ana=[ref, cog2_ana, cog3_ana],
    cognates_opt=[ref, cog2_ana, opt3.linkage],
    save_path=P("opt_parameter_comparison.png"),
)

# ── 4. Optimization animation + PNG ──────────────────────────────────────────
print("Building optimization animation …")
oa = OptimizationAnimator(opt3, ref_curve=ref_curve,
                           title='Cognate 3 (A₀, C₀)', interval=50)
oa.build()
oa.save_final_png(P("opt_cog3_final.png"))
oa.save(P("opt_cog3_history.gif"), fps=15)
# oa.save(P("opt_cog3_history.mp4"), fps=20)

oa.show()

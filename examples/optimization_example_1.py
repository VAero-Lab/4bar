#!/usr/bin/env python3
"""
Optimization Example 1 – Numerically Recovering Cognate 2  (B₀, C₀)
=====================================================================

Reference linkage (same as Analytical Example 1):
    O_A=(0,0), O_B=(6,0),  a=2, b=5, c=4, p=3, α=30°

Outputs (saved to  ./optimization_example_1/):
  - opt_coupler_curves.png
  - opt_parameter_comparison.png
  - opt_cog2_history.gif / .mp4
  - opt_cog2_final.png

Run
---
    python examples/optimization_example_1.py
"""

import numpy as np
import pathlib
import matplotlib; matplotlib.use('TkAgg')  # interactive; change to 'Agg' for headless

from robchev import FourBarLinkage, CognateAnalyzer, CognateOptimizer
from robchev import Plotter, OptimizationAnimator

# ── Output directory (named after this script) ───────────────────────────────
# ── Output directory (named after this script) ───────────────────────────────
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

ref_curve = ref.generate_coupler_curve(720, mode=1)
print(f"Reference curve: {len(ref_curve)} points\n")

# ── 2. Optimize Cognate 2  (B₀, C₀) ─────────────────────────────────────────
print("=" * 60)
print("  Optimising Cognate 2  (B₀, C₀)")
print("=" * 60)

opt2 = CognateOptimizer(ref_curve, O_B, O_C,
                         n_restarts=5, n_points_coarse=180, n_points_fine=1440,
                         verbose=True)
opt2.run()

print(f"\n  Found linkage:\n    {opt2.linkage}")
print(f"  Best sweep : {opt2.best_sweep}")
print(f"  Final error: {opt2.error:.3e}\n")

# ── 3. Static comparison plots ────────────────────────────────────────────────
Plotter.coupler_curves(analyzer, opt_cog2=opt2.linkage,
                        save_path=P("opt_coupler_curves.png"))

Plotter.parameter_comparison(
    cognates_ana=[ref, cog2_ana, cog3_ana],
    cognates_opt=[ref, opt2.linkage, cog3_ana],
    save_path=P("opt_parameter_comparison.png"),
)

# ── 4. Optimization animation + PNG ──────────────────────────────────────────
print("Building optimization animation …")
oa = OptimizationAnimator(opt2, ref_curve=ref_curve,
                           title='Cognate 2 (B₀, C₀)', interval=50)
oa.build()
oa.save_final_png(P("opt_cog2_final.png"))
oa.save(P("opt_cog2_history.gif"), fps=15)   # always save GIF
# Uncomment for MP4 (requires ffmpeg):
# oa.save(P("opt_cog2_history.mp4"), fps=20)

oa.show()

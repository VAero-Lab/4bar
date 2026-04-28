#!/usr/bin/env python3
"""
Analytical Example 1 – Standard Crank-Rocker
=============================================

Reference linkage (original from the paper):
    O_A = (0, 0),  O_B = (6, 0)
    a = 2  (crank),  b = 5  (coupler),  c = 4  (rocker)
    p = 3,  α = 30°  (coupler-point definition)

This example demonstrates:
  1. Constructing a reference FourBarLinkage.
  2. Analytically deriving the two cognate linkages with CognateAnalyzer.
  3. Plotting the multi-snapshot figure (static PNG).
  4. Generating and saving an animated figure (.gif).

Outputs (saved to ./analytical_example_1/):
  - cognate_snapshots.png
  - cognate_animation.gif  (also .mp4 if ffmpeg is available)

Run
---
    python examples/analytical_example_1.py
"""

import numpy as np
import pathlib
import matplotlib; matplotlib.use('TkAgg')

from robchev import FourBarLinkage, CognateAnalyzer
from robchev import Plotter, CognateAnimator

# ── Output directory ──────────────────────────────────────────────────────────
_SCRIPT = pathlib.Path(__file__).stem           # e.g. "analytical_example_1"
_ROOT   = pathlib.Path(__file__).parent.parent  # repo root
OUT     = _ROOT / "output" / _SCRIPT
OUT.mkdir(parents=True, exist_ok=True)
def P(name): return str(OUT / name)

# ── 1. Reference linkage ──────────────────────────────────────────────────────
O_A = np.array([0.0, 0.0])
O_B = np.array([6.0, 0.0])

ref = FourBarLinkage(
    O1=O_A, O2=O_B,
    L2=2.0,          # crank
    L3=5.0,          # coupler
    L4=4.0,          # rocker
    d_cp=3.0,        # |AP|
    alpha_cp=np.radians(30),
)
print("Reference linkage:")
print(f"  {ref}")
print(f"  Grashof: {ref.grashof_info()}\n")

# ── 2. Analytical cognates ────────────────────────────────────────────────────
analyzer = CognateAnalyzer(ref)
O_C, cog2, cog3 = analyzer.get_cognates()

print(f"  Third pivot O_C = ({O_C[0]:.4f}, {O_C[1]:.4f})")
print(f"\nCognate 2 (B₀-C₀):\n  {cog2}")
print(f"Cognate 3 (A₀-C₀):\n  {cog3}")

# ── 3. Static multi-snapshot plot ─────────────────────────────────────────────
fig_snap = Plotter.snapshot(analyzer, n_snapshots=8,
                             save_path=P("cognate_snapshots.png"))

# ── 4. Animated cognates ──────────────────────────────────────────────────────
anim = CognateAnimator(analyzer, n_frames=120, interval=40)
anim.build()
anim.save(P("cognate_animation.gif"), fps=20)    # always save GIF
# anim.save(P("cognate_animation.mp4"), fps=24)  # uncomment for MP4
anim.show()

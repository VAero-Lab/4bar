#!/usr/bin/env python3
"""
Analytical Example 2 – Grashof Double-Rocker
=============================================

This linkage uses the *coupler* as the shortest link, making it a
Grashof double-rocker (neither input nor output link can rotate fully,
but the coupler can).

    O_A = (0, 0),  O_B = (8, 0)
    a = 6  (input rocker),  b = 3  (coupler — shortest),
    c = 5  (output rocker)
    p = 2.5,  α = 50°

Outputs (saved to ./analytical_example_2/):
  - cognate_snapshots.png
  - cognate_animation.gif  (also .mp4 if ffmpeg is available)

Run
---
    python examples/analytical_example_2.py
"""

import numpy as np
import pathlib
import matplotlib; matplotlib.use('TkAgg')

from robchev import FourBarLinkage, CognateAnalyzer
from robchev import Plotter, CognateAnimator

# ── Output directory ──────────────────────────────────────────────────────────
_SCRIPT = pathlib.Path(__file__).stem
_ROOT   = pathlib.Path(__file__).parent.parent
OUT     = _ROOT / "output" / _SCRIPT
OUT.mkdir(parents=True, exist_ok=True)
def P(name): return str(OUT / name)

# ── 1. Reference linkage (Grashof double-rocker) ──────────────────────────────
O_A = np.array([0.0, 0.0])
O_B = np.array([8.0, 0.0])

ref = FourBarLinkage(
    O1=O_A, O2=O_B,
    L2=6.0,          # input rocker
    L3=3.0,          # coupler (shortest → Grashof double-rocker)
    L4=5.0,          # output rocker
    d_cp=2.5,        # |AP|
    alpha_cp=np.radians(50),
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
# For a double-rocker, coupler-sweep is used; snapshot uses the
# Cayley parallelogram construction which works for any Grashof type.
fig_snap = Plotter.snapshot(analyzer, n_snapshots=8,
                             save_path=P("cognate_snapshots.png"))

# ── 4. Animated cognates ──────────────────────────────────────────────────────
anim = CognateAnimator(analyzer, n_frames=150, interval=35)
anim.build()
anim.save(P("cognate_animation.gif"), fps=20)
# anim.save(P("cognate_animation.mp4"), fps=24)
anim.show()

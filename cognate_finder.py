#!/usr/bin/env python3
"""
Roberts–Chebyshev Cognate Linkage Finder
=========================================

Given a reference four-bar linkage (Cognate 1), this script:

1. **Analytically** computes the two cognate linkages and their ground
   pivots using the Cayley-diagram / focal-triangle construction.
2. **Numerically** recovers each cognate via ``scipy.optimize.differential_evolution``
   — only the ground-pivot coordinates are assumed known; link lengths and
   coupler-point parameters are treated as unknowns.
3. Compares analytical vs. optimised results and generates multi-snapshot
   and coupler-curve overlay plots.

Theory (Roberts–Chebyshev Theorem)
----------------------------------
For any planar four-bar linkage tracing a coupler curve, there exist two
additional four-bar linkages (cognates) whose coupler points trace the
**same** curve.  The three sets of ground pivots form a "focal triangle"
similar to the coupler triangle of the original mechanism.

Reference example
-----------------
    O_A = (0, 0),  O_B = (6, 0)
    a = 2 (crank), b = 5 (coupler), c = 4 (rocker)
    p = 3, α = 30° (coupler-point definition)

Author: Generated for Victor Alulema's PhD research
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import time

from fourbar_kinematics import FourBarLinkage


# =====================================================================
#  1.  ANALYTICAL COGNATE COMPUTATION
# =====================================================================

def compute_coupler_triangle(linkage):
    """
    Compute the coupler-triangle parameters from a FourBarLinkage.

    The coupler triangle A-B-P has:
        |AP| = p = linkage.d_cp
        |AB| = b = linkage.L3
        ∠PAB = α = linkage.alpha_cp  (CCW from A→B to A→P)

    Returns
    -------
    p, q, alpha, beta  where
        q = |BP|     (via law of cosines)
        β = angle at B  (from B→A direction to B→P direction)
    """
    p = linkage.d_cp
    b = linkage.L3
    alpha = linkage.alpha_cp

    # |BP|² = p² + b² - 2pb cos α   (law of cosines in △ABP)
    q = np.sqrt(p ** 2 + b ** 2 - 2 * p * b * np.cos(alpha))

    # β via the complex-number relation  1/(1 − λe^{iα})
    # where λ = p/b.  The argument of that expression equals β.
    lam = p / b
    beta = np.arctan2(lam * np.sin(alpha), 1.0 - lam * np.cos(alpha))
    return p, q, alpha, beta


def compute_third_pivot(O_A, O_B, p, b, alpha):
    """
    Compute the third ground pivot O_C of the focal triangle.

    The focal triangle  (O_A, O_B, O_C) is similar to the coupler
    triangle  (A, B, P).  From the similarity:

        O_C = O_A + (O_B − O_A) · (p/b) · e^{iα}
    """
    d_vec = O_B - O_A
    lam = p / b
    ca, sa = np.cos(alpha), np.sin(alpha)
    rotated = np.array([d_vec[0] * ca - d_vec[1] * sa,
                        d_vec[0] * sa + d_vec[1] * ca])
    return O_A + lam * rotated


def compute_analytical_cognates(ref_linkage):
    """
    Derive the two cognate linkages analytically.

    Labelling convention (matching the user's request):
        Cognate 1 : ground  (A0, B0)  — the reference linkage.
        Cognate 2 : ground  (B0, C0)
        Cognate 3 : ground  (A0, C0)

    Returns
    -------
    O_C : ndarray (2,)
        Third ground pivot.
    cognate2 : FourBarLinkage
        Ground pivots (O_B, O_C).
    cognate3 : FourBarLinkage
        Ground pivots (O_A, O_C).
    """
    O_A, O_B = ref_linkage.O1, ref_linkage.O2
    a = ref_linkage.L2   # crank
    b = ref_linkage.L3   # coupler
    c = ref_linkage.L4   # rocker

    p, q, alpha, beta = compute_coupler_triangle(ref_linkage)
    O_C = compute_third_pivot(O_A, O_B, p, b, alpha)

    # --- Cognate 2  (O_B, O_C)  — from Cayley-diagram Cognate III ---
    #   crank at O_B  :  q
    #   coupler        :  c·q / b
    #   rocker at O_C  :  a·q / b
    #   coupler-point  :  d = c,  angle = β  (from B₃→C₃ to B₃→P)
    cognate2 = FourBarLinkage(
        O1=O_B, O2=O_C,
        L2=q,
        L3=c * q / b,
        L4=a * q / b,
        d_cp=c,
        alpha_cp=beta,
    )

    # --- Cognate 3  (O_A, O_C)  — from Cayley-diagram Cognate II ---
    #   crank at O_A  :  p
    #   coupler        :  a·p / b
    #   rocker at O_C  :  c·p / b
    #   coupler-point  :  d = a,  angle = −α  (reflected triangle)
    cognate3 = FourBarLinkage(
        O1=O_A, O2=O_C,
        L2=p,
        L3=a * p / b,
        L4=c * p / b,
        d_cp=a,
        alpha_cp=-alpha,
    )

    return O_C, cognate2, cognate3


def compute_all_joints_analytical(ref_linkage, theta2, O_C):
    """
    Given the reference linkage at crank angle θ₂, compute joint positions
    of **all three** cognates using the Cayley-diagram parallelogram /
    similar-triangle construction (no separate position solving needed).

    Returns
    -------
    dict  with keys  'cognate1', 'cognate2', 'cognate3',
    each mapping to a dict with keys  O1, A, B, O2, P.
    Returns None if the reference cannot assemble at this θ₂.
    """
    joints = ref_linkage.get_all_joints(theta2, mode=1)
    if joints is None:
        return None

    O_A = joints['O1']
    O_B = joints['O2']
    A   = joints['A']
    B   = joints['B']
    P   = joints['P']

    p, _, alpha, _ = compute_coupler_triangle(ref_linkage)
    b = ref_linkage.L3
    lam = p / b
    ca, sa = np.cos(alpha), np.sin(alpha)

    def _rotate(v):
        return np.array([v[0] * ca - v[1] * sa,
                         v[0] * sa + v[1] * ca])

    # Cognate 3 (O_A, O_C): Cayley prlgm  O_A-A-P-A₂
    A2 = O_A + (P - A)                         # parallelogram vertex
    C2 = A2 + lam * _rotate(A - O_A)           # similar triangle

    # Cognate 2 (O_B, O_C): Cayley prlgm  O_B-B-P-B₃
    B3 = O_B + (P - B)
    C3 = P + lam * _rotate(O_B - B)            # similar triangle

    return {
        'cognate1': dict(O1=O_A, A=A, B=B, O2=O_B, P=P),
        'cognate2': dict(O1=O_B, A=B3, B=C3, O2=O_C, P=P),
        'cognate3': dict(O1=O_A, A=A2, B=C2, O2=O_C, P=P),
    }


# =====================================================================
#  2.  NUMERICAL OPTIMISATION
# =====================================================================

def _best_curve_for_linkage(linkage, n_points, ref_curve, ref_tree,
                            sweep_types=('crank', 'coupler', 'rocker')):
    """
    Try specified sweep variants and return the curve with the lowest 
    bidirectional error.
    """
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    best_error = 1e10
    best_curve = np.empty((0, 2))
    best_name = ""

    for mode in [1, -1]:
        sweeps = []
        if 'crank' in sweep_types:
            sweeps.append(("crank", linkage._crank_sweep_vec(thetas, mode)))
        if 'coupler' in sweep_types:
            sweeps.append(("coupler", linkage._coupler_sweep_vec(thetas, mode)))
        if 'rocker' in sweep_types:
            sweeps.append(("rocker", linkage._rocker_sweep_vec(thetas, mode)))

        for sweep_name, (pts, _, _, _) in sweeps:
            if len(pts) < 15:
                continue
            cand_tree = KDTree(pts)
            d_fwd, _ = cand_tree.query(ref_curve)
            d_bwd, _ = ref_tree.query(pts)
            err = np.mean(d_fwd ** 2) + np.mean(d_bwd ** 2)
            if err < best_error:
                best_error = err
                best_curve = pts
                best_name = f"{sweep_name}/mode={mode:+d}"

    return best_curve, best_error, best_name


def _coupler_curve_error(params, O1, O2, ref_curve, ref_tree, n_points=360,
                         sweep_types=('crank', 'coupler', 'rocker'),
                         is_double_rocker_params=False):
    """
    Bidirectional nearest-neighbour curve-matching error.
    """
    if is_double_rocker_params:
        # Parameterised to guarantee L3 is the shortest link
        L3, delta2, delta4, d_cp, alpha_cp = params
        L2 = L3 + delta2
        L4 = L3 + delta4
    else:
        L2, L3, L4, d_cp, alpha_cp = params

    if L2 <= 0 or L3 <= 0 or L4 <= 0 or d_cp <= 0:
        return 1e10

    try:
        linkage = FourBarLinkage(O1, O2, L2, L3, L4, d_cp, alpha_cp)
    except Exception:
        return 1e10

    _, best_error, _ = _best_curve_for_linkage(
        linkage, n_points=n_points, ref_curve=ref_curve, ref_tree=ref_tree,
        sweep_types=sweep_types
    )
    return best_error


def optimize_cognate(ref_curve, O1, O2, verbose=True, n_restarts=10,
                     n_points_coarse=180, n_points_fine=1440,
                     grashof_target=None):
    """
    Find a cognate linkage whose coupler curve matches *ref_curve*
    using a **two-phase** strategy:

      Phase 1 — Differential Evolution (global, coarse resolution)
      Phase 2 — L-BFGS-B local refinement (fine resolution)
    """
    O1, O2 = np.asarray(O1, float), np.asarray(O2, float)
    ref_tree = KDTree(ref_curve)
    L1 = np.linalg.norm(O2 - O1)

    # ── Physically-motivated bounds ──
    d_O1 = np.linalg.norm(ref_curve - O1, axis=1)
    d_O2 = np.linalg.norm(ref_curve - O2, axis=1)
    max_reach = max(d_O1.max(), d_O2.max())
    curve_span = np.ptp(ref_curve, axis=0)
    curve_diag = np.linalg.norm(curve_span)

    link_ub = max(max_reach, curve_diag, L1) * 1.5
    link_lb = 0.05 * min(L1, 1.0)

    if grashof_target == 'double-rocker':
        # Constraint: L3 must be the shortest link.
        # Params: [L3, delta2, delta4, d_cp, alpha_cp]
        bounds = [
            (link_lb, min(L1, link_ub)),  # L3 (shortest)
            (0.0, link_ub),               # delta2
            (0.0, link_ub),               # delta4
            (link_lb, link_ub),           # d_cp
            (-np.pi, np.pi),             # alpha_cp
        ]
        sweep_types = ('coupler',)      # Crank sweeps fail for double-rockers
        is_dr = True
    else:
        # Standard search
        bounds = [
            (link_lb, link_ub),           # L2
            (link_lb, link_ub),           # L3
            (link_lb, link_ub),           # L4
            (link_lb, link_ub),           # d_cp
            (-np.pi, np.pi),             # alpha_cp
        ]
        sweep_types = ('crank', 'coupler', 'rocker')
        is_dr = False

    if verbose:
        mode_str = " (Double-Rocker parameterisation)" if is_dr else ""
        print(f"  Bounds: links ∈ [{link_lb:.3f}, {link_ub:.3f}]  "
              f"(L1={L1:.3f}){mode_str}")

    t0 = time.time()
    best_result = None
    best_history_x_de = []
    best_history_err_de = []

    # ── Phase 1: Coarse global search (Differential Evolution) ──
    for restart in range(n_restarts):
        seed = restart * 13 + 3
        if verbose:
            print(f"  Phase 1  [DE restart {restart+1}/{n_restarts}, seed={seed}]")

        tracker_x_de = []
        tracker_err_de = []
        def cb_de(xk, convergence=None):
            tracker_x_de.append(xk.copy())
            # For callback we just record the error at coarse points
            err = _coupler_curve_error(xk, O1, O2, ref_curve, ref_tree, n_points_coarse, sweep_types, is_dr)
            tracker_err_de.append(err)

        result = differential_evolution(
            _coupler_curve_error,
            bounds=bounds,
            args=(O1, O2, ref_curve, ref_tree, n_points_coarse, sweep_types, is_dr),
            maxiter=300,
            popsize=30,
            init='sobol',
            tol=1e-10,
            atol=1e-10,
            seed=seed,
            polish=False,
            disp=False,
            callback=cb_de,
        )

        if verbose:
            print(f"    → error = {result.fun:.3e}  ({result.nit} iters)")

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_history_x_de = list(tracker_x_de)
            best_history_err_de = list(tracker_err_de)

        if best_result.fun < 1e-3:
            if verbose:
                print(f"    ✓ Good basin found, proceeding to Phase 2.")
            break

    t_phase1 = time.time() - t0
    if verbose:
        print(f"  Phase 1 done in {t_phase1:.1f} s  |  coarse error = {best_result.fun:.3e}")

    # ── Phase 2: Fine local refinement (L-BFGS-B) ──
    if verbose:
        print(f"  Phase 2  (L-BFGS-B, {n_points_fine} pts per sweep)")

    history_x_phase2 = []
    history_err_phase2 = []
    def cb_min(xk):
        history_x_phase2.append(xk.copy())
        err = _coupler_curve_error(xk, O1, O2, ref_curve, ref_tree, n_points_fine, sweep_types, is_dr)
        history_err_phase2.append(err)

    from scipy.optimize import minimize
    fine_result = minimize(
        _coupler_curve_error,
        x0=best_result.x,
        args=(O1, O2, ref_curve, ref_tree, n_points_fine, sweep_types, is_dr),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12},
        callback=cb_min,
    )

    t_total = time.time() - t0
    if verbose:
        print(f"  Phase 2 done  |  fine error = {fine_result.fun:.3e}")
        print(f"  Total time: {t_total:.1f} s")

    # Reconstruct standard parameters
    if is_dr:
        L3, delta2, delta4, d_cp, alpha_cp = fine_result.x
        L2, L4 = L3 + delta2, L3 + delta4
    else:
        L2, L3, L4, d_cp, alpha_cp = fine_result.x

    linkage = FourBarLinkage(O1, O2, L2, L3, L4, d_cp, alpha_cp)

    _, final_error, best_sweep = _best_curve_for_linkage(
        linkage, n_points=n_points_fine, ref_curve=ref_curve, ref_tree=ref_tree,
        sweep_types=sweep_types
    )

    history = {
        'phase1_x': np.array(best_history_x_de),
        'phase1_err': np.array(best_history_err_de),
        'phase2_x': np.array(history_x_phase2),
        'phase2_err': np.array(history_err_phase2),
        'is_dr': is_dr
    }

    return linkage, final_error, best_sweep, history


# =====================================================================
#  3.  VISUALISATION
# =====================================================================

# Colour palette
COLORS = {
    'cog1': {'link': '#2563EB', 'curve': '#3B82F6', 'fill': '#93C5FD',
             'label': 'Cognate 1 (A₀-B₀)'},
    'cog2': {'link': '#DC2626', 'curve': '#EF4444', 'fill': '#FCA5A5',
             'label': 'Cognate 2 (B₀-C₀)'},
    'cog3': {'link': '#16A34A', 'curve': '#22C55E', 'fill': '#86EFAC',
             'label': 'Cognate 3 (A₀-C₀)'},
}


def _draw_linkage(ax, joints, color_key, lw=2.2, zorder=3):
    """Draw a single four-bar linkage on *ax*."""
    c = COLORS[color_key]
    O1, A, B, O2, P = joints['O1'], joints['A'], joints['B'], joints['O2'], joints['P']

    # Links
    for seg in [(O1, A), (A, B), (B, O2)]:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                '-', color=c['link'], lw=lw, solid_capstyle='round', zorder=zorder)

    # Coupler triangle fill (A-B-P)
    tri = plt.Polygon([A, B, P], alpha=0.4, color=c['fill'], zorder=zorder - 1)
    ax.add_patch(tri)

    # Joints (small circles)
    for pt in [A, B, P]:
        ax.plot(*pt, 'o', color=c['link'], ms=4.5, mec='white', mew=0.6, zorder=zorder + 1)


def _draw_ground_pivots(ax, O_A, O_B, O_C, ms=9):
    """Draw triangular ground-pivot markers."""
    for pt, label in [(O_A, 'A₀'), (O_B, 'B₀'), (O_C, 'C₀')]:
        ax.plot(*pt, '^', color='#1E293B', ms=ms, mec='white', mew=1.0, zorder=10)
        ax.annotate(label, pt, textcoords='offset points', xytext=(6, 6),
                    fontsize=8, fontweight='bold', color='#1E293B')


def plot_multi_snapshot(ref_linkage, O_C, n_snapshots=8, save_path=None):
    """
    Create a multi-panel figure showing all three cognates at several
    time steps (crank-angle values).

    All joint positions are computed **analytically** from the reference.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=130)
    fig.suptitle('Roberts–Chebyshev Cognate Linkages — Instantaneous Configurations',
                 fontsize=15, fontweight='bold', y=0.97)

    thetas = np.linspace(0, 2 * np.pi, n_snapshots, endpoint=False)

    # Generate reference curve once for background
    ref_curve = ref_linkage.generate_coupler_curve(720, mode=1)

    for idx, (ax, t2) in enumerate(zip(axes.flat, thetas)):
        all_j = compute_all_joints_analytical(ref_linkage, t2, O_C)
        if all_j is None:
            ax.set_visible(False)
            continue

        # Background: coupler curve (light gray)
        ax.plot(ref_curve[:, 0], ref_curve[:, 1], '-', color='#CBD5E1',
                lw=1.0, zorder=1)

        # Draw each cognate
        for key in ['cognate1', 'cognate2', 'cognate3']:
            _draw_linkage(ax, all_j[key], key.replace('nate', ''))

        # Ground pivots
        _draw_ground_pivots(ax, ref_linkage.O1, ref_linkage.O2, O_C, ms=7)

        # Shared coupler point
        P = all_j['cognate1']['P']
        ax.plot(*P, 'o', color='black', ms=7, mec='white', mew=1.2, zorder=12)

        # Focal triangle (dashed)
        tri = plt.Polygon([ref_linkage.O1, ref_linkage.O2, O_C],
                          fill=False, edgecolor='#94A3B8', ls='--', lw=0.8, zorder=0)
        ax.add_patch(tri)

        ax.set_title(f'θ₂ = {np.degrees(t2):.0f}°', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)

    # Shared legend
    handles = [Line2D([0], [0], color=COLORS[f'cog{i}']['link'], lw=2.5,
                      label=COLORS[f'cog{i}']['label']) for i in [1, 2, 3]]
    handles.append(Line2D([0], [0], marker='o', color='black', lw=0, ms=7,
                          mec='white', mew=1.2, label='Coupler point P'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
              frameon=True, fancybox=True, shadow=True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"  Saved snapshot figure → {save_path}")
    return fig


def plot_coupler_curves(ref_linkage, cog2_ana, cog3_ana,
                        cog2_opt, cog3_opt,
                        save_path=None):
    """
    Plot coupler-curve comparison: analytical vs. optimised cognates.
    Uses _best_curve_for_linkage to select the correct assembly branch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=130)
    fig.suptitle('Coupler-Curve Comparison — Analytical vs. Optimised Cognates',
                 fontsize=14, fontweight='bold', y=1.00)

    ref_curve = ref_linkage.generate_coupler_curve(720, mode=1)
    ref_tree = KDTree(ref_curve)

    titles = ['Cognate 1 (reference)', 'Cognate 2 (B₀, C₀)', 'Cognate 3 (A₀, C₀)']
    ana_linkages = [ref_linkage, cog2_ana, cog3_ana]
    opt_linkages = [None, cog2_opt, cog3_opt]

    for i, ax in enumerate(axes):
        # Reference curve in gray
        ax.plot(ref_curve[:, 0], ref_curve[:, 1], '-', color='#94A3B8',
                lw=4.5, alpha=0.45, label='Reference curve', zorder=1)

        # Analytical — use best sweep variant
        c_ana, _, _ = _best_curve_for_linkage(
            ana_linkages[i], 720, ref_curve, ref_tree)
        if len(c_ana) > 0:
            ax.plot(c_ana[:, 0], c_ana[:, 1], '--',
                    color=COLORS[f'cog{i+1}']['link'], lw=2.0,
                    label='Analytical', zorder=3)

        # Optimised — use best sweep variant
        if opt_linkages[i] is not None:
            c_opt, _, _ = _best_curve_for_linkage(
                opt_linkages[i], 720, ref_curve, ref_tree)
            if len(c_opt) > 0:
                ax.plot(c_opt[:, 0], c_opt[:, 1], ':',
                        color=COLORS[f'cog{i+1}']['curve'], lw=2.5,
                        label='Optimised', zorder=2)

        ax.set_title(titles[i], fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"  Saved curve figure → {save_path}")
    return fig


def print_comparison_table(ref_linkage, cognates_ana, cognates_opt, errors_opt):
    """
    Print a formatted comparison of analytical vs. optimised parameters
    for all three cognates (Cognate 1 = reference for verification).
    """
    labels = ['Cog 1 (A₀,B₀)', 'Cog 2 (B₀,C₀)', 'Cog 3 (A₀,C₀)']
    sep = '─' * 110
    print(f"\n{sep}")
    print("  COGNATE PARAMETER COMPARISON — Analytical vs. Optimised")
    print(sep)

    header = f"  {'Parameter':<14}"
    for lbl in labels:
        header += f"  {'Ana':>8}  {'Opt':>8}  {'Err%':>7}  │"
    print(header)
    print(sep)

    param_names = ['L2 (crank)', 'L3 (coupler)', 'L4 (rocker)', 'd_cp', 'α_cp (deg)', 'L1 (ground)']

    for pname in param_names:
        row = f"  {pname:<14}"
        for i in range(3):
            ana, opt = cognates_ana[i], cognates_opt[i]
            if pname == 'L2 (crank)':    a, o = ana.L2, opt.L2
            elif pname == 'L3 (coupler)': a, o = ana.L3, opt.L3
            elif pname == 'L4 (rocker)':  a, o = ana.L4, opt.L4
            elif pname == 'd_cp':         a, o = ana.d_cp, opt.d_cp
            elif pname == 'α_cp (deg)':   a, o = np.degrees(ana.alpha_cp), np.degrees(opt.alpha_cp)
            else:                         a, o = ana.L1, opt.L1
            pct = abs(a - o) / abs(a) * 100 if abs(a) > 1e-12 else 0.0
            row += f"  {a:>8.4f}  {o:>8.4f}  {pct:>6.3f}%  │"
        print(row)
    print(sep)

    # Curve-matching errors
    ref_curve = ref_linkage.generate_coupler_curve(720, mode=1)
    ref_tree = KDTree(ref_curve)

    for i, lbl in enumerate(labels):
        for tag, lnk in [("analytical", cognates_ana[i]), ("optimised", cognates_opt[i])]:
            c, err, sweep = _best_curve_for_linkage(lnk, 1440, ref_curve, ref_tree)
            if len(c) < 5:
                print(f"  {lbl} ({tag}): curve too short")
                continue
            ct = KDTree(c)
            d1, _ = ct.query(ref_curve)
            hausdorff = d1.max()
            rmse = np.sqrt(np.mean(d1 ** 2))
            print(f"  {lbl} {tag:<12}  RMSE = {rmse:.2e}   Hausdorff = {hausdorff:.2e}   [{sweep}]")

    print(f"\n  Objective function errors (fine resolution):")
    for i, lbl in enumerate(labels):
        print(f"    {lbl}:  {errors_opt[i]:.3e}")
    print(sep + "\n")


def plot_parameter_comparison(cognates_ana, cognates_opt, save_path=None):
    """
    Grouped bar chart comparing analytical vs. optimised link parameters
    for all three cognates, with percent-error annotations.
    """
    labels = ['Cognate 1\n(A₀, B₀)', 'Cognate 2\n(B₀, C₀)', 'Cognate 3\n(A₀, C₀)']
    param_names = ['L2', 'L3', 'L4', 'd_cp', 'α_cp']

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=130)
    fig.suptitle('Parameter Comparison — Analytical vs. Optimised',
                 fontsize=14, fontweight='bold', y=1.02)

    bar_colors_ana = ['#2563EB', '#DC2626', '#16A34A']
    bar_colors_opt = ['#93C5FD', '#FCA5A5', '#86EFAC']

    x = np.arange(3)
    width = 0.35

    for j, (ax, pname) in enumerate(zip(axes, param_names)):
        ana_vals, opt_vals, pct_errs = [], [], []
        for i in range(3):
            a, o = cognates_ana[i], cognates_opt[i]
            if pname == 'L2':    av, ov = a.L2, o.L2
            elif pname == 'L3':  av, ov = a.L3, o.L3
            elif pname == 'L4':  av, ov = a.L4, o.L4
            elif pname == 'd_cp': av, ov = a.d_cp, o.d_cp
            else:  # α_cp
                av, ov = np.degrees(a.alpha_cp), np.degrees(o.alpha_cp)
            ana_vals.append(av)
            opt_vals.append(ov)
            pct_errs.append(abs(av - ov) / abs(av) * 100 if abs(av) > 1e-12 else 0.0)

        bars1 = ax.bar(x - width/2, ana_vals, width, label='Analytical',
                       color=bar_colors_ana, edgecolor='white', linewidth=0.8)
        bars2 = ax.bar(x + width/2, opt_vals, width, label='Optimised',
                       color=bar_colors_opt, edgecolor='white', linewidth=0.8)

        # Annotate with percent error
        for k, (b, pct) in enumerate(zip(bars2, pct_errs)):
            ax.annotate(f'{pct:.3f}%',
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7,
                        color='#374151', fontweight='bold')

        unit = ' (deg)' if pname == 'α_cp' else ''
        ax.set_title(f'{pname}{unit}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
        if j == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"  Saved parameter comparison → {save_path}")
    return fig


def plot_optimization_history(histories, labels, save_path=None):
    """
    Plot the convergence of the _coupler_curve_error and the link parameters
    for each cognate over the optimization iterations.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=130)
    fig.suptitle('Optimization Convergence History', fontsize=15, fontweight='bold', y=0.97)

    for hist, ax1, label in zip(histories, axes, labels):
        if hist is None:
            continue
            
        # Combine phase 1 and phase 2 histories
        err1 = hist['phase1_err']
        err2 = hist['phase2_err']
        x1 = hist['phase1_x']
        x2 = hist['phase2_x']
        
        # Total iterations
        iters1 = np.arange(1, len(err1) + 1)
        iters2 = np.arange(len(err1) + 1, len(err1) + len(err2) + 1)
        
        ax1.set_title(label, fontsize=12, fontweight='bold')
        ax1.set_xlabel('Iteration (Gen for DE, Iter for L-BFGS-B)', fontsize=10)
        ax1.set_ylabel('Coupler Curve Error', color='tab:red', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_yscale('log')
        
        # Plot errors
        if len(err1) > 0:
            ax1.plot(iters1, err1, 'r-', marker='.', markersize=4, alpha=0.7, label='Phase 1 (DE) Error')
        if len(err2) > 0:
            ax1.plot(iters2, err2, 'r--', marker='.', markersize=4, alpha=0.7, label='Phase 2 (L-BFGS-B) Error')
            
        # Vertical line indicating phase transition
        if len(err1) > 0 and len(err2) > 0:
            ax1.axvline(x=len(err1), color='gray', linestyle=':', alpha=0.5)
            
        ax1.grid(True, alpha=0.3)
        
        # Second Y axis for parameters
        ax2 = ax1.twinx()
        ax2.set_ylabel('Parameter Value', color='tab:blue', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        
        # Convert x to standard parameters: L2, L3, L4, d_cp, alpha_cp
        params_all = []
        is_dr = hist['is_dr']
        for x in list(x1) + list(x2):
            if is_dr:
                params_all.append([x[0] + x[1], x[0], x[0] + x[2], x[3], x[4]])
            else:
                params_all.append(list(x))
        
        if len(params_all) > 0:
            params_all = np.array(params_all)
            iters_all = np.arange(1, len(params_all) + 1)
            
            # Plot each parameter
            param_names = ['L2', 'L3', 'L4', 'd_cp', 'α_cp (rad)']
            colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']
            
            for j, pname in enumerate(param_names):
                ax2.plot(iters_all, params_all[:, j], color=colors[j], alpha=0.7, label=pname)
                
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"  Saved optimization history → {save_path}")
    return fig


# =====================================================================
#  4.  MAIN SCRIPT
# =====================================================================

def main():
    print("=" * 72)
    print("  ROBERTS–CHEBYSHEV  COGNATE  LINKAGE  FINDER")
    print("=" * 72)

    # ------------------------------------------------------------------
    #  Reference four-bar  (Cognate 1)
    # ------------------------------------------------------------------
    O_A = np.array([0.0, 0.0])
    O_B = np.array([6.0, 0.0])
    a   = 2.0          # crank
    b   = 5.0          # coupler
    c   = 4.0          # rocker
    p   = 3.0          # |AP|  coupler-point distance
    alpha = np.pi / 6  # 30°   coupler-point angle

    ref = FourBarLinkage(O_A, O_B, L2=a, L3=b, L4=c, d_cp=p, alpha_cp=alpha)
    print(f"\nReference linkage:\n  {ref}")
    print(f"  Grashof: {ref.grashof_info()}")

    # ------------------------------------------------------------------
    #  Analytical cognates
    # ------------------------------------------------------------------
    print("\n── Analytical cognates ──")
    O_C, cog2_ana, cog3_ana = compute_analytical_cognates(ref)
    print(f"  Third pivot  O_C = ({O_C[0]:.4f}, {O_C[1]:.4f})")
    print(f"\n  Cognate 2 (B₀, C₀):\n    {cog2_ana}")
    print(f"  Cognate 3 (A₀, C₀):\n    {cog3_ana}")

    # ------------------------------------------------------------------
    #  Reference coupler curve (high resolution)
    # ------------------------------------------------------------------
    ref_curve = ref.generate_coupler_curve(720, mode=1)
    print(f"\n  Reference curve: {len(ref_curve)} points")

    # ------------------------------------------------------------------
    #  Numerical optimisation — Cognate 1  (ground: A₀, B₀)  [VERIFICATION]
    # ------------------------------------------------------------------
    print("\n── Optimising Cognate 1 (A₀, B₀) — self-recovery verification ──")
    cog1_opt, err1, sweep1, hist1 = optimize_cognate(ref_curve, O_A, O_B, verbose=True)
    print(f"  Result:\n    {cog1_opt}")
    print(f"  Best sweep: {sweep1}  |  Fine error: {err1:.3e}")

    # ------------------------------------------------------------------
    #  Numerical optimisation — Cognate 2  (ground: B₀, C₀)
    # ------------------------------------------------------------------
    print("\n── Optimising Cognate 2 (B₀, C₀) ──")
    cog2_opt, err2, sweep2, hist2 = optimize_cognate(ref_curve, O_B, O_C, verbose=True)
    print(f"  Result:\n    {cog2_opt}")
    print(f"  Best sweep: {sweep2}  |  Fine error: {err2:.3e}")

    # ------------------------------------------------------------------
    #  Numerical optimisation — Cognate 3  (ground: A₀, C₀)
    # ------------------------------------------------------------------
    print("\n── Optimising Cognate 3 (A₀, C₀) ──")
    opt_cog3, err3, sweep3, hist3 = optimize_cognate(
        ref_curve, O_A, O_C,
        n_restarts=10, n_points_coarse=180, n_points_fine=1440,
        grashof_target='double-rocker'
    )
    cog3_opt = opt_cog3
    print(f"  Result:\n    {cog3_opt}")
    print(f"  Best sweep: {sweep3}  |  Fine error: {err3:.3e}")

    # ------------------------------------------------------------------
    #  Comparison table (all 3 cognates)
    # ------------------------------------------------------------------
    cognates_ana = [ref,      cog2_ana, cog3_ana]
    cognates_opt = [cog1_opt, cog2_opt, cog3_opt]
    errors_opt   = [err1,     err2,     err3]
    print_comparison_table(ref, cognates_ana, cognates_opt, errors_opt)

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    base_dir = "/home/victor-alulema/PhD Program/Paper Iso-performance"

    fig1 = plot_multi_snapshot(
        ref, O_C, n_snapshots=8,
        save_path=f"{base_dir}/cognate_snapshots.png",
    )

    fig2 = plot_coupler_curves(
        ref, cog2_ana, cog3_ana,
        cog2_opt, cog3_opt,
        save_path=f"{base_dir}/cognate_curves.png",
    )

    fig3 = plot_parameter_comparison(
        cognates_ana, cognates_opt,
        save_path=f"{base_dir}/cognate_parameters.png",
    )

    fig4 = plot_optimization_history(
        [hist1, hist2, hist3],
        ['Cognate 1 (A₀, B₀)', 'Cognate 2 (B₀, C₀)', 'Cognate 3 (A₀, C₀)'],
        save_path=f"{base_dir}/cognate_optimization_history.png",
    )

    plt.show()
    print("Done.")


if __name__ == '__main__':
    main()

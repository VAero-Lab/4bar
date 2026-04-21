"""
End-to-end test of the BIOpt framework on Case 1 (structural sections).

Matches [A, Ixx, Iyy] (m=3, dim(M₀)=1); J is excluded because
open-section (I-beam) and closed-section (RHS) torsional constants
differ by orders of magnitude — a genuine physical result.

Includes visualization of:
  1. Cross-section comparison (I-beam vs. equivalent RHS)
  2. Section-property parity (bar chart)
  3. Manifold sampling (all equivalent RHS shapes overlaid)
  4. δ-sensitivity curve (ε* vs. δ)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import sys
from pathlib import Path

# Ensure project root is on sys.path (for standalone execution)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biopt.residual import IsoperformanceResidual
from biopt.pair import PairProblem
from biopt.sampling import ManifoldSampler
from biopt.optimizers import DEOptimizer
from analysis.sweeps import delta_sweep
from applications.structural.families import IBeamFamily, RHSFamily


# ── Global settings ──────────────────────────────────────────────────
X1_REF = np.array([0.10, 0.008, 0.184, 0.006])   # IPE 200-like
EVAL_PARAMS = [np.array([0.0])]                     # condition-independent
W_NO_J = np.array([1.0, 1.0, 1.0, 0.0])            # match A, Ixx, Iyy only

# Plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Color palette
C_IBEAM = "#2563EB"    # blue
C_RHS   = "#DC2626"    # red
C_FILL1 = "#DBEAFE"    # light blue fill
C_FILL2 = "#FEE2E2"    # light red fill
C_FEAS  = "#16A34A"    # green for feasible
C_INFEAS = "#DC2626"   # red for infeasible
C_ACCENT = "#7C3AED"   # purple accent


def make_residual(ibeam, rhs):
    return IsoperformanceResidual(ibeam, rhs, eval_params=EVAL_PARAMS, w=W_NO_J)


# ═══════════════════════════════════════════════════════════════════
#  Geometry helpers for drawing cross-sections
# ═══════════════════════════════════════════════════════════════════

def ibeam_polygon(x):
    """Return the closed I-beam polygon vertices (12 corners)."""
    b_f, t_f, h_w, t_w = x
    H = h_w + 2 * t_f
    hb, htw, hH, hhw = b_f / 2, t_w / 2, H / 2, h_w / 2
    verts = np.array([
        [-hb, -hH], [hb, -hH], [hb, -hH + t_f], [htw, -hH + t_f],
        [htw,  hH - t_f], [hb, hH - t_f], [hb, hH], [-hb, hH],
        [-hb, hH - t_f], [-htw, hH - t_f], [-htw, -hH + t_f], [-hb, -hH + t_f],
    ])
    return np.vstack([verts, verts[0:1]])  # close the polygon


def rhs_polygons(x):
    """Return outer and inner rectangles for the RHS cross-section."""
    B, H, t_h, t_v = x
    hB, hH = B / 2, H / 2
    outer = np.array([[-hB, -hH], [hB, -hH], [hB, hH], [-hB, hH], [-hB, -hH]])
    Bi, Hi = B - 2 * t_v, H - 2 * t_h
    hBi, hHi = Bi / 2, Hi / 2
    inner = np.array([[-hBi, -hHi], [hBi, -hHi], [hBi, hHi], [-hBi, hHi], [-hBi, -hHi]])
    return outer, inner


def draw_ibeam(ax, x, color=C_IBEAM, fill_color=C_FILL1, label="I-beam",
               lw=2.0, alpha_fill=0.4, scale=1e3):
    """Draw an I-beam cross-section on the given axes (dimensions in mm)."""
    verts = ibeam_polygon(x) * scale
    patch = mpatches.Polygon(verts[:-1], closed=True,
                             facecolor=fill_color, edgecolor=color,
                             linewidth=lw, alpha=alpha_fill, label=label)
    ax.add_patch(patch)
    # Draw the outline on top
    ax.plot(verts[:, 0], verts[:, 1], color=color, linewidth=lw, solid_capstyle="round")
    return patch


def draw_rhs(ax, x, color=C_RHS, fill_color=C_FILL2, label="RHS",
             lw=2.0, alpha_fill=0.4, scale=1e3):
    """Draw an RHS cross-section on the given axes (dimensions in mm)."""
    outer, inner = rhs_polygons(x)
    outer *= scale
    inner *= scale
    # Outer filled
    patch = mpatches.Polygon(outer[:-1], closed=True,
                             facecolor=fill_color, edgecolor=color,
                             linewidth=lw, alpha=alpha_fill, label=label)
    ax.add_patch(patch)
    # Inner hole (white fill)
    hole = mpatches.Polygon(inner[:-1], closed=True,
                            facecolor="white", edgecolor=color,
                            linewidth=lw * 0.7, linestyle="--", alpha=0.95)
    ax.add_patch(hole)
    ax.plot(outer[:, 0], outer[:, 1], color=color, linewidth=lw, solid_capstyle="round")
    return patch


# ═══════════════════════════════════════════════════════════════════
#  Test 1: Dimensional pre-analysis
# ═══════════════════════════════════════════════════════════════════

def test_dimensional_pre_analysis():
    ibeam, rhs = IBeamFamily(), RHSFamily()
    alpha = np.array([0.0])

    da1 = ibeam.dimensional_pre_analysis(m=4, x_test=X1_REF, alpha_test=alpha)
    da2 = rhs.dimensional_pre_analysis(m=4, x_test=rhs.bounds.mean(axis=1), alpha_test=alpha)

    print(f"I-beam: n={da1.n}, m={da1.m}, dim={da1.predicted_dim}, rank={da1.jacobian_rank}")
    print(f"RHS:    n={da2.n}, m={da2.m}, dim={da2.predicted_dim}, rank={da2.jacobian_rank}")
    assert da1.predicted_dim == 0 and da2.predicted_dim == 0
    assert da1.jacobian_rank == 4 and da2.jacobian_rank == 4
    print("✓ Dimensional pre-analysis passed.\n")


# ═══════════════════════════════════════════════════════════════════
#  Test 2: Pair problem + visualization
# ═══════════════════════════════════════════════════════════════════

def test_pair_problem():
    ibeam, rhs = IBeamFamily(), RHSFamily()
    residual = make_residual(ibeam, rhs)
    optimizer = DEOptimizer(maxiter=300, popsize=15, seed=42)

    pair = PairProblem(ibeam, rhs, residual, delta=0.005, epsilon_tol=1e-3, optimizer=optimizer)
    result = pair.solve(X1_REF)

    props_ref = ibeam.section_properties(X1_REF)
    props_eq  = rhs.section_properties(result.x2)

    print(f"Ref I-beam: A={props_ref[0]:.5f}, Ixx={props_ref[1]:.4e}, Iyy={props_ref[2]:.4e}")
    print(f"Equiv RHS:  A={props_eq[0]:.5f}, Ixx={props_eq[1]:.4e}, Iyy={props_eq[2]:.4e}")
    print(f"RHS: B={result.x2[0]:.4f}, H={result.x2[1]:.4f}, "
          f"t_h={result.x2[2]:.4f}, t_v={result.x2[3]:.4f}")
    print(f"ε* = {result.epsilon_star:.2e}, dS = {result.distance:.4f}, "
          f"feasible = {result.feasible}")

    assert result.feasible, f"Expected feasible, got ε*={result.epsilon_star:.2e}"
    rel = np.abs(props_eq[:3] - props_ref[:3]) / np.abs(props_ref[:3])
    assert np.all(rel < 1e-2), f"Relative errors too large: {rel}"
    print("✓ Pair problem passed.\n")

    # ── Visualization ──
    plot_pair_comparison(X1_REF, result.x2, ibeam, rhs, result)
    return result


def plot_pair_comparison(x1, x2, ibeam, rhs, pair_result):
    """Figure 1: Cross-section comparison + section-property bar chart."""
    props_ref = ibeam.section_properties(x1)
    props_eq  = rhs.section_properties(x2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})

    # ── Panel (a): I-beam cross-section ──
    ax = axes[0]
    draw_ibeam(ax, x1)
    b_f, t_f, h_w, t_w = x1
    H = h_w + 2 * t_f
    ax.set_xlim(-b_f * 1e3 * 0.7, b_f * 1e3 * 0.7)
    ax.set_ylim(-H * 1e3 * 0.65, H * 1e3 * 0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(a) Reference I-beam")
    ax.grid(True, alpha=0.3, linestyle="--")
    # Dimension annotations
    ax.annotate("", xy=(b_f * 500, -H * 550), xytext=(-b_f * 500, -H * 550),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(0, -H * 580, f"$b_f$ = {b_f*1e3:.0f} mm",
            ha="center", va="top", fontsize=9, color="gray")

    # ── Panel (b): RHS cross-section ──
    ax = axes[1]
    draw_rhs(ax, x2)
    B, H_rhs, t_h, t_v = x2
    ax.set_xlim(-B * 1e3 * 0.7, B * 1e3 * 0.7)
    ax.set_ylim(-H_rhs * 1e3 * 0.65, H_rhs * 1e3 * 0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(b) Equivalent RHS")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.annotate("", xy=(B * 500, -H_rhs * 550), xytext=(-B * 500, -H_rhs * 550),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text(0, -H_rhs * 580, f"$B$ = {B*1e3:.1f} mm",
            ha="center", va="top", fontsize=9, color="gray")

    # ── Panel (c): Section-property comparison ──
    ax = axes[2]
    labels = ["$A$\n[mm²]", "$I_{xx}$\n[mm⁴]", "$I_{yy}$\n[mm⁴]", "$J$\n[mm⁴]"]
    # Convert to mm units: A in mm², moments in mm⁴
    scale = np.array([1e6, 1e12, 1e12, 1e12])
    vals_ref = props_ref * scale
    vals_eq  = props_eq * scale

    x_pos = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x_pos - width / 2, vals_ref, width, label="I-beam (ref)",
                   color=C_IBEAM, alpha=0.75, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x_pos + width / 2, vals_eq, width, label="RHS (equiv)",
                   color=C_RHS, alpha=0.75, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("(c) Section Properties Comparison")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Add relative error annotations on matched properties
    for i in range(3):  # A, Ixx, Iyy
        rel_err = abs(vals_eq[i] - vals_ref[i]) / abs(vals_ref[i]) * 100
        y_pos = max(vals_ref[i], vals_eq[i]) * 1.15
        ax.text(x_pos[i], y_pos, f"Δ = {rel_err:.1e}%",
                ha="center", va="bottom", fontsize=8, color=C_FEAS, fontweight="bold")

    # Mark J as excluded
    ax.text(x_pos[3], max(vals_ref[3], vals_eq[3]) * 1.15, "excluded",
            ha="center", va="bottom", fontsize=8, color="gray", fontstyle="italic")

    # Super title
    fig.suptitle(
        f"Isoperformance Pair:  ε* = {pair_result.epsilon_star:.2e},  "
        f"$d_S$ = {pair_result.distance:.4f}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "tests" / "fig_pair_comparison.png")
    print("  → Saved fig_pair_comparison.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Test 3: Manifold sampling + visualization
# ═══════════════════════════════════════════════════════════════════

def test_manifold_sampling():
    ibeam, rhs = IBeamFamily(), RHSFamily()
    residual = make_residual(ibeam, rhs)
    optimizer = DEOptimizer(maxiter=300, popsize=15, seed=99)

    sampler = ManifoldSampler(ibeam, rhs, residual, delta=0.01,
                              epsilon_tol=1e-3, optimizer=optimizer)
    result = sampler.sample(X1_REF, R=5)

    print(f"Samples found: {result.n_accepted}, terminated early: {result.terminated_early}")
    for i, (x2, eps) in enumerate(zip(result.samples, result.residuals)):
        print(f"  #{i+1}: ε*={eps:.2e}, B={x2[0]:.4f}, H={x2[1]:.4f}")
    assert result.n_accepted >= 1, "Expected at least one equivalent"
    print("✓ Manifold sampling passed.\n")

    # ── Visualization ──
    plot_manifold_sampling(X1_REF, result, ibeam, rhs)
    return result


def plot_manifold_sampling(x1, sampling_result, ibeam, rhs):
    """Figure 2: Reference I-beam + all equivalent RHS shapes overlaid."""
    n_samples = sampling_result.n_accepted
    n_cols = min(n_samples + 1, 4)
    n_rows = 1 + (n_samples) // 4  # at least 1 row for ref + samples
    total_panels = 1 + n_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    # Flatten axes for easy indexing
    ax_list = axes.ravel()

    # Compute global axis limits across all shapes
    all_half_widths = [x1[0] / 2 * 1e3]  # I-beam b_f/2
    all_half_heights = [(x1[2] + 2 * x1[1]) / 2 * 1e3]  # I-beam H/2
    for x2 in sampling_result.samples:
        all_half_widths.append(x2[0] / 2 * 1e3)
        all_half_heights.append(x2[1] / 2 * 1e3)
    max_hw = max(all_half_widths) * 1.3
    max_hh = max(all_half_heights) * 1.3

    # ── Panel 0: Reference I-beam ──
    ax = ax_list[0]
    draw_ibeam(ax, x1)
    ax.set_xlim(-max_hw, max_hw)
    ax.set_ylim(-max_hh, max_hh)
    ax.set_aspect("equal")
    ax.set_title("Reference\nI-beam", fontsize=11)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── Panels 1..N: Equivalent RHS samples ──
    cmap = plt.cm.RdYlGn_r
    eps_vals = sampling_result.residuals
    eps_max = max(eps_vals) if eps_vals else 1.0
    eps_min = min(eps_vals) if eps_vals else 0.0

    for i, (x2, eps) in enumerate(zip(sampling_result.samples, eps_vals)):
        ax = ax_list[i + 1]
        # Color by residual
        if eps_max > eps_min:
            t = (eps - eps_min) / (eps_max - eps_min)
        else:
            t = 0.0
        color = cmap(t)
        fill_color = (*color[:3], 0.15)

        draw_rhs(ax, x2, color=(*color[:3], 1.0),
                 fill_color=fill_color, label=None)
        ax.set_xlim(-max_hw, max_hw)
        ax.set_ylim(-max_hh, max_hh)
        ax.set_aspect("equal")
        ax.set_title(f"RHS #{i+1}\nε* = {eps:.2e}", fontsize=10)
        ax.set_xlabel("x [mm]")
        ax.grid(True, alpha=0.2, linestyle="--")

        # Show dimensions
        B, H, t_h, t_v = x2
        ax.text(0.02, 0.98, f"B={B*1e3:.1f}\nH={H*1e3:.1f}\n"
                f"$t_h$={t_h*1e3:.2f}\n$t_v$={t_v*1e3:.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="gray"))

    # Hide unused axes
    for j in range(total_panels, len(ax_list)):
        ax_list[j].set_visible(False)

    fig.suptitle(f"Manifold Sampling:  {n_samples} Equivalent RHS Designs Found",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PROJECT_ROOT / "tests" / "fig_manifold_sampling.png")
    print("  → Saved fig_manifold_sampling.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Test 4: δ-sweep + visualization
# ═══════════════════════════════════════════════════════════════════

def test_delta_sweep():
    ibeam, rhs = IBeamFamily(), RHSFamily()
    residual = make_residual(ibeam, rhs)
    optimizer = DEOptimizer(maxiter=200, popsize=15, seed=42)

    delta_values = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

    result = delta_sweep(
        x1=X1_REF, family_ref=ibeam, family_target=rhs,
        residual=residual, delta_values=delta_values,
        epsilon_tol=1e-3, optimizer=optimizer,
    )

    print("δ-sweep:")
    for d, eps in zip(result.delta_values, result.epsilon_star):
        print(f"  δ={d:.3f} → ε*={eps:.2e} {'✓' if eps <= 1e-3 else '✗'}")
    print(f"  δ_max = {result.delta_max}")
    assert result.delta_max is not None, "Expected at least one feasible δ"
    print("✓ δ-sweep passed.\n")

    # ── Visualization ──
    plot_delta_sweep(result, ibeam, rhs)
    return result


def plot_delta_sweep(sweep_result, ibeam, rhs):
    """Figure 3: δ-sensitivity curve + equivalent shapes at selected δ."""
    deltas = np.array(sweep_result.delta_values)
    epsilons = np.array(sweep_result.epsilon_star)
    eps_tol = 1e-3

    feasible_mask = epsilons <= eps_tol

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # ── Panel (a): ε*(δ) curve ──
    ax = axes[0]

    # Fill feasible region
    if np.any(feasible_mask):
        d_max = deltas[feasible_mask][-1]
        ax.axvspan(0, d_max, alpha=0.08, color=C_FEAS, label="Feasible region")

    # Plot curve
    ax.semilogy(deltas[feasible_mask], epsilons[feasible_mask],
                "o-", color=C_FEAS, markersize=8, linewidth=2.2,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"Feasible (ε* ≤ {eps_tol:.0e})", zorder=5)
    ax.semilogy(deltas[~feasible_mask], epsilons[~feasible_mask],
                "s-", color=C_INFEAS, markersize=8, linewidth=2.2,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"Infeasible (ε* > {eps_tol:.0e})", zorder=5)

    # Connect the last feasible to first infeasible if adjacent
    feas_idx = np.where(feasible_mask)[0]
    infeas_idx = np.where(~feasible_mask)[0]
    if len(feas_idx) > 0 and len(infeas_idx) > 0:
        last_f = feas_idx[-1]
        first_inf = infeas_idx[infeas_idx > last_f]
        if len(first_inf) > 0:
            j = first_inf[0]
            ax.semilogy([deltas[last_f], deltas[j]], [epsilons[last_f], epsilons[j]],
                        "--", color="gray", linewidth=1.5, alpha=0.6, zorder=3)

    # Tolerance line
    ax.axhline(eps_tol, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(deltas[-1] * 0.98, eps_tol * 1.3, f"ε_tol = {eps_tol:.0e}",
            ha="right", va="bottom", fontsize=9, color="gray", fontstyle="italic")

    # δ_max annotation
    if sweep_result.delta_max is not None:
        ax.axvline(sweep_result.delta_max, color=C_ACCENT, linestyle="--",
                   linewidth=1.5, alpha=0.7)
        ax.text(sweep_result.delta_max, ax.get_ylim()[0] * 2,
                f"$δ_{{max}}$ = {sweep_result.delta_max:.3f}",
                ha="center", va="bottom", fontsize=10, color=C_ACCENT,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor=C_ACCENT))

    ax.set_xlabel("Shape-space separation δ")
    ax.set_ylabel("Optimal residual ε*")
    ax.set_title("(a) δ-Sensitivity Curve")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.set_xlim(0, deltas[-1] * 1.05)

    # ── Panel (b): RHS shapes at each δ, stacked ──
    ax = axes[1]

    # Draw reference I-beam centered at origin, faint
    draw_ibeam(ax, X1_REF, color=C_IBEAM, fill_color=C_FILL1,
               alpha_fill=0.15, lw=1.5, label="Reference I-beam")

    # Overlay equivalent RHS at each δ with varying opacity
    n = len(sweep_result.pair_results)
    for i, (d, pr) in enumerate(zip(deltas, sweep_result.pair_results)):
        if pr.epsilon_star > 1e20:
            continue  # skip infeasible (death penalty)
        alpha_val = 0.15 + 0.6 * (i / max(n - 1, 1))
        color = C_RHS if pr.feasible else C_INFEAS
        lw = 1.8 if pr.feasible else 1.0
        ls = "-" if pr.feasible else ":"

        outer, inner = rhs_polygons(pr.x2)
        outer *= 1e3
        inner *= 1e3
        ax.plot(outer[:, 0], outer[:, 1], color=color, linewidth=lw,
                alpha=alpha_val, linestyle=ls)
        # Label a few
        if i == 0 or i == n // 2 or i == n - 1:
            B = pr.x2[0]
            H = pr.x2[1]
            ax.text(B * 500 + 2, H * 500,
                    f"δ={d:.3f}", fontsize=7, color=color, alpha=0.9)

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(b) Equivalent Shapes at Varying δ")
    ax.grid(True, alpha=0.2, linestyle="--")

    # Set consistent limits
    b_f = X1_REF[0]
    H_ib = X1_REF[2] + 2 * X1_REF[1]
    lim = max(b_f, H_ib) * 1e3 * 0.85
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.suptitle("δ-Sensitivity Analysis:  I-beam → RHS Equivalence",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "tests" / "fig_delta_sweep.png")
    print("  → Saved fig_delta_sweep.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("BIOpt Framework — Structural Sections Test")
    print("=" * 60 + "\n")

    test_dimensional_pre_analysis()

    pair_res = test_pair_problem()
    sampling_res = test_manifold_sampling()
    sweep_res = test_delta_sweep()

    print("=" * 60)
    print("All tests passed.  Figures saved in tests/")
    print("=" * 60)

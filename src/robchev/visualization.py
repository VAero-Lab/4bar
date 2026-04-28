"""
Visualization module for the robchev package.

Contains:
    COLORS              – shared colour palette
    Plotter             – static figures (snapshots, curves, parameters, opt-history)
    CognateAnimator     – animated three-cognate linkage
    OptimizationAnimator – animated optimization convergence + linkage geometry
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree

from .kinematics import FourBarLinkage
from .optimization import _best_curve_for_linkage

# ─────────────────────────────────────────────────────────────────────────────
#  Shared colour palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'cog1': {'link': '#2563EB', 'curve': '#3B82F6', 'fill': '#93C5FD',
             'label': 'Cognate 1 (A₀-B₀)'},
    'cog2': {'link': '#DC2626', 'curve': '#EF4444', 'fill': '#FCA5A5',
             'label': 'Cognate 2 (B₀-C₀)'},
    'cog3': {'link': '#16A34A', 'curve': '#22C55E', 'fill': '#86EFAC',
             'label': 'Cognate 3 (A₀-C₀)'},
}

# ─────────────────────────────────────────────────────────────────────────────
#  Low-level drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_linkage(ax, joints, color_key, lw=2.2, zorder=3):
    """Draw a single four-bar linkage on *ax*."""
    c = COLORS[color_key]
    O1, A, B, O2, P = joints['O1'], joints['A'], joints['B'], joints['O2'], joints['P']
    for seg in [(O1, A), (A, B), (B, O2)]:
        ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                '-', color=c['link'], lw=lw, solid_capstyle='round', zorder=zorder)
    tri = plt.Polygon([A, B, P], alpha=0.4, color=c['fill'], zorder=zorder - 1)
    ax.add_patch(tri)
    for pt in [A, B, P]:
        ax.plot(*pt, 'o', color=c['link'], ms=4.5, mec='white', mew=0.6, zorder=zorder + 1)


def _draw_ground_pivots(ax, O_A, O_B, O_C, ms=9):
    for pt, label in [(O_A, 'A₀'), (O_B, 'B₀'), (O_C, 'C₀')]:
        ax.plot(*pt, '^', color='#1E293B', ms=ms, mec='white', mew=1.0, zorder=10)
        ax.annotate(label, pt, textcoords='offset points', xytext=(6, 6),
                    fontsize=8, fontweight='bold', color='#1E293B')


def _legend_handles():
    handles = [Line2D([0], [0], color=COLORS[f'cog{i}']['link'], lw=2.5,
                      label=COLORS[f'cog{i}']['label']) for i in [1, 2, 3]]
    handles.append(Line2D([0], [0], marker='o', color='black', lw=0, ms=7,
                          mec='white', mew=1.2, label='Coupler point P'))
    return handles

# ─────────────────────────────────────────────────────────────────────────────
#  Plotter – static figures
# ─────────────────────────────────────────────────────────────────────────────

class Plotter:
    """Static plot generation for cognate linkages."""

    @staticmethod
    def snapshot(analyzer, n_snapshots=8, save_path=None):
        """Multi-panel snapshot of all three cognates at evenly-spaced crank angles."""
        ref = analyzer.ref_linkage
        O_C = analyzer.O_C
        ref_curve = ref.generate_coupler_curve(720, mode=1)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=130)
        fig.suptitle('Roberts–Chebyshev Cognate Linkages — Instantaneous Configurations',
                     fontsize=15, fontweight='bold', y=0.97)

        thetas = np.linspace(0, 2 * np.pi, n_snapshots, endpoint=False)
        for ax, t2 in zip(axes.flat, thetas):
            all_j = analyzer.compute_all_joints(t2)
            if all_j is None:
                ax.set_visible(False)
                continue

            ax.plot(ref_curve[:, 0], ref_curve[:, 1], '-', color='#CBD5E1', lw=1.0, zorder=1)
            for key in ['cognate1', 'cognate2', 'cognate3']:
                _draw_linkage(ax, all_j[key], key.replace('nate', ''))
            _draw_ground_pivots(ax, ref.O1, ref.O2, O_C, ms=7)

            P = all_j['cognate1']['P']
            ax.plot(*P, 'o', color='black', ms=7, mec='white', mew=1.2, zorder=12)

            tri = plt.Polygon([ref.O1, ref.O2, O_C],
                              fill=False, edgecolor='#94A3B8', ls='--', lw=0.8, zorder=0)
            ax.add_patch(tri)

            ax.set_title(f'θ₂ = {np.degrees(t2):.0f}°', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)

        fig.legend(handles=_legend_handles(), loc='lower center', ncol=4, fontsize=10,
                   frameon=True, fancybox=True, shadow=True)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"  Saved → {save_path}")
        return fig

    @staticmethod
    def coupler_curves(analyzer, opt_cog2=None, opt_cog3=None, save_path=None):
        """Overlay of analytical vs optimised coupler curves for all cognates."""
        ref = analyzer.ref_linkage
        cog2_ana, cog3_ana = analyzer.cognate2, analyzer.cognate3
        ref_curve = ref.generate_coupler_curve(720, mode=1)
        ref_tree  = KDTree(ref_curve)

        fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=130)
        fig.suptitle('Coupler-Curve Comparison — Analytical vs. Optimised Cognates',
                     fontsize=14, fontweight='bold', y=1.00)

        titles       = ['Cognate 1 (reference)', 'Cognate 2 (B₀, C₀)', 'Cognate 3 (A₀, C₀)']
        ana_linkages = [ref, cog2_ana, cog3_ana]
        opt_linkages = [None, opt_cog2, opt_cog3]

        for i, ax in enumerate(axes):
            ax.plot(ref_curve[:, 0], ref_curve[:, 1], '-', color='#94A3B8',
                    lw=4.5, alpha=0.45, label='Reference curve', zorder=1)

            c_ana, _, _ = _best_curve_for_linkage(ana_linkages[i], 720, ref_curve, ref_tree)
            if len(c_ana) > 0:
                ax.plot(c_ana[:, 0], c_ana[:, 1], '--',
                        color=COLORS[f'cog{i+1}']['link'], lw=2.0, label='Analytical', zorder=3)

            if opt_linkages[i] is not None:
                c_opt, _, _ = _best_curve_for_linkage(opt_linkages[i], 720, ref_curve, ref_tree)
                if len(c_opt) > 0:
                    ax.plot(c_opt[:, 0], c_opt[:, 1], ':',
                            color=COLORS[f'cog{i+1}']['curve'], lw=2.5, label='Optimised', zorder=2)

            ax.set_title(titles[i], fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=8)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"  Saved → {save_path}")
        return fig

    @staticmethod
    def parameter_comparison(cognates_ana, cognates_opt, save_path=None):
        """Grouped bar chart: analytical vs optimised link parameters."""
        param_names  = ['L2', 'L3', 'L4', 'd_cp', 'α_cp']
        bar_ana = ['#2563EB', '#DC2626', '#16A34A']
        bar_opt = ['#93C5FD', '#FCA5A5', '#86EFAC']
        labels  = ['Cognate 1\n(A₀, B₀)', 'Cognate 2\n(B₀, C₀)', 'Cognate 3\n(A₀, C₀)']

        fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=130)
        fig.suptitle('Parameter Comparison — Analytical vs. Optimised',
                     fontsize=14, fontweight='bold', y=1.02)

        x, w = np.arange(3), 0.35
        for j, (ax, pname) in enumerate(zip(axes, param_names)):
            av_list, ov_list, pct = [], [], []
            for i in range(3):
                a, o = cognates_ana[i], cognates_opt[i]
                av = getattr(a, pname, None)
                ov = getattr(o, pname, None)
                if pname == 'α_cp':
                    av = np.degrees(a.alpha_cp)
                    ov = np.degrees(o.alpha_cp)
                else:
                    av = getattr(a, {'L2':'L2','L3':'L3','L4':'L4','d_cp':'d_cp'}[pname])
                    ov = getattr(o, {'L2':'L2','L3':'L3','L4':'L4','d_cp':'d_cp'}[pname])
                av_list.append(av); ov_list.append(ov)
                pct.append(abs(av - ov) / abs(av) * 100 if abs(av) > 1e-12 else 0.0)

            ax.bar(x - w/2, av_list, w, label='Analytical', color=bar_ana,
                   edgecolor='white', linewidth=0.8)
            bars2 = ax.bar(x + w/2, ov_list, w, label='Optimised', color=bar_opt,
                           edgecolor='white', linewidth=0.8)
            for b, p in zip(bars2, pct):
                ax.annotate(f'{p:.3f}%',
                            xy=(b.get_x() + b.get_width()/2, b.get_height()),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7,
                            color='#374151', fontweight='bold')
            unit = ' (deg)' if pname == 'α_cp' else ''
            ax.set_title(f'{pname}{unit}', fontsize=11, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
            ax.grid(axis='y', alpha=0.25, lw=0.5)
            if j == 0:
                ax.legend(fontsize=8, loc='upper left')

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"  Saved → {save_path}")
        return fig

    @staticmethod
    def optimization_history(optimizers, labels, save_path=None):
        """Convergence plot (error + parameters) for each optimizer."""
        n = len(optimizers)
        fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), dpi=130)
        if n == 1:
            axes = [axes]
        fig.suptitle('Optimization Convergence History', fontsize=15, fontweight='bold', y=0.97)

        param_labels = ['L2', 'L3', 'L4', 'd_cp', 'α_cp (rad)']
        colors_p = ['tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']

        for opt, ax1, lbl in zip(optimizers, axes, labels):
            if opt is None or opt.history is None:
                continue
            h = opt.history
            err1, err2 = h['phase1_err'], h['phase2_err']
            x1,   x2  = h['phase1_x'],   h['phase2_x']

            n1 = len(err1); n2 = len(err2)
            iters1 = np.arange(1, n1 + 1)
            iters2 = np.arange(n1 + 1, n1 + n2 + 1)

            ax1.set_title(lbl, fontsize=12, fontweight='bold')
            ax1.set_xlabel('Iteration (Gen for DE, Iter for L-BFGS-B)', fontsize=10)
            ax1.set_ylabel('Coupler Curve Error', color='tab:red', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.set_yscale('log')

            if n1 > 0:
                ax1.plot(iters1, err1, 'r-', marker='.', ms=4, alpha=0.7, label='Phase 1 (DE)')
            if n2 > 0:
                ax1.plot(iters2, err2, 'r--', marker='.', ms=4, alpha=0.7, label='Phase 2 (L-BFGS-B)')
            if n1 > 0 and n2 > 0:
                ax1.axvline(x=n1, color='gray', linestyle=':', alpha=0.5)
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.set_ylabel('Parameter Value', color='tab:blue', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            all_x = list(x1) + list(x2)
            if all_x:
                params = []
                for xk in all_x:
                    if h['is_dr']:
                        L3, d2, d4, dcp, acp = xk
                        params.append([L3+d2, L3, L3+d4, dcp, acp])
                    else:
                        params.append(list(xk))
                params = np.array(params)
                iters_all = np.arange(1, len(params) + 1)
                for j, pn in enumerate(param_labels):
                    ax2.plot(iters_all, params[:, j], color=colors_p[j], alpha=0.7, label=pn)

            l1, lb1 = ax1.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lb1 + lb2, loc='center left',
                       bbox_to_anchor=(1.05, 0.5), fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"  Saved → {save_path}")
        return fig

    @staticmethod
    def print_comparison_table(ref_linkage, cognates_ana, cognates_opt, errors_opt):
        labels = ['Cog 1 (A₀,B₀)', 'Cog 2 (B₀,C₀)', 'Cog 3 (A₀,C₀)']
        sep = '─' * 110
        print(f"\n{sep}\n  COGNATE PARAMETER COMPARISON — Analytical vs. Optimised\n{sep}")
        header = f"  {'Parameter':<14}"
        for lbl in labels:
            header += f"  {'Ana':>8}  {'Opt':>8}  {'Err%':>7}  │"
        print(header); print(sep)

        ref_curve = ref_linkage.generate_coupler_curve(720, mode=1)
        ref_tree  = KDTree(ref_curve)

        for pname in ['L2 (crank)', 'L3 (coupler)', 'L4 (rocker)', 'd_cp', 'α_cp (deg)', 'L1 (ground)']:
            row = f"  {pname:<14}"
            for i in range(3):
                a, o = cognates_ana[i], cognates_opt[i]
                if pname == 'L2 (crank)':    av, ov = a.L2, o.L2
                elif pname == 'L3 (coupler)': av, ov = a.L3, o.L3
                elif pname == 'L4 (rocker)':  av, ov = a.L4, o.L4
                elif pname == 'd_cp':          av, ov = a.d_cp, o.d_cp
                elif pname == 'α_cp (deg)':    av, ov = np.degrees(a.alpha_cp), np.degrees(o.alpha_cp)
                else:                          av, ov = a.L1, o.L1
                pct = abs(av - ov) / abs(av) * 100 if abs(av) > 1e-12 else 0.0
                row += f"  {av:>8.4f}  {ov:>8.4f}  {pct:>6.3f}%  │"
            print(row)
        print(sep)

        for i, lbl in enumerate(labels):
            for tag, lnk in [("analytical", cognates_ana[i]), ("optimised", cognates_opt[i])]:
                c, err, sweep = _best_curve_for_linkage(lnk, 1440, ref_curve, ref_tree)
                if len(c) < 5:
                    print(f"  {lbl} ({tag}): curve too short"); continue
                ct = KDTree(c)
                d1, _ = ct.query(ref_curve)
                rmse = np.sqrt(np.mean(d1**2))
                print(f"  {lbl} {tag:<12}  RMSE = {rmse:.2e}   [{sweep}]")

        print(f"\n  Objective errors (fine):")
        for i, lbl in enumerate(labels):
            print(f"    {lbl}:  {errors_opt[i]:.3e}")
        print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CognateAnimator – animated three-cognate linkage
# ─────────────────────────────────────────────────────────────────────────────

class CognateAnimator:
    """
    Animate all three Roberts-Chebyshev cognates simultaneously.

    The animation shows the three linkages sweeping through one full crank
    revolution on a single axes, with the shared coupler point tracing a
    growing trail.

    Parameters
    ----------
    analyzer : CognateAnalyzer
    n_frames : int
        Number of animation frames (crank-angle steps).
    interval : int
        Milliseconds between frames.
    figsize  : tuple

    Usage
    -----
    anim = CognateAnimator(analyzer)
    anim.build()
    anim.show()
    # or
    anim.save("output.gif")   # requires Pillow
    anim.save("output.mp4")   # requires ffmpeg
    """

    def __init__(self, analyzer, n_frames=120, interval=40, figsize=(10, 8)):
        self.analyzer  = analyzer
        self.n_frames  = n_frames
        self.interval  = interval
        self.figsize   = figsize
        self._anim     = None
        self._fig      = None

    def build(self):
        ref   = self.analyzer.ref_linkage
        O_C   = self.analyzer.O_C
        thetas = np.linspace(0, 2 * np.pi, self.n_frames, endpoint=False)

        # Pre-compute valid frames
        frames = []
        for t in thetas:
            j = self.analyzer.compute_all_joints(t)
            if j is not None:
                frames.append((t, j))

        if not frames:
            raise RuntimeError("No valid assembly positions found.")

        ref_curve = ref.generate_coupler_curve(720, mode=1)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=110)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')

        # Static: full coupler curve (faint background)
        ax.plot(ref_curve[:, 0], ref_curve[:, 1], '-', color='#475569', lw=1.0, zorder=1)

        # Static: focal triangle
        focal_tri = plt.Polygon([ref.O1, ref.O2, O_C],
                                 fill=False, edgecolor='#64748B', ls='--', lw=0.8, zorder=0)
        ax.add_patch(focal_tri)

        # Static: ground pivots
        for pt, lbl in [(ref.O1, 'A₀'), (ref.O2, 'B₀'), (O_C, 'C₀')]:
            ax.plot(*pt, '^', color='#E2E8F0', ms=10, mec='#0F172A', mew=1.0, zorder=11)
            ax.annotate(lbl, pt, textcoords='offset points', xytext=(7, 7),
                        fontsize=9, fontweight='bold', color='#E2E8F0')

        # Coupler-point trail (growing)
        trail_x, trail_y = [], []
        trail_line, = ax.plot([], [], '-', color='#F8FAFC', lw=1.2, alpha=0.6, zorder=5)

        # Dynamic link artists – 3 links each (O1-A, A-B, B-O2) + triangle + joints
        link_artists = {}
        for i, key in enumerate(['cog1', 'cog2', 'cog3']):
            c = COLORS[key]
            lines = [ax.plot([], [], '-', color=c['link'], lw=2.2,
                             solid_capstyle='round', zorder=3)[0] for _ in range(3)]
            tri_patch = plt.Polygon([[0,0],[1,0],[0,1]], alpha=0.35,
                                    color=c['fill'], zorder=2)
            ax.add_patch(tri_patch)
            pts_art = [ax.plot([], [], 'o', color=c['link'], ms=5,
                               mec='white', mew=0.7, zorder=4)[0] for _ in range(3)]
            link_artists[key] = dict(lines=lines, tri=tri_patch, pts=pts_art)

        coupler_dot, = ax.plot([], [], 'o', color='white', ms=9,
                               mec='#0F172A', mew=1.5, zorder=12)

        # Legend
        handles = [Line2D([0], [0], color=COLORS[f'cog{i}']['link'], lw=2.5,
                          label=COLORS[f'cog{i}']['label']) for i in [1, 2, 3]]
        handles.append(Line2D([0], [0], marker='o', color='white', lw=0,
                              ms=8, mec='#0F172A', mew=1.5, label='Coupler point P'))
        ax.legend(handles=handles, loc='upper right', fontsize=9,
                  framealpha=0.3, facecolor='#1E293B', edgecolor='#475569',
                  labelcolor='white')

        title = ax.set_title('', fontsize=11, color='#E2E8F0', pad=8)

        # Compute axis limits from all positions
        all_pts = np.vstack([ref_curve] +
                            [np.array(list(j.values())[:5]) for _, j_all in frames
                             for j in j_all.values()])
        xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
        ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
        pad = max(xmax - xmin, ymax - ymin) * 0.12
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, color='#475569', lw=0.5)
        ax.tick_params(colors='#94A3B8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

        fig.tight_layout()
        self._fig = fig

        def _update(frame_idx):
            _, all_j = frames[frame_idx]
            P = all_j['cognate1']['P']
            trail_x.append(P[0]); trail_y.append(P[1])
            trail_line.set_data(trail_x, trail_y)

            for key, cog_key in [('cog1','cognate1'),('cog2','cognate2'),('cog3','cognate3')]:
                j = all_j[cog_key]
                O1_, A_, B_, O2_, P_ = j['O1'], j['A'], j['B'], j['O2'], j['P']
                segs = [(O1_, A_), (A_, B_), (B_, O2_)]
                art  = link_artists[key]
                for ln, (p1, p2) in zip(art['lines'], segs):
                    ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                art['tri'].set_xy([A_, B_, P_])
                for pt_art, pt in zip(art['pts'], [A_, B_, P_]):
                    pt_art.set_data([pt[0]], [pt[1]])

            coupler_dot.set_data([P[0]], [P[1]])
            theta_deg = np.degrees(frames[frame_idx][0])
            title.set_text(f'Roberts–Chebyshev Cognates  |  θ₂ = {theta_deg:.1f}°')
            return [trail_line, coupler_dot, title] + \
                   [a for art in link_artists.values()
                    for a in art['lines'] + art['pts'] + [art['tri']]]

        self._anim = FuncAnimation(fig, _update, frames=len(frames),
                                   interval=self.interval, blit=True)
        return self

    def show(self):
        if self._anim is None:
            self.build()
        plt.show()
        return self

    def save(self, path, fps=None, dpi=120):
        """
        Save the animation.  Supports .gif (Pillow) and .mp4/.mov (ffmpeg).
        """
        if self._anim is None:
            self.build()
        fps = fps or max(1, int(1000 / self.interval))
        if path.endswith('.gif'):
            self._anim.save(path, writer='pillow', fps=fps, dpi=dpi)
        else:
            self._anim.save(path, writer='ffmpeg', fps=fps, dpi=dpi,
                            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"  Animation saved → {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
#  OptimizationAnimator – convergence history + evolving linkage geometry
# ─────────────────────────────────────────────────────────────────────────────

def _extract_params_history(history):
    """Return (iters, params_array, phase_split_idx) where params_array is (N,5): L2 L3 L4 d_cp alpha_cp."""
    x1, x2 = history['phase1_x'], history['phase2_x']
    is_dr   = history['is_dr']
    all_x   = list(x1) + list(x2)
    if not all_x:
        return np.array([]), np.empty((0, 5)), len(x1)
    params = []
    for xk in all_x:
        if is_dr:
            L3, d2, d4, dcp, acp = xk
            params.append([L3+d2, L3, L3+d4, dcp, acp])
        else:
            params.append(list(xk))
    return np.arange(1, len(params)+1), np.array(params), len(x1)


class OptimizationAnimator:
    """
    Animate the optimization process for a single cognate.

    Layout (3 panels):
      ┌────────────────┬────────────────┬──────────────────────┐
      │ Error (log)    │  Link params   │  Geometry            │
      │ convergence    │  evolution     │  Target vs Candidate │
      └────────────────┴────────────────┴──────────────────────┘

    Parameters
    ----------
    optimizer : CognateOptimizer  (already run)
    ref_curve : ndarray (N,2)     target coupler curve (optional, uses stored)
    title     : str
    interval  : int   ms between frames
    figsize   : tuple

    Usage
    -----
    oa = OptimizationAnimator(opt, title='Cognate 2')
    oa.build()
    oa.save("opt_cog2.gif")    # Pillow
    oa.save("opt_cog2.mp4")    # ffmpeg
    oa.save_final_png("opt_cog2_final.png")
    oa.show()
    """

    _PARAM_NAMES  = ['L2', 'L3', 'L4', 'd_cp', 'α_cp (rad)']
    _PARAM_COLORS = ['#38BDF8', '#34D399', '#A78BFA', '#FB923C', '#F472B6']

    def __init__(self, optimizer, ref_curve=None, title='', interval=60, figsize=(18, 6)):
        self.optimizer = optimizer
        self.ref_curve = (np.asarray(ref_curve) if ref_curve is not None
                          else optimizer.history['ref_curve'])
        self.title    = title
        self.interval = interval
        self.figsize  = figsize
        self._anim    = None
        self._fig     = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _ax_style(self, ax):
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#94A3B8', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334155')

    def _best_sweep_curve(self, lnk, n=360):
        """Return best candidate coupler curve for a linkage (tries all sweeps)."""
        thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
        ref_tree = KDTree(self.ref_curve)
        c, _, _ = _best_curve_for_linkage(lnk, n, self.ref_curve, ref_tree,
                                           sweep_types=self.optimizer.history['sweep_types'])
        return c

    def _linkage_snapshot(self, lnk):
        """Return joint dict at probe angle (tries both modes)."""
        for mode in [1, -1]:
            for theta in [np.pi/2, np.pi/4, np.pi/3, np.pi*3/4]:
                jnt = lnk.get_all_joints(theta, mode=mode)
                if jnt is not None:
                    return jnt
        return None

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self):
        opt        = self.optimizer
        hist       = opt.history
        all_errors = opt.all_errors
        n_total    = opt.n_iter_total
        n_phase1   = len(hist['phase1_err'])

        if n_total == 0:
            raise RuntimeError("Optimizer has no history. Call optimizer.run() first.")

        ref_curve             = self.ref_curve
        iters_p, params_h, _ = _extract_params_history(hist)

        fig, (ax_err, ax_par, ax_geo) = plt.subplots(1, 3, figsize=self.figsize, dpi=110)
        fig.patch.set_facecolor('#0F172A')
        for ax in (ax_err, ax_par, ax_geo):
            self._ax_style(ax)

        iters_all = np.arange(1, n_total + 1)

        # ── Panel 1: Error convergence ────────────────────────────────────────
        ax_err.set_yscale('log')
        ax_err.set_xlabel('Iteration', fontsize=9, color='#94A3B8')
        ax_err.set_ylabel('Curve Error (log scale)', fontsize=9, color='#EF4444')
        ax_err.tick_params(axis='y', labelcolor='#EF4444')
        ax_err.set_title('Convergence', fontsize=10, color='#E2E8F0', pad=6)
        ax_err.grid(True, alpha=0.2, color='#475569', lw=0.5)

        # Ghost of full error history (faint guide)
        ax_err.plot(iters_all, all_errors, '-', color='#334155', lw=0.8, alpha=0.5, zorder=1)
        if n_phase1 > 0:
            ax_err.axvline(x=n_phase1, color='#64748B', ls=':', lw=1.0)
            ax_err.text(n_phase1 + 0.3, all_errors[all_errors > 0].min() * 2,
                        ' P2', fontsize=7, color='#64748B', va='bottom')

        err_trail, = ax_err.plot([], [], '-', color='#EF4444', lw=1.8, alpha=0.85, zorder=3)
        err_dot,   = ax_err.plot([], [], 'o', color='#F59E0B', ms=7, zorder=5)
        phase_txt   = ax_err.text(0.03, 0.96, '', transform=ax_err.transAxes,
                                   fontsize=8, color='#94A3B8', va='top')

        # ── Panel 2: Parameter evolution ──────────────────────────────────────
        ax_par.set_xlabel('Iteration', fontsize=9, color='#94A3B8')
        ax_par.set_ylabel('Parameter Value', fontsize=9, color='#94A3B8')
        ax_par.set_title('Link Parameters', fontsize=10, color='#E2E8F0', pad=6)
        ax_par.grid(True, alpha=0.2, color='#475569', lw=0.5)

        # Ghost full history for all params
        par_lines_ghost = []
        par_lines_live  = []
        par_dots        = []
        if len(params_h) > 0:
            ax_par.set_xlim(1, n_total)
            ax_par.set_ylim(params_h.min() - 0.1, params_h.max() + 0.5)
            for j, (pn, pc) in enumerate(zip(self._PARAM_NAMES, self._PARAM_COLORS)):
                ax_par.plot(iters_p, params_h[:, j], '-', color=pc,
                            lw=0.7, alpha=0.25, zorder=1)
                ln, = ax_par.plot([], [], '-', color=pc, lw=1.8, alpha=0.9,
                                  label=pn, zorder=3)
                dot, = ax_par.plot([], [], 'o', color=pc, ms=5, zorder=5)
                par_lines_live.append(ln)
                par_dots.append(dot)

        ax_par.legend(fontsize=7, loc='upper right', framealpha=0.3,
                      facecolor='#1E293B', edgecolor='#475569', labelcolor='white',
                      ncol=2)

        # ── Panel 3: Geometry – target + candidate curve ──────────────────────
        # Static: target curve
        tgt_line, = ax_geo.plot(ref_curve[:, 0], ref_curve[:, 1], '-',
                                 color='#94A3B8', lw=3.0, alpha=0.55,
                                 label='Target curve', zorder=1)

        # Dynamic: candidate curve
        cand_line, = ax_geo.plot([], [], '--', color='#38BDF8', lw=2.0,
                                  alpha=0.9, label='Candidate curve', zorder=3)

        # Ground pivots (static)
        O1, O2 = opt.O1, opt.O2
        for pt, lbl in [(O1, 'O₁'), (O2, 'O₂')]:
            ax_geo.plot(*pt, '^', color='#E2E8F0', ms=10, mec='#0F172A', mew=1.0, zorder=11)
            ax_geo.annotate(lbl, pt, textcoords='offset points', xytext=(6, 6),
                            fontsize=9, fontweight='bold', color='#E2E8F0')

        # Dynamic: linkage snapshot
        snap_lines = [ax_geo.plot([], [], '-', color='#38BDF8', lw=2.5,
                                   solid_capstyle='round', zorder=4)[0] for _ in range(3)]
        snap_tri    = plt.Polygon([[0,0],[1,0],[0,1]], alpha=0.28, color='#7DD3FC', zorder=3)
        ax_geo.add_patch(snap_tri)
        snap_joints = [ax_geo.plot([], [], 'o', color='#38BDF8', ms=5.5,
                                    mec='white', mew=0.8, zorder=5)[0] for _ in range(3)]
        snap_cp,    = ax_geo.plot([], [], 'o', color='#F8FAFC', ms=9,
                                   mec='#0F172A', mew=1.5, zorder=6, label='Coupler pt')

        # Axis limits (generous padding around target + pivots)
        all_geo = np.vstack([ref_curve, O1.reshape(1,2), O2.reshape(1,2)])
        xmn, xmx = all_geo[:,0].min(), all_geo[:,0].max()
        ymn, ymx = all_geo[:,1].min(), all_geo[:,1].max()
        span = max(xmx - xmn, ymx - ymn, 1.0)
        pad  = span * 0.28
        ax_geo.set_xlim(xmn - pad, xmx + pad)
        ax_geo.set_ylim(ymn - pad, ymx + pad)
        ax_geo.set_aspect('equal')
        ax_geo.set_title('Target vs Candidate Curve', fontsize=10, color='#E2E8F0', pad=6)
        ax_geo.grid(True, alpha=0.15, color='#475569', lw=0.5)
        ax_geo.legend(fontsize=8, framealpha=0.3, facecolor='#1E293B',
                       edgecolor='#475569', labelcolor='white')

        suptitle = fig.suptitle(f'Optimization: {self.title}  |  Iter 0/{n_total}',
                                 fontsize=12, color='#E2E8F0', y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self._fig = fig

        # ── Update function ───────────────────────────────────────────────────
        def _update(frame):
            idx   = frame
            phase = 'Phase 1 – DE' if idx < n_phase1 else 'Phase 2 – L-BFGS-B'
            phase_txt.set_text(phase)

            # Panel 1 – error
            err_trail.set_data(iters_all[:idx+1], all_errors[:idx+1])
            err_dot.set_data([iters_all[idx]], [all_errors[idx]])

            # Panel 2 – parameters
            if len(params_h) > 0 and idx < len(params_h):
                for j, (ln, dot) in enumerate(zip(par_lines_live, par_dots)):
                    ln.set_data(iters_p[:idx+1], params_h[:idx+1, j])
                    dot.set_data([iters_p[idx]], [params_h[idx, j]])

            # Panel 3 – geometry
            lnk = opt.linkage_at_iter(idx)
            if lnk is not None:
                c_pts = self._best_sweep_curve(lnk)
                if len(c_pts) >= 5:
                    cand_line.set_data(c_pts[:, 0], c_pts[:, 1])
                else:
                    cand_line.set_data([], [])
                jnt = self._linkage_snapshot(lnk)
                if jnt is not None:
                    O1_, A_, B_, O2_, P_ = jnt['O1'], jnt['A'], jnt['B'], jnt['O2'], jnt['P']
                    for ln, (p1, p2) in zip(snap_lines, [(O1_, A_), (A_, B_), (B_, O2_)]):
                        ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    snap_tri.set_xy([A_, B_, P_])
                    for dot, pt in zip(snap_joints, [A_, B_, P_]):
                        dot.set_data([pt[0]], [pt[1]])
                    snap_cp.set_data([P_[0]], [P_[1]])

            suptitle.set_text(
                f'Optimization: {self.title}  |  Iter {idx+1}/{n_total}  '
                f'|  Error: {all_errors[idx]:.3e}')

            return ([err_trail, err_dot, phase_txt, cand_line, snap_cp, suptitle]
                    + par_lines_live + par_dots + snap_lines + snap_joints + [snap_tri])

        self._anim = FuncAnimation(fig, _update, frames=n_total,
                                   interval=self.interval, blit=False)
        return self

    # ── IO ────────────────────────────────────────────────────────────────────

    def show(self):
        if self._anim is None:
            self.build()
        plt.show()
        return self

    def save(self, path, fps=None, dpi=120):
        """Save as .gif (Pillow) or .mp4 / .mov (ffmpeg)."""
        if self._anim is None:
            self.build()
        fps = fps or max(1, int(1000 / self.interval))
        if path.endswith('.gif'):
            self._anim.save(path, writer='pillow', fps=fps, dpi=dpi)
        else:
            self._anim.save(path, writer='ffmpeg', fps=fps, dpi=dpi,
                            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"  Animation saved → {path}")
        return self

    def save_final_png(self, path, dpi=180):
        """
        3-panel static PNG: full convergence history | full parameter history
        | final linkage configuration vs target curve.
        """
        opt        = self.optimizer
        hist       = opt.history
        all_errors = opt.all_errors
        n_total    = opt.n_iter_total
        n_phase1   = len(hist['phase1_err'])
        ref_curve  = self.ref_curve
        iters_p, params_h, _ = _extract_params_history(hist)

        fig, (ax_err, ax_par, ax_geo) = plt.subplots(1, 3, figsize=self.figsize, dpi=dpi)
        fig.suptitle(f'Optimization Final State — {self.title}',
                     fontsize=13, fontweight='bold')

        iters = np.arange(1, n_total + 1)

        # Panel 1 – full error
        ax_err.set_yscale('log')
        ax_err.set_xlabel('Iteration')
        ax_err.set_ylabel('Curve Error (log)', color='tab:red')
        ax_err.tick_params(axis='y', labelcolor='tab:red')
        ax_err.set_title('Convergence History', fontweight='bold')
        if n_phase1 > 0:
            ax_err.plot(iters[:n_phase1], all_errors[:n_phase1],
                        'r-', marker='.', ms=3, alpha=0.8, label='Phase 1 (DE)')
        if n_total > n_phase1:
            ax_err.plot(iters[n_phase1:], all_errors[n_phase1:],
                        'r--', marker='.', ms=3, alpha=0.8, label='Phase 2 (L-BFGS-B)')
        if n_phase1 > 0 and n_total > n_phase1:
            ax_err.axvline(x=n_phase1, color='gray', ls=':', alpha=0.5)
        ax_err.legend(fontsize=8); ax_err.grid(True, alpha=0.3)

        # Panel 2 – full parameter history
        ax_par.set_xlabel('Iteration')
        ax_par.set_ylabel('Parameter Value')
        ax_par.set_title('Parameter History', fontweight='bold')
        if len(params_h) > 0:
            for j, (pn, pc) in enumerate(zip(self._PARAM_NAMES, self._PARAM_COLORS)):
                ax_par.plot(iters_p, params_h[:, j], '-', marker='.', ms=2,
                            color=pc, alpha=0.85, label=pn)
            if n_phase1 > 0 and n_total > n_phase1:
                ax_par.axvline(x=n_phase1, color='gray', ls=':', alpha=0.5)
        ax_par.legend(fontsize=8, ncol=2); ax_par.grid(True, alpha=0.3)

        # Panel 3 – final geometry
        ax_geo.plot(ref_curve[:,0], ref_curve[:,1], '-', color='#94A3B8',
                    lw=3, alpha=0.6, label='Target curve', zorder=1)
        ax_geo.set_title('Final Linkage Configuration', fontweight='bold')

        if opt.linkage is not None:
            ref_tree = KDTree(ref_curve)
            c, _, _ = _best_curve_for_linkage(
                opt.linkage, 720, ref_curve, ref_tree,
                sweep_types=hist['sweep_types'])
            if len(c) > 5:
                ax_geo.plot(c[:,0], c[:,1], '--', color='#2563EB',
                             lw=2, label='Final curve', zorder=3)
            jnt = self._linkage_snapshot(opt.linkage)
            if jnt is not None:
                O1_, A_, B_, O2_, P_ = jnt['O1'], jnt['A'], jnt['B'], jnt['O2'], jnt['P']
                for seg in [(O1_,A_),(A_,B_),(B_,O2_)]:
                    ax_geo.plot([seg[0][0],seg[1][0]],[seg[0][1],seg[1][1]],
                                '-', color='#2563EB', lw=2.5, solid_capstyle='round')
                tri = plt.Polygon([A_,B_,P_], alpha=0.3, color='#93C5FD', zorder=2)
                ax_geo.add_patch(tri)
                for pt in [A_,B_,P_]:
                    ax_geo.plot(*pt, 'o', color='#2563EB', ms=6, mec='white', mew=0.8)
                ax_geo.plot(*P_, 'o', color='black', ms=9, mec='white', mew=1.5,
                             zorder=6, label='Coupler point P')

        O1, O2 = opt.O1, opt.O2
        for pt, lbl in [(O1, 'O₁'), (O2, 'O₂')]:
            ax_geo.plot(*pt, '^', color='#1E293B', ms=10, mec='white', mew=1.0, zorder=10)
            ax_geo.annotate(lbl, pt, textcoords='offset points', xytext=(6, 6),
                            fontsize=9, fontweight='bold', color='#1E293B')
        ax_geo.set_aspect('equal')
        ax_geo.grid(True, alpha=0.25, lw=0.5)
        ax_geo.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"  Final PNG saved → {path}")
        plt.close(fig)
        return self

    _PARAM_NAMES  = ['L2', 'L3', 'L4', 'd_cp', 'α_cp (rad)']
    _PARAM_COLORS = ['#38BDF8', '#34D399', '#A78BFA', '#FB923C', '#F472B6']

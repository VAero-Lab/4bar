"""
Numerical cognate optimization module.

Provides ``CognateOptimizer``, which finds a four-bar linkage
whose coupler curve matches a reference curve using a two-phase
strategy:

  Phase 1 – Differential Evolution (global, coarse resolution)
  Phase 2 – L-BFGS-B local refinement (fine resolution)

The full iteration history (design variables + error at each step)
is stored for downstream animated visualization.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import KDTree

from .kinematics import FourBarLinkage


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_curve_for_linkage(linkage: FourBarLinkage, n_points: int,
                             ref_curve, ref_tree,
                             sweep_types=('crank', 'coupler', 'rocker')):
    """
    Try the requested sweep variants and return the curve with the
    lowest bidirectional nearest-neighbour error.

    Returns
    -------
    best_curve : ndarray (M, 2)
    best_error : float
    best_name  : str  (e.g. 'crank/mode=+1')
    """
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    best_error = 1e10
    best_curve = np.empty((0, 2))
    best_name  = ""

    for mode in [1, -1]:
        sweeps = []
        if 'crank'   in sweep_types:
            sweeps.append(("crank",   linkage._crank_sweep_vec(thetas, mode)))
        if 'coupler' in sweep_types:
            sweeps.append(("coupler", linkage._coupler_sweep_vec(thetas, mode)))
        if 'rocker'  in sweep_types:
            sweeps.append(("rocker",  linkage._rocker_sweep_vec(thetas, mode)))

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
                best_name  = f"{sweep_name}/mode={mode:+d}"

    return best_curve, best_error, best_name


def _coupler_curve_error(params, O1, O2, ref_curve, ref_tree,
                          n_points=360,
                          sweep_types=('crank', 'coupler', 'rocker'),
                          is_double_rocker=False):
    """
    Bidirectional nearest-neighbour curve-matching objective.
    """
    if is_double_rocker:
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
        sweep_types=sweep_types)
    return best_error


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class CognateOptimizer:
    """
    Find a four-bar linkage whose coupler curve matches a reference curve
    via a two-phase optimization (Differential Evolution → L-BFGS-B).

    Parameters
    ----------
    ref_curve : ndarray (N, 2)
        Target coupler-curve points.
    O1, O2 : array-like (2,)
        Fixed ground pivots for the new cognate.
    grashof_target : {None, 'double-rocker'}
        Pass ``'double-rocker'`` to constrain the search so that L3 is
        the shortest link (i.e. the coupler is the smallest link).
    n_restarts : int
        Number of independent Differential Evolution runs.
    n_points_coarse : int
        Curve resolution during DE (lower = faster).
    n_points_fine : int
        Curve resolution during L-BFGS-B refinement (higher = more accurate).
    verbose : bool
        Print convergence info.

    Attributes (after calling `run`)
    ----------------------------------
    linkage   : FourBarLinkage  – best linkage found.
    error     : float           – final objective value.
    best_sweep: str             – which sweep/mode combo was best.
    history   : dict            – iteration history for animation.
    """

    def __init__(self, ref_curve, O1, O2,
                 grashof_target=None,
                 n_restarts=10,
                 n_points_coarse=180,
                 n_points_fine=1440,
                 verbose=True):
        self.ref_curve       = np.asarray(ref_curve, dtype=float)
        self.O1              = np.asarray(O1, dtype=float)
        self.O2              = np.asarray(O2, dtype=float)
        self.grashof_target  = grashof_target
        self.n_restarts      = n_restarts
        self.n_points_coarse = n_points_coarse
        self.n_points_fine   = n_points_fine
        self.verbose         = verbose

        # Results – populated by run()
        self.linkage    = None
        self.error      = None
        self.best_sweep = None
        self.history    = None

        # Pre-compute reference tree and bounds
        self._ref_tree = KDTree(self.ref_curve)
        self._L1       = np.linalg.norm(self.O2 - self.O1)
        self._bounds, self._sweep_types, self._is_dr = self._make_bounds()

    # ------------------------------------------------------------------
    def _make_bounds(self):
        d_O1 = np.linalg.norm(self.ref_curve - self.O1, axis=1)
        d_O2 = np.linalg.norm(self.ref_curve - self.O2, axis=1)
        max_reach  = max(d_O1.max(), d_O2.max())
        curve_span = np.ptp(self.ref_curve, axis=0)
        curve_diag = np.linalg.norm(curve_span)

        link_ub = max(max_reach, curve_diag, self._L1) * 1.5
        link_lb = 0.05 * min(self._L1, 1.0)

        if self.grashof_target == 'double-rocker':
            bounds = [
                (link_lb, min(self._L1, link_ub)),   # L3 (shortest)
                (0.0, link_ub),                       # delta2
                (0.0, link_ub),                       # delta4
                (link_lb, link_ub),                   # d_cp
                (-np.pi, np.pi),                      # alpha_cp
            ]
            sweep_types = ('coupler',)
            is_dr = True
        else:
            bounds = [
                (link_lb, link_ub),   # L2
                (link_lb, link_ub),   # L3
                (link_lb, link_ub),   # L4
                (link_lb, link_ub),   # d_cp
                (-np.pi, np.pi),      # alpha_cp
            ]
            sweep_types = ('crank', 'coupler', 'rocker')
            is_dr = False

        return bounds, sweep_types, is_dr

    # ------------------------------------------------------------------
    def run(self):
        """Execute the two-phase optimization and populate result attributes."""
        if self.verbose:
            mode_str = " (Double-Rocker parameterisation)" if self._is_dr else ""
            link_lb, link_ub = self._bounds[0]
            print(f"  Bounds: links ∈ [{link_lb:.3f}, {link_ub:.3f}]  "
                  f"(L1={self._L1:.3f}){mode_str}")

        t0 = time.time()

        # ── Phase 1: Differential Evolution ──────────────────────────────
        best_de_result = None
        best_phase1_x   = []
        best_phase1_err = []

        for restart in range(self.n_restarts):
            seed = restart * 13 + 3
            if self.verbose:
                print(f"  Phase 1  [DE restart {restart+1}/{self.n_restarts}, seed={seed}]")

            tracker_x   = []
            tracker_err = []

            def _cb_de(xk, convergence=None):
                tracker_x.append(xk.copy())
                err = _coupler_curve_error(
                    xk, self.O1, self.O2,
                    self.ref_curve, self._ref_tree,
                    self.n_points_coarse,
                    self._sweep_types, self._is_dr)
                tracker_err.append(err)

            result = differential_evolution(
                _coupler_curve_error,
                bounds=self._bounds,
                args=(self.O1, self.O2,
                      self.ref_curve, self._ref_tree,
                      self.n_points_coarse,
                      self._sweep_types, self._is_dr),
                maxiter=300, popsize=30,
                init='sobol', tol=1e-10, atol=1e-10,
                seed=seed, polish=False, disp=False,
                callback=_cb_de,
            )

            if self.verbose:
                print(f"    → error = {result.fun:.3e}  ({result.nit} iters)")

            if best_de_result is None or result.fun < best_de_result.fun:
                best_de_result  = result
                best_phase1_x   = list(tracker_x)
                best_phase1_err = list(tracker_err)

            if best_de_result.fun < 1e-3:
                if self.verbose:
                    print("    ✓ Good basin found, proceeding to Phase 2.")
                break

        t_phase1 = time.time() - t0
        if self.verbose:
            print(f"  Phase 1 done in {t_phase1:.1f} s  |  "
                  f"coarse error = {best_de_result.fun:.3e}")

        # ── Phase 2: L-BFGS-B refinement ─────────────────────────────────
        if self.verbose:
            print(f"  Phase 2  (L-BFGS-B, {self.n_points_fine} pts per sweep)")

        phase2_x   = []
        phase2_err = []

        def _cb_min(xk):
            phase2_x.append(xk.copy())
            err = _coupler_curve_error(
                xk, self.O1, self.O2,
                self.ref_curve, self._ref_tree,
                self.n_points_fine,
                self._sweep_types, self._is_dr)
            phase2_err.append(err)

        fine_result = minimize(
            _coupler_curve_error,
            x0=best_de_result.x,
            args=(self.O1, self.O2,
                  self.ref_curve, self._ref_tree,
                  self.n_points_fine,
                  self._sweep_types, self._is_dr),
            method='L-BFGS-B',
            bounds=self._bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12},
            callback=_cb_min,
        )

        t_total = time.time() - t0
        if self.verbose:
            print(f"  Phase 2 done  |  fine error = {fine_result.fun:.3e}")
            print(f"  Total time: {t_total:.1f} s")

        # ── Reconstruct standard parameters ──────────────────────────────
        if self._is_dr:
            L3, delta2, delta4, d_cp, alpha_cp = fine_result.x
            L2, L4 = L3 + delta2, L3 + delta4
        else:
            L2, L3, L4, d_cp, alpha_cp = fine_result.x

        self.linkage = FourBarLinkage(
            self.O1, self.O2, L2, L3, L4, d_cp, alpha_cp)

        best_curve, final_error, best_sweep = _best_curve_for_linkage(
            self.linkage, n_points=self.n_points_fine,
            ref_curve=self.ref_curve, ref_tree=self._ref_tree,
            sweep_types=self._sweep_types)

        self.error      = final_error
        self.best_sweep = best_sweep

        # ── Store history ─────────────────────────────────────────────────
        self.history = {
            'phase1_x':   np.array(best_phase1_x)   if best_phase1_x   else np.empty((0, 5)),
            'phase1_err': np.array(best_phase1_err)  if best_phase1_err else np.array([]),
            'phase2_x':   np.array(phase2_x)         if phase2_x        else np.empty((0, 5)),
            'phase2_err': np.array(phase2_err)        if phase2_err      else np.array([]),
            'is_dr':      self._is_dr,
            'O1':         self.O1.copy(),
            'O2':         self.O2.copy(),
            'ref_curve':  self.ref_curve.copy(),
            'bounds':     self._bounds,
            'sweep_types': self._sweep_types,
        }

        return self

    # ------------------------------------------------------------------
    def params_at_iter(self, idx):
        """
        Return standard (L2, L3, L4, d_cp, alpha_cp) for a history index.
        Indexes phase1 first, then phase2.
        """
        p1 = self.history['phase1_x']
        p2 = self.history['phase2_x']
        total1 = len(p1)
        if idx < total1:
            x = p1[idx]
        else:
            x = p2[idx - total1]

        if self.history['is_dr']:
            L3, d2, d4, d_cp, alpha_cp = x
            return L3 + d2, L3, L3 + d4, d_cp, alpha_cp
        return tuple(x)

    def linkage_at_iter(self, idx):
        """Return a FourBarLinkage built from the parameters at iteration idx."""
        try:
            L2, L3, L4, d_cp, alpha_cp = self.params_at_iter(idx)
            if L2 <= 0 or L3 <= 0 or L4 <= 0 or d_cp <= 0:
                return None
            return FourBarLinkage(
                self.O1, self.O2, L2, L3, L4, d_cp, alpha_cp)
        except Exception:
            return None

    @property
    def n_iter_total(self):
        return len(self.history['phase1_x']) + len(self.history['phase2_x'])

    @property
    def all_errors(self):
        return np.concatenate([self.history['phase1_err'],
                               self.history['phase2_err']])

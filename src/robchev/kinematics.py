"""
Four-bar linkage kinematics module.

Provides a FourBarLinkage class with closed-form (vectorized) position analysis,
coupler-curve generation, and joint-position computation.

Conventions
-----------
    O1 ── L2 (crank) ── A ── L3 (coupler) ── B ── L4 (rocker) ── O2
    |                                                               |
    └──────────────── L1 (ground link) ────────────────────────────┘

    The coupler point P is located at distance  d_cp  from A at an angle
    alpha_cp  (measured CCW from the coupler direction A→B).

Author: Generated for Victor Alulema's PhD research
"""

import numpy as np


class FourBarLinkage:
    """Planar four-bar linkage mechanism with a coupler point."""

    def __init__(self, O1, O2, L2, L3, L4, d_cp, alpha_cp):
        """
        Parameters
        ----------
        O1 : array-like, shape (2,)
            First ground pivot (crank pivot).
        O2 : array-like, shape (2,)
            Second ground pivot (rocker pivot).
        L2 : float
            Crank length  |O1-A|.
        L3 : float
            Coupler length |A-B|.
        L4 : float
            Rocker length  |O2-B|.
        d_cp : float
            Distance from A to the coupler point P.
        alpha_cp : float
            Angle (rad) of P relative to the coupler direction A→B,
            measured counter-clockwise.
        """
        self.O1 = np.asarray(O1, dtype=float)
        self.O2 = np.asarray(O2, dtype=float)
        self.L2 = float(L2)
        self.L3 = float(L3)
        self.L4 = float(L4)
        self.d_cp = float(d_cp)
        self.alpha_cp = float(alpha_cp)

        g = self.O2 - self.O1
        self.L1 = np.linalg.norm(g)
        self.theta1 = np.arctan2(g[1], g[0])

    # ------------------------------------------------------------------
    #  Vectorised closed-form solvers
    # ------------------------------------------------------------------

    def _crank_sweep_vec(self, theta2, mode=1):
        """
        Vectorised position analysis over an array of crank angles.

        Returns
        -------
        points : ndarray, shape (M, 2)
            Valid coupler-point coordinates (M ≤ len(theta2)).
        theta2_valid : ndarray, shape (M,)
            Crank angles that led to a valid assembly.
        theta3_valid : ndarray, shape (M,)
            Corresponding coupler angles.
        theta4_valid : ndarray, shape (M,)
            Corresponding rocker angles.
        """
        t2 = np.asarray(theta2, dtype=float)
        Dx = self.O2[0] - self.O1[0] - self.L2 * np.cos(t2)
        Dy = self.O2[1] - self.O1[1] - self.L2 * np.sin(t2)
        R2 = Dx ** 2 + Dy ** 2
        denom = np.sqrt(R2)

        C = (self.L3 ** 2 - R2 - self.L4 ** 2) / (2.0 * self.L4)
        cos_val = np.where(denom > 1e-15, C / denom, 2.0)
        valid = np.abs(cos_val) <= 1.0

        cos_val_c = np.clip(cos_val, -1.0, 1.0)
        phi = np.arctan2(Dy, Dx)
        theta4 = phi + mode * np.arccos(cos_val_c)

        ex = Dx + self.L4 * np.cos(theta4)
        ey = Dy + self.L4 * np.sin(theta4)
        theta3 = np.arctan2(ey, ex)

        Px = self.O1[0] + self.L2 * np.cos(t2) + self.d_cp * np.cos(theta3 + self.alpha_cp)
        Py = self.O1[1] + self.L2 * np.sin(t2) + self.d_cp * np.sin(theta3 + self.alpha_cp)
        pts = np.column_stack([Px, Py])

        return pts[valid], t2[valid], theta3[valid], theta4[valid]

    def _coupler_sweep_vec(self, theta3, mode=1):
        """
        Vectorised solver parameterised by the coupler angle θ₃.

        Eliminates θ₂ from the loop closure.  Useful for double-rocker
        configurations where the crank cannot rotate fully.

        Returns
        -------
        points, theta2_valid, theta3_valid, theta4_valid
        """
        t3 = np.asarray(theta3, dtype=float)
        Kx = self.O1[0] - self.O2[0] + self.L3 * np.cos(t3)
        Ky = self.O1[1] - self.O2[1] + self.L3 * np.sin(t3)
        K2 = Kx ** 2 + Ky ** 2
        denom = np.sqrt(K2)

        C = (self.L4 ** 2 - self.L2 ** 2 - K2) / (2.0 * self.L2)
        cos_val = np.where(denom > 1e-15, C / denom, 2.0)
        valid = np.abs(cos_val) <= 1.0

        cos_val_c = np.clip(cos_val, -1.0, 1.0)
        phi = np.arctan2(Ky, Kx)
        theta2 = phi + mode * np.arccos(cos_val_c)

        # Recover θ₄ from B = O1 + L2 e^{iθ₂} + L3 e^{iθ₃}
        Bx = self.O1[0] + self.L2 * np.cos(theta2) + self.L3 * np.cos(t3)
        By = self.O1[1] + self.L2 * np.sin(theta2) + self.L3 * np.sin(t3)
        theta4 = np.arctan2(By - self.O2[1], Bx - self.O2[0])

        Px = self.O1[0] + self.L2 * np.cos(theta2) + self.d_cp * np.cos(t3 + self.alpha_cp)
        Py = self.O1[1] + self.L2 * np.sin(theta2) + self.d_cp * np.sin(t3 + self.alpha_cp)
        pts = np.column_stack([Px, Py])

        return pts[valid], theta2[valid], t3[valid], theta4[valid]

    def _rocker_sweep_vec(self, theta4, mode=1):
        """
        Vectorised solver parameterised by the rocker angle θ₄.

        Eliminates θ₃ from the loop closure to solve for θ₂.
        Essential for Grashof linkages where L4 is the shortest link
        (crank-at-O2 configurations) and can rotate fully while L2
        and L3 cannot.

        Returns
        -------
        points, theta2_valid, theta3_valid, theta4_valid
        """
        t4 = np.asarray(theta4, dtype=float)

        # B = O2 + L4·e^{iθ₄}  (known)
        Bx = self.O2[0] + self.L4 * np.cos(t4)
        By = self.O2[1] + self.L4 * np.sin(t4)

        # Vector from O1 to B
        D4x = Bx - self.O1[0]
        D4y = By - self.O1[1]
        D4_sq = D4x ** 2 + D4y ** 2
        denom = np.sqrt(D4_sq)

        # From |A - B| = L3  and  A = O1 + L2·e^{iθ₂}:
        #   D4x·cos θ₂ + D4y·sin θ₂ = (L2² + D4² - L3²) / (2·L2)
        C = (self.L2 ** 2 + D4_sq - self.L3 ** 2) / (2.0 * self.L2)
        cos_val = np.where(denom > 1e-15, C / denom, 2.0)
        valid = np.abs(cos_val) <= 1.0

        cos_val_c = np.clip(cos_val, -1.0, 1.0)
        phi = np.arctan2(D4y, D4x)
        theta2 = phi + mode * np.arccos(cos_val_c)

        # Recover θ₃  from  A→B direction
        Ax = self.O1[0] + self.L2 * np.cos(theta2)
        Ay = self.O1[1] + self.L2 * np.sin(theta2)
        theta3 = np.arctan2(By - Ay, Bx - Ax)

        # Coupler point
        Px = Ax + self.d_cp * np.cos(theta3 + self.alpha_cp)
        Py = Ay + self.d_cp * np.sin(theta3 + self.alpha_cp)
        pts = np.column_stack([Px, Py])

        return pts[valid], theta2[valid], theta3[valid], t4[valid]

    # ------------------------------------------------------------------
    #  Scalar solvers (single-angle queries)
    # ------------------------------------------------------------------

    def solve_position(self, theta2, mode=1):
        """
        Closed-form position analysis for a single crank angle.

        Returns
        -------
        (theta3, theta4) or None if not assemblable.
        """
        Dx = self.O2[0] - self.O1[0] - self.L2 * np.cos(theta2)
        Dy = self.O2[1] - self.O1[1] - self.L2 * np.sin(theta2)
        R2 = Dx * Dx + Dy * Dy
        denom = np.sqrt(R2)
        if denom < 1e-15:
            return None
        C = (self.L3 ** 2 - R2 - self.L4 ** 2) / (2.0 * self.L4)
        cos_val = C / denom
        if abs(cos_val) > 1.0 + 1e-10:
            return None
        cos_val = np.clip(cos_val, -1.0, 1.0)
        phi = np.arctan2(Dy, Dx)
        theta4 = phi + mode * np.arccos(cos_val)
        ex = Dx + self.L4 * np.cos(theta4)
        ey = Dy + self.L4 * np.sin(theta4)
        theta3 = np.arctan2(ey, ex)
        return theta3, theta4

    def coupler_point(self, theta2, mode=1):
        """Return the coupler-point position (Px, Py) or None."""
        result = self.solve_position(theta2, mode)
        if result is None:
            return None
        theta3, _ = result
        Px = self.O1[0] + self.L2 * np.cos(theta2) + self.d_cp * np.cos(theta3 + self.alpha_cp)
        Py = self.O1[1] + self.L2 * np.sin(theta2) + self.d_cp * np.sin(theta3 + self.alpha_cp)
        return np.array([Px, Py])

    def get_all_joints(self, theta2, mode=1):
        """
        Return all joint positions for a given crank angle.

        Returns
        -------
        dict with keys  O1, A, B, O2, P, theta2, theta3, theta4
        or None if the linkage cannot assemble.
        """
        result = self.solve_position(theta2, mode)
        if result is None:
            return None
        theta3, theta4 = result
        A = self.O1 + self.L2 * np.array([np.cos(theta2), np.sin(theta2)])
        B = self.O2 + self.L4 * np.array([np.cos(theta4), np.sin(theta4)])
        P = self.O1 + self.L2 * np.array([np.cos(theta2), np.sin(theta2)]) + \
            self.d_cp * np.array([np.cos(theta3 + self.alpha_cp),
                                  np.sin(theta3 + self.alpha_cp)])
        return dict(O1=self.O1.copy(), A=A, B=B, O2=self.O2.copy(), P=P,
                    theta2=theta2, theta3=theta3, theta4=theta4)

    # ------------------------------------------------------------------
    #  Coupler-curve generation
    # ------------------------------------------------------------------

    def generate_coupler_curve(self, n_points=360, mode=1):
        """Generate the coupler curve by sweeping the crank angle 0 → 2π."""
        thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        pts, _, _, _ = self._crank_sweep_vec(thetas, mode)
        return pts

    def generate_coupler_curve_robust(self, n_points=180, mode=1):
        """
        Generate a dense coupler curve using **all three** angular
        parameterisations — crank (θ₂), coupler (θ₃), and rocker (θ₄) —
        for the given assembly mode.

        This ensures full coverage for **every** Grashof type:
          • crank-rocker    → crank sweep covers full curve
          • double-rocker   → coupler sweep covers full curve
          • rocker-crank    → rocker sweep covers full curve
          • double-crank    → any sweep covers full curve
        """
        thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        pts_crank,   _, _, _ = self._crank_sweep_vec(thetas, mode)
        pts_coupler, _, _, _ = self._coupler_sweep_vec(thetas, mode)
        pts_rocker,  _, _, _ = self._rocker_sweep_vec(thetas, mode)

        parts = [p for p in [pts_crank, pts_coupler, pts_rocker] if len(p) > 0]
        if not parts:
            return np.empty((0, 2))
        combined = np.vstack(parts)
        # Remove near-duplicates
        _, idx = np.unique(np.round(combined, 6), axis=0, return_index=True)
        return combined[np.sort(idx)]

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def grashof_info(self):
        """Return a human-readable Grashof classification string."""
        links = sorted([self.L1, self.L2, self.L3, self.L4])
        s, p, q, l_ = links
        if s + l_ > p + q:
            return "non-Grashof (triple-rocker)"
        label = "change-point" if (s + l_ == p + q) else "Grashof"
        if min(self.L1, self.L2, self.L3, self.L4) == self.L2:
            detail = "crank-rocker (crank = input link L2)"
        elif min(self.L1, self.L2, self.L3, self.L4) == self.L1:
            detail = "double-crank (shortest = ground)"
        elif min(self.L1, self.L2, self.L3, self.L4) == self.L3:
            detail = "Grashof double-rocker (shortest = coupler)"
        else:
            detail = "rocker-crank (shortest = output link L4)"
        return f"{label}: {detail}"

    def __repr__(self):
        return (f"FourBarLinkage(\n"
                f"  O1={self.O1}, O2={self.O2},\n"
                f"  L2={self.L2:.4f}, L3={self.L3:.4f}, L4={self.L4:.4f},\n"
                f"  d_cp={self.d_cp:.4f}, α_cp={np.degrees(self.alpha_cp):.2f}°,\n"
                f"  L1(ground)={self.L1:.4f}, {self.grashof_info()}\n)")

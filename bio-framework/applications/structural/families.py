"""
Case 1: Structural cross-sections — I-beam vs. Rectangular Hollow Section.

Closed-form section properties: A, Ixx, Iyy, J.
All quantities admit analytic Jacobians.
Evaluation-parameter-independent (P = 1).
"""

from __future__ import annotations

import numpy as np

from biopt.families import ParameterizationFamily


class IBeamFamily(ParameterizationFamily):
    """Symmetric I-beam parameterized by [b_f, t_f, h_w, t_w].

    Parameters
    ----------
    bounds_dict : dict, optional
        Override default bounds.  Keys: ``b_f``, ``t_f``, ``h_w``, ``t_w``.
    n_boundary : int
        Number of perimeter points for the shape-space realization.
    """

    PROPERTY_NAMES = ["A", "Ixx", "Iyy", "J"]

    def __init__(
        self,
        bounds_dict: dict | None = None,
        n_boundary: int = 200,
        metrics: list[int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        metrics : list[int], optional
            Indices into [A, Ixx, Iyy, J] selecting which properties
            constitute the performance vector.  Default: all four.
            Example: ``[0, 1]`` matches only area and Ixx.
        """
        defaults = {
            "b_f": (0.05, 0.30),   # flange width [m]
            "t_f": (0.005, 0.03),  # flange thickness [m]
            "h_w": (0.05, 0.40),   # web height [m]
            "t_w": (0.003, 0.02),  # web thickness [m]
        }
        if bounds_dict:
            defaults.update(bounds_dict)
        self._bounds = np.array(list(defaults.values()))
        self._n_boundary = n_boundary
        self._metrics = metrics or [0, 1, 2, 3]

    @property
    def n_params(self) -> int:
        return 4

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def name(self) -> str:
        return "I-beam"

    @staticmethod
    def section_properties(x: np.ndarray) -> np.ndarray:
        """Compute [A, Ixx, Iyy, J] from parameters [b_f, t_f, h_w, t_w]."""
        b_f, t_f, h_w, t_w = x
        H = h_w + 2 * t_f

        A = 2 * b_f * t_f + h_w * t_w
        Ixx = (b_f * H**3 - (b_f - t_w) * h_w**3) / 12.0
        Iyy = (2 * t_f * b_f**3 + h_w * t_w**3) / 12.0
        J = (2 * b_f * t_f**3 + h_w * t_w**3) / 3.0

        return np.array([A, Ixx, Iyy, J])

    def performance(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Performance = selected section properties (condition-independent)."""
        return self.section_properties(x)[self._metrics]

    def realize(self, x: np.ndarray) -> np.ndarray:
        """Map I-beam parameters to boundary-point coordinates.

        Returns (x, y) pairs for N points along the perimeter,
        centroid-centered, parameterized by arclength fraction.
        """
        b_f, t_f, h_w, t_w = x
        H = h_w + 2 * t_f

        # I-beam boundary as a polygon (12 vertices, counterclockwise)
        half_b = b_f / 2
        half_tw = t_w / 2
        half_H = H / 2
        half_hw = h_w / 2

        vertices = np.array([
            [-half_b, -half_H],
            [half_b, -half_H],
            [half_b, -half_H + t_f],
            [half_tw, -half_H + t_f],
            [half_tw, half_H - t_f],
            [half_b, half_H - t_f],
            [half_b, half_H],
            [-half_b, half_H],
            [-half_b, half_H - t_f],
            [-half_tw, half_H - t_f],
            [-half_tw, -half_H + t_f],
            [-half_b, -half_H + t_f],
        ])
        return self._resample_boundary(vertices, self._n_boundary)

    def jacobian(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Analytic Jacobian ∂F/∂x, shape (4, 4)."""
        b_f, t_f, h_w, t_w = x
        H = h_w + 2 * t_f

        J = np.zeros((4, 4))

        # ∂A/∂[b_f, t_f, h_w, t_w]
        J[0] = [2 * t_f, 2 * b_f, t_w, h_w]

        # ∂Ixx/∂[b_f, t_f, h_w, t_w]
        J[1, 0] = (H**3 - h_w**3) / 12.0                        # ∂Ixx/∂b_f
        J[1, 1] = (b_f * 3 * H**2 * 2) / 12.0                   # ∂Ixx/∂t_f (via H)
        J[1, 2] = (b_f * 3 * H**2 - (b_f - t_w) * 3 * h_w**2) / 12.0  # ∂Ixx/∂h_w
        J[1, 3] = h_w**3 / 12.0                                  # ∂Ixx/∂t_w

        # ∂Iyy/∂[b_f, t_f, h_w, t_w]
        J[2, 0] = (2 * t_f * 3 * b_f**2) / 12.0
        J[2, 1] = (2 * b_f**3) / 12.0
        J[2, 2] = t_w**3 / 12.0
        J[2, 3] = (h_w * 3 * t_w**2) / 12.0

        # ∂J_torsion/∂[b_f, t_f, h_w, t_w]
        J[3, 0] = (2 * t_f**3) / 3.0
        J[3, 1] = (2 * b_f * 3 * t_f**2) / 3.0
        J[3, 2] = t_w**3 / 3.0
        J[3, 3] = (h_w * 3 * t_w**2) / 3.0

        return J

    @staticmethod
    def _resample_boundary(vertices: np.ndarray, n: int) -> np.ndarray:
        """Resample a closed polygon to n equally-spaced perimeter points."""
        verts = np.vstack([vertices, vertices[0:1]])
        diffs = np.diff(verts, axis=0)
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cum_len[-1]
        targets = np.linspace(0, total, n, endpoint=False)
        # Vectorized interpolation
        idx = np.searchsorted(cum_len, targets, side="right") - 1
        idx = np.clip(idx, 0, len(seg_lengths) - 1)
        safe_seg = np.where(seg_lengths[idx] > 0, seg_lengths[idx], 1.0)
        frac = (targets - cum_len[idx]) / safe_seg
        pts = verts[idx] + frac[:, None] * diffs[idx]
        return pts.ravel()


class RHSFamily(ParameterizationFamily):
    """Rectangular Hollow Section parameterized by [B, H, t_h, t_v].

    Parameters
    ----------
    bounds_dict : dict, optional
        Override default bounds.  Keys: ``B``, ``H``, ``t_h``, ``t_v``.
    n_boundary : int
        Number of perimeter points for the shape-space realization.
    """

    PROPERTY_NAMES = ["A", "Ixx", "Iyy", "J"]

    def __init__(
        self,
        bounds_dict: dict | None = None,
        n_boundary: int = 200,
        metrics: list[int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        metrics : list[int], optional
            Indices into [A, Ixx, Iyy, J] selecting which properties
            constitute the performance vector.  Default: all four.
        """
        defaults = {
            "B": (0.05, 0.30),     # outer width [m]
            "H": (0.05, 0.40),     # outer height [m]
            "t_h": (0.003, 0.03),  # horizontal wall thickness [m]
            "t_v": (0.003, 0.03),  # vertical wall thickness [m]
        }
        if bounds_dict:
            defaults.update(bounds_dict)
        self._bounds = np.array(list(defaults.values()))
        self._n_boundary = n_boundary
        self._metrics = metrics or [0, 1, 2, 3]

    @property
    def n_params(self) -> int:
        return 4

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def name(self) -> str:
        return "RHS"

    @staticmethod
    def section_properties(x: np.ndarray) -> np.ndarray:
        """Compute [A, Ixx, Iyy, J] from parameters [B, H, t_h, t_v]."""
        B, H, t_h, t_v = x
        Bi = B - 2 * t_v  # inner width
        Hi = H - 2 * t_h  # inner height

        A = B * H - Bi * Hi
        Ixx = (B * H**3 - Bi * Hi**3) / 12.0
        Iyy = (H * B**3 - Hi * Bi**3) / 12.0

        # Bredt-Batho torsional constant for thin-walled closed section
        Am = (B - t_v) * (H - t_h)  # enclosed area (midline)
        perimeter_over_t = (B - t_v) / t_h + (H - t_h) / t_v  # Σ(s/t)
        # Guard against degenerate geometry
        if perimeter_over_t > 0 and Am > 0:
            J = 4 * Am**2 / perimeter_over_t
        else:
            J = 0.0

        return np.array([A, Ixx, Iyy, J])

    def performance(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return self.section_properties(x)[self._metrics]

    def realize(self, x: np.ndarray) -> np.ndarray:
        """Map RHS parameters to boundary-point coordinates (outer boundary)."""
        B, H, t_h, t_v = x
        half_B = B / 2
        half_H = H / 2

        vertices = np.array([
            [-half_B, -half_H],
            [half_B, -half_H],
            [half_B, half_H],
            [-half_B, half_H],
        ])
        return IBeamFamily._resample_boundary(vertices, self._n_boundary)

    def jacobian(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Analytic Jacobian ∂F/∂x, shape (4, 4)."""
        B, H, t_h, t_v = x
        Bi = B - 2 * t_v
        Hi = H - 2 * t_h

        Jac = np.zeros((4, 4))

        # ∂A/∂[B, H, t_h, t_v]
        Jac[0] = [H - Hi, B - Bi, 2 * Bi, 2 * Hi]

        # ∂Ixx/∂[B, H, t_h, t_v]
        Jac[1, 0] = H**3 / 12.0                                          # ∂Ixx/∂B
        Jac[1, 1] = (B * 3 * H**2 - Bi * 3 * Hi**2) / 12.0              # ∂Ixx/∂H
        Jac[1, 2] = (Bi * 3 * Hi**2 * 2) / 12.0                         # ∂Ixx/∂t_h
        Jac[1, 3] = Hi**3 * 2 / 12.0                                    # ∂Ixx/∂t_v

        # ∂Iyy/∂[B, H, t_h, t_v]
        Jac[2, 0] = (H * 3 * B**2 - Hi * 3 * Bi**2) / 12.0
        Jac[2, 1] = B**3 / 12.0
        Jac[2, 2] = Bi**3 * 2 / 12.0
        Jac[2, 3] = (Hi * 3 * Bi**2 * 2) / 12.0

        # ∂J/∂[B, H, t_h, t_v] — numerical for Bredt-Batho
        eps = 1e-8
        J0 = self.section_properties(x)[3]
        for k in range(4):
            xp = x.copy()
            xp[k] += eps
            Jac[3, k] = (self.section_properties(xp)[3] - J0) / eps

        return Jac

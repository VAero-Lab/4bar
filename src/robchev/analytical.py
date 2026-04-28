import numpy as np
from .kinematics import FourBarLinkage

class CognateAnalyzer:
    """
    Class for analytically computing the two cognate linkages of a reference
    four-bar linkage using the Cayley-diagram / focal-triangle construction.
    """
    def __init__(self, ref_linkage: FourBarLinkage):
        self.ref_linkage = ref_linkage
        self.p, self.q, self.alpha, self.beta = self._compute_coupler_triangle()
        self.O_C = self._compute_third_pivot()
        self.cognate2, self.cognate3 = self._compute_analytical_cognates()

    def _compute_coupler_triangle(self):
        """
        Compute the coupler-triangle parameters from the reference FourBarLinkage.
        """
        p = self.ref_linkage.d_cp
        b = self.ref_linkage.L3
        alpha = self.ref_linkage.alpha_cp

        # |BP|² = p² + b² - 2pb cos α   (law of cosines in △ABP)
        q = np.sqrt(p ** 2 + b ** 2 - 2 * p * b * np.cos(alpha))

        # β via the complex-number relation  1/(1 − λe^{iα})
        # where λ = p/b.  The argument of that expression equals β.
        lam = p / b
        beta = np.arctan2(lam * np.sin(alpha), 1.0 - lam * np.cos(alpha))
        return p, q, alpha, beta

    def _compute_third_pivot(self):
        """
        Compute the third ground pivot O_C of the focal triangle.
        """
        O_A = self.ref_linkage.O1
        O_B = self.ref_linkage.O2
        p, b, alpha = self.p, self.ref_linkage.L3, self.alpha

        d_vec = O_B - O_A
        lam = p / b
        ca, sa = np.cos(alpha), np.sin(alpha)
        rotated = np.array([d_vec[0] * ca - d_vec[1] * sa,
                            d_vec[0] * sa + d_vec[1] * ca])
        return O_A + lam * rotated

    def _compute_analytical_cognates(self):
        """
        Derive the two cognate linkages analytically.
        """
        O_A = self.ref_linkage.O1
        O_B = self.ref_linkage.O2
        a = self.ref_linkage.L2   # crank
        b = self.ref_linkage.L3   # coupler
        c = self.ref_linkage.L4   # rocker

        p, q, alpha, beta = self.p, self.q, self.alpha, self.beta
        O_C = self.O_C

        # --- Cognate 2  (O_B, O_C)  — from Cayley-diagram Cognate III ---
        cognate2 = FourBarLinkage(
            O1=O_B, O2=O_C,
            L2=q,
            L3=c * q / b,
            L4=a * q / b,
            d_cp=c,
            alpha_cp=beta,
        )

        # --- Cognate 3  (O_A, O_C)  — from Cayley-diagram Cognate II ---
        cognate3 = FourBarLinkage(
            O1=O_A, O2=O_C,
            L2=p,
            L3=a * p / b,
            L4=c * p / b,
            d_cp=a,
            alpha_cp=-alpha,
        )

        return cognate2, cognate3

    def get_cognates(self):
        """Returns the third pivot O_C, cognate2, and cognate3."""
        return self.O_C, self.cognate2, self.cognate3

    def compute_all_joints(self, theta2):
        """
        Given the reference linkage at crank angle θ₂, compute joint positions
        of **all three** cognates using the Cayley-diagram parallelogram /
        similar-triangle construction (no separate position solving needed).
        """
        joints = self.ref_linkage.get_all_joints(theta2, mode=1)
        if joints is None:
            return None

        O_A = joints['O1']
        O_B = joints['O2']
        A   = joints['A']
        B   = joints['B']
        P   = joints['P']

        b = self.ref_linkage.L3
        lam = self.p / b
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)

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
            'cognate2': dict(O1=O_B, A=B3, B=C3, O2=self.O_C, P=P),
            'cognate3': dict(O1=O_A, A=A2, B=C2, O2=self.O_C, P=P),
        }

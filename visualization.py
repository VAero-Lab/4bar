"""
Four-Bar Linkage Parameter Visualization
Generates an annotated plot of a 4-bar linkage using the exact
parameters specified in the user's script.

Dependencies:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. User Defined Parameters (from script)
# ==========================================
O_A = np.array([0.0, 0.0])
O_B = np.array([6.0, 0.0])
a = 2.0                 # Crank
b = 5.0                 # Coupler baseline
c = 4.0                 # Rocker
p = 3.0                 # |AP| distance
alpha = np.pi / 6       # 30 deg coupler angle

# Input angle for visualization purposes (arbitrary, chosen for clear geometry)
theta_2 = np.radians(0) 

# ==========================================
# 2. Kinematic Forward Solution (Position Analysis)
# ==========================================
# Calculate Pivot A
A = O_A + np.array([a * np.cos(theta_2), a * np.sin(theta_2)])

# Calculate Pivot B (Intersection of two circles)
# Circle 1: Center A, radius b
# Circle 2: Center O_B, radius c
d = np.linalg.norm(O_B - A)

# Check for assemblability at this angle
if d > b + c or d < abs(b - c):
    raise ValueError("Mechanism cannot be assembled at the chosen input angle.")

# Using analytical circle-circle intersection
a_dist = (b**2 - c**2 + d**2) / (2 * d)
h = np.sqrt(b**2 - a_dist**2)

P2 = A + a_dist * (O_B - A) / d

# Two valid kinematic branches (elbow up / elbow down). We select one.
B_x = P2[0] + h * (O_B[1] - A[1]) / d
B_y = P2[1] - h * (O_B[0] - A[0]) / d
B = np.array([B_x, B_y])

# Calculate Coupler Point P
# Find the angle of vector AB
theta_3 = np.arctan2(B[1] - A[1], B[0] - A[0])
# P is located at distance p, rotated by alpha relative to AB
P = A + np.array([p * np.cos(theta_3 + alpha), p * np.sin(theta_3 + alpha)])

# ==========================================
# 3. Plotting & Annotation
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)

# Plot links
ax.plot([O_A[0], A[0]], [O_A[1], A[1]], 'r-', lw=3, label=f'Crank (a={a})')
ax.plot([A[0], B[0]], [A[1], B[1]], 'b-', lw=3, label=f'Coupler base (b={b})')
ax.plot([B[0], O_B[0]], [B[1], O_B[1]], 'g-', lw=3, label=f'Rocker (c={c})')
ax.plot([O_A[0], O_B[0]], [O_A[1], O_B[1]], 'k-', lw=3, label='Ground')

# Plot Rigid Coupler Triangle (A - B - P)
coupler_triangle = plt.Polygon([A, B, P], color='blue', alpha=0.2)
ax.add_patch(coupler_triangle)

# Plot joints and points
pts_x = [O_A[0], O_B[0], A[0], B[0], P[0]]
pts_y = [O_A[1], O_B[1], A[1], B[1], P[1]]
ax.plot(pts_x, pts_y, 'ko', markersize=8)

# Annotate Points
ax.text(O_A[0], O_A[1]-0.5, '$O_A$', fontsize=14, ha='center')
ax.text(O_B[0], O_B[1]-0.5, '$O_B$', fontsize=14, ha='center')
ax.text(A[0]-0.2, A[1]+0.3, '$A$', fontsize=14, ha='right')
ax.text(B[0]+0.3, B[1]+0.2, '$B$', fontsize=14, ha='left')
ax.text(P[0]+0.2, P[1]+0.2, '$P$ (Coupler Pt)', fontsize=14, color='darkblue', fontweight='bold')

# Annotate Parameters
ax.text((O_A[0]+A[0])/2 - 0.5, (O_A[1]+A[1])/2, 'a', fontsize=14, color='red', fontweight='bold')
ax.text((A[0]+B[0])/2, (A[1]+B[1])/2 - 0.4, 'b', fontsize=14, color='blue', fontweight='bold')
ax.text((B[0]+O_B[0])/2 + 0.3, (B[1]+O_B[1])/2, 'c', fontsize=14, color='green', fontweight='bold')

# Draw vector AP (distance p)
ax.plot([A[0], P[0]], [A[1], P[1]], 'b--', lw=2)
ax.text((A[0]+P[0])/2 - 0.2, (A[1]+P[1])/2 + 0.3, 'p', fontsize=14, color='darkblue', fontweight='bold')

# Draw alpha angle arc
arc = patches.Arc((A[0], A[1]), 1.5, 1.5, angle=np.degrees(theta_3), theta1=0, theta2=np.degrees(alpha), color='k', lw=2)
ax.add_patch(arc)
ax.text(A[0] + 0.8 * np.cos(theta_3 + alpha/2), A[1] + 0.8 * np.sin(theta_3 + alpha/2), 
        r'$\alpha$', fontsize=14, color='black', fontweight='bold')

# Formatting
ax.set_title("Four-Bar Linkage Parameter Definitions\n(Rigid Coupler Triangle in Blue)", fontsize=16)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend(loc='lower left')



plt.tight_layout()
plt.show()
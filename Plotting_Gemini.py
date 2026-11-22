import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Load the saved arrays from integration.py
data = np.load("project1_graphing_data_Gemini.npz")
r = data["RADIUS"]

# ==============================================================================
# DELIVERABLE 3: TRAILING EDGE DISTRIBUTIONS (Cz, Cr, beta, rho)
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Project 8.1: Trailing Edge Radial Distributions", fontsize=14, fontweight="bold")

# 1. Axial Velocity CZ(r)
ax = axes[0, 0]
ax.plot(r, data["CZ_TE_P1"], "r-o", label="Part 1: Compressible + Losses")
ax.plot(r, data["CZ_TE_P2"], "b--s", label="Part 2: Incompressible Lossless")
ax.plot(r, data["CZ_TE_ANA"], "k:", linewidth=2, label="Analytical (Quasi-1D)")
ax.set_xlabel("Radius r [m]")
ax.set_ylabel("$C_z(r)$ [m/s]")
ax.set_title("Axial Velocity $C_z(r)$ at T.E.")
ax.grid(True)
ax.legend()

# 2. Radial Velocity CR(r)
ax = axes[0, 1]
ax.plot(r, data["CR_TE_P1"], "r-o", label="Part 1 (at T.E.)")
ax.plot(r, data["CR_MID_P1"], "m-.^", label="Part 1 (at Mid-Chord)")
ax.plot(r, data["CR_TE_P2"], "b--s", label="Part 2 (at T.E.)")
ax.plot(r, data["CR_TE_ANA"], "k:", linewidth=2, label="Analytical (Quasi-1D)")
ax.set_xlabel("Radius r [m]")
ax.set_ylabel("$C_r(r)$ [m/s]")
ax.set_title("Radial Velocity $C_r(r)$")
ax.grid(True)
ax.legend()

# 3. Relative Blade Angle BETA(r)
ax = axes[1, 0]
ax.plot(r, data["BETA_TE_P1"], "r-o", label="Part 1: Compressible + Losses")
ax.plot(r, data["BETA_TE_P2"], "b--s", label="Part 2: Incompressible Lossless")
ax.plot(r, data["BETA_TE_ANA"], "k:", linewidth=2, label="Analytical (Quasi-1D)")
ax.set_xlabel("Radius r [m]")
ax.set_ylabel(r"Relative Angle $\beta(r)$ [deg]")
ax.set_title(r"Relative Blade Angle $\beta(r)$ at T.E.")
ax.grid(True)
ax.legend()

# 4. Density RHO(r)
ax = axes[1, 1]
ax.plot(r, data["DENSITY_TE_P1"], "r-o", label="Part 1: Compressible + Losses")
ax.plot(r, data["DENSITY_TE_P2"], "b--s", label="Part 2: Incompressible Lossless")
ax.plot(r, data["DENSITY_TE_ANA"], "k:", linewidth=2, label="Analytical (Quasi-1D)")
ax.set_xlabel("Radius r [m]")
ax.set_ylabel(r"Density $\rho(r)$ [kg/m$^3$]")
ax.set_title(r"Static Density $\rho(r)$ at T.E.")
ax.grid(True)
ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("Deliverable3_TE_Distributions.png", dpi=300)
plt.show()

# ==============================================================================
# DELIVERABLE 4: 3D DESIGNED BLADE COMPARISON (ALL 3 CASES)
# ==============================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(data["Z1"], data["Y1"], data["X1"], color="crimson", alpha=0.65, edgecolor="darkred", linewidth=0.4)
ax.plot_surface(data["Z2"], data["Y2"], data["X2"], color="royalblue", alpha=0.45, edgecolor="navy", linewidth=0.4)
ax.plot_wireframe(data["Z3"], data["Y3"], data["X3"], color="limegreen", linewidth=1.6)

legend_handles = [
    Patch(facecolor="crimson", edgecolor="darkred", alpha=0.65, label="Part 1: Compressible + Losses"),
    Patch(facecolor="royalblue", edgecolor="navy", alpha=0.45, label="Part 2: Incompressible Lossless"),
    Line2D([0], [0], color="limegreen", lw=2, label="Analytical (Quasi-1D Dixon)"),
]
ax.legend(handles=legend_handles, loc="upper left")

ax.set_title("Project 8.1: 3D Designed Rotor Blade (All 3 Cases)", fontsize=13, fontweight="bold")
ax.set_xlabel("Axial Chord z [m]")
ax.set_ylabel("Tangential y = r sin(θ) [m]")
ax.set_zlabel("Radial x = r cos(θ) [m]")
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=25, azim=-50)

plt.tight_layout()
plt.savefig("Deliverable4_3D_Blades.png", dpi=300)
plt.show()
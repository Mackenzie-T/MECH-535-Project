"""
Trailing-edge comparison plots and a 3D view of the designed rotor blade for:
    Part 1 - compressible flow with losses (solver)
    Part 2 - incompressible, lossless flow (solver)
    Analytical - quasi-1D solution (uniform Cz, zero Cr, constant density)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from thermodynamics import run_ree_solver


def blade_xyz(r, z_blade, beta_deg):
    """
    Build the blade camber surface by integrating d(theta)/dz = tan(beta) / r
    along each radius (trapezoidal rule). Returns Cartesian X, Y, Z arrays.
    """
    tan_b = np.tan(np.radians(beta_deg))
    theta = np.zeros_like(beta_deg)
    dz = z_blade[1] - z_blade[0]

    for i in range(1, len(z_blade)):
        theta[i, :] = theta[i - 1, :] + 0.5 * (tan_b[i - 1, :] + tan_b[i, :]) * (dz / r)

    R, Z = np.meshgrid(r, z_blade - z_blade[0])
    return R * np.cos(theta), R * np.sin(theta), Z


def analytical_solution(params):
    """Quasi-1D analytical values: Cz, Cr, density, and beta through the blade row."""
    r = params["RADIUS"]
    i_LE, i_TE = params["i_LE"], params["i_TE"]

    cz_ana = np.full_like(r, params["Cx_inlet"])
    cr_ana = np.zeros_like(r)
    rho_ana = np.full_like(r, params["rho_inlet_ref"])
    u_speed = params["OMEGA"] * r

    # rCu varies linearly from RCU_1 at the LE to RCU_2 at the TE
    n_blade = i_TE - i_LE + 1
    beta_blade_ana = np.zeros((n_blade, len(r)))
    for i in range(n_blade):
        frac = i / (n_blade - 1)
        cu = (params["RCU_1"] + frac * (params["RCU_2"] - params["RCU_1"])) / r
        beta_blade_ana[i, :] = np.degrees(np.arctan2(u_speed - cu, cz_ana))

    return cz_ana, cr_ana, rho_ana, beta_blade_ana


def plot_te_distributions(params, p1, p2, ana):
    """2x2 grid of Cz, Cr, beta and density versus radius at the trailing edge."""
    r = params["RADIUS"]
    i_LE, i_TE = params["i_LE"], params["i_TE"]
    i_MID = (i_LE + i_TE) // 2
    cz_ana, cr_ana, rho_ana, beta_blade_ana = ana

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Trailing Edge Radial Distributions")

    axes[0, 0].plot(r, p1["CZ"][i_TE, :], "r-o", label="Part 1: Comp + Losses")
    axes[0, 0].plot(r, p2["CZ"][i_TE, :], "b--s", label="Part 2: Incomp Lossless")
    axes[0, 0].plot(r, cz_ana, "k:", linewidth=2, label="Analytical")
    axes[0, 0].set(xlabel="r [m]", ylabel="Cz [m/s]", title="Axial Velocity Cz")

    axes[0, 1].plot(r, p1["CR"][i_TE, :], "r-o", label="Part 1 (T.E.)")
    axes[0, 1].plot(r, p1["CR"][i_MID, :], "m-.^", label="Part 1 (Mid-Chord)")
    axes[0, 1].plot(r, p2["CR"][i_TE, :], "b--s", label="Part 2 (T.E.)")
    axes[0, 1].plot(r, cr_ana, "k:", linewidth=2, label="Analytical")
    axes[0, 1].set(xlabel="r [m]", ylabel="Cr [m/s]", title="Radial Velocity Cr")

    axes[1, 0].plot(r, p1["BETA"][i_TE, :], "r-o", label="Part 1: Comp + Losses")
    axes[1, 0].plot(r, p2["BETA"][i_TE, :], "b--s", label="Part 2: Incomp Lossless")
    axes[1, 0].plot(r, beta_blade_ana[-1, :], "k:", linewidth=2, label="Analytical")
    axes[1, 0].set(xlabel="r [m]", ylabel="beta [deg]", title="Relative Angle beta")

    axes[1, 1].plot(r, p1["DENSITY"][i_TE, :], "r-o", label="Part 1: Comp + Losses")
    axes[1, 1].plot(r, p2["DENSITY"][i_TE, :], "b--s", label="Part 2: Incomp Lossless")
    axes[1, 1].plot(r, rho_ana, "k:", linewidth=2, label="Analytical")
    axes[1, 1].set(xlabel="r [m]", ylabel="rho [kg/m^3]", title="Density rho")

    for ax in axes.flat:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_3d_blades(params, p1, p2, beta_blade_ana):
    """3D blade surfaces for Part 1, Part 2 and the analytical solution."""
    r = params["RADIUS"]
    i_LE, i_TE = params["i_LE"], params["i_TE"]
    z_blade = params["Z"][i_LE : i_TE + 1]

    X1, Y1, Z1 = blade_xyz(r, z_blade, p1["BETA"][i_LE : i_TE + 1, :])
    X2, Y2, Z2 = blade_xyz(r, z_blade, p2["BETA"][i_LE : i_TE + 1, :])
    X3, Y3, Z3 = blade_xyz(r, z_blade, beta_blade_ana)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Z1, Y1, X1, color="crimson", alpha=0.65, edgecolor="darkred", linewidth=0.4)
    ax.plot_surface(Z2, Y2, X2, color="royalblue", alpha=0.45, edgecolor="navy", linewidth=0.4)
    ax.plot_wireframe(Z3, Y3, X3, color="limegreen", linewidth=1.5)

    ax.legend(
        handles=[
            Patch(facecolor="crimson", edgecolor="darkred", alpha=0.65, label="Part 1: Comp + Losses"),
            Patch(facecolor="royalblue", edgecolor="navy", alpha=0.45, label="Part 2: Incomp Lossless"),
            Line2D([0], [0], color="limegreen", lw=2, label="Analytical"),
        ],
        loc="upper left",
    )

    ax.set(title="3D Designed Rotor Blade", xlabel="z [m]", ylabel="y [m]", zlabel="x [m]")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=-50)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    params, p1 = run_ree_solver(compressible=True, with_losses=True)
    _, p2 = run_ree_solver(compressible=False, with_losses=False)
    ana = analytical_solution(params)

    plot_te_distributions(params, p1, p2, ana)
    plot_3d_blades(params, p1, p2, ana[3])

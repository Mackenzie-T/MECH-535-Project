"""
Side-by-side heatmaps of selected flow fields over the whole (z, r) domain,
comparing Part 1 (compressible + losses) with Part 2 (incompressible, lossless).
One PNG per field is saved to the heatmaps/ folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# (field key, label, file name) for the fields worth showing in the report.
REPORT_FIELDS = [
    ("CR", r"Radial velocity $C_r$ [m/s]", "radial_velocity"),
    ("CZ", r"Axial velocity $C_z$ [m/s]", "axial_velocity"),
    ("CU", r"Whirl velocity $C_\theta$ [m/s]", "tangential_velocity"),
    ("DENSITY", r"Density $\rho$ [kg/m$^3$]", "density"),
    ("ENTROPY", r"Entropy $s$ [J/kg/K]", "entropy"),
]

CASE_TITLES = ("Part 1: Compressible + Losses", "Part 2: Incompressible Lossless")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmaps")


def _shared_color_scale(left_data, right_data):
    """
    Pick one colour scale for both plots, so the same colour means the same
    value on the left and right. Returns (vmin, vmax, colormap name).
    """
    # Smallest and largest value found in either plot
    vmin = min(left_data.min(), right_data.min())
    vmax = max(left_data.max(), right_data.max())

    # Values on both sides of zero (e.g. radial velocity): use red for positive,
    # blue for negative, and make the limits symmetric so zero sits at white.
    if vmin < 0 and vmax > 0:
        biggest = max(-vmin, vmax)
        return -biggest, biggest, "RdBu_r"

    # Otherwise: light blue for low values, dark blue for high values
    return vmin, vmax, "Blues"


def plot_field_comparison(
    params, fields_comp, fields_incomp, key, label, save_path=None, show=False
):
    """
    Plot one field over the whole (z, r) domain for the compressible case (left)
    and the incompressible case (right), sharing one colour scale.
    Dashed lines mark the rotor leading and trailing edges.
    """
    z = params["Z"]
    r = params["RADIUS"]
    z_le, z_te = z[params["i_LE"]], z[params["i_TE"]]

    # fields are (NSTATN, NSTRM) -> transpose so rows = r, cols = z
    data = [np.asarray(f[key]).T for f in (fields_comp, fields_incomp)]
    vmin, vmax, cmap = _shared_color_scale(data[0], data[1])

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 3.6), sharey=True, constrained_layout=True
    )
    for ax, d, case in zip(axes, data, CASE_TITLES):
        mesh = ax.pcolormesh(
            z, r, d, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
        )
        for z_line in (z_le, z_te):
            ax.axvline(z_line, color="0.25", linestyle="--", linewidth=0.8)
        ax.set_title(case, fontsize=10)
        ax.set_xlabel("z [m]")
    axes[0].set_ylabel("r [m]")

    cbar = fig.colorbar(mesh, ax=axes, pad=0.02)
    cbar.set_label(label)
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()

    fig.suptitle(f"{label}  (dashed: rotor LE / TE)")

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig


def plot_report_heatmaps(
    params, fields_comp, fields_incomp, out_dir=OUTPUT_DIR, show=False
):
    """Save one side-by-side PNG per report field into out_dir. Returns the file paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for key, label, name in REPORT_FIELDS:
        path = os.path.join(out_dir, f"{name}_heatmap.png")
        fig = plot_field_comparison(
            params, fields_comp, fields_incomp, key, label, save_path=path, show=show
        )
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    from thermodynamics import run_ree_solver

    params, fields_comp = run_ree_solver(compressible=True, with_losses=True)
    _, fields_incomp = run_ree_solver(compressible=False, with_losses=False)

    for p in plot_report_heatmaps(params, fields_comp, fields_incomp):
        print("Saved", p)

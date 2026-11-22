"""
================================================================================
MECH 535 - PROJECT 8.1: FREE-VORTEX DESIGN (INTEGRATION & OUTPUT FILE)
================================================================================
Runs:
  1. Part 1: Compressible Flow with Losses (OMEGLOS_TE = 0.03)
  2. Part 2: Incompressible Lossless Flow (DENSITY = rho_0, OMEGLOS = 0)
  3. Case 3: Analytical Quasi-1D Solution (Dixon pp. 157-160)

Outputs:
  - Section 8.7 formatted plane tables printed to the terminal
  - 'project1_graphing_data.npz' containing all Trailing Edge distributions
    (Deliverable 3) and 3D blade coordinates for all 3 cases (Deliverable 4)
================================================================================
"""

import numpy as np
from thermodynamics import run_ree_solver


# ------------------------------------------------------------------------------
# 1. ANALYTICAL QUASI-1D SOLUTION (DIXON pp. 157-160 / CH. 7 SLIDES 34-37)
# ------------------------------------------------------------------------------
def compute_analytical_solution(params):
    RADIUS = params["RADIUS"]
    OMEGA = params["OMEGA"]

    CZ_ANA = np.full_like(RADIUS, params["Cx_inlet"])
    CR_ANA = np.zeros_like(RADIUS)
    DENSITY_ANA = np.full_like(RADIUS, params["rho_inlet_ref"])

    USPEED = OMEGA * RADIUS
    CU_LE = params["RCU_1"] / RADIUS
    CU_TE = params["RCU_2"] / RADIUS

    BETA_LE = np.degrees(np.arctan2(USPEED - CU_LE, CZ_ANA))
    BETA_TE = np.degrees(np.arctan2(USPEED - CU_TE, CZ_ANA))

    # Linearly ramp RCU from L.E. to T.E. to construct the analytical 3D blade surface
    n_blade_nodes = params["i_TE"] - params["i_LE"] + 1
    BETA_BLADE = np.zeros((n_blade_nodes, len(RADIUS)))
    for idx in range(n_blade_nodes):
        frac = idx / float(n_blade_nodes - 1)
        cu_local = (params["RCU_1"] + frac * (params["RCU_2"] - params["RCU_1"])) / RADIUS
        BETA_BLADE[idx, :] = np.degrees(np.arctan2(USPEED - cu_local, CZ_ANA))

    return {
        "RADIUS": RADIUS,
        "CZ_TE": CZ_ANA,
        "CR_TE": CR_ANA,
        "BETA_LE": BETA_LE,
        "BETA_TE": BETA_TE,
        "DENSITY_TE": DENSITY_ANA,
        "BETA_BLADE": BETA_BLADE,
    }


# ------------------------------------------------------------------------------
# 2. 3D BLADE CAMBER SURFACE COORDINATES (DELIVERABLE 4)
# ------------------------------------------------------------------------------
def compute_blade_3d_xyz(RADIUS, Z_BLADE, BETA_BLADE_DEG):
    """
    Integrates r * d(theta)/dz = tan(BETA(z, r)) from L.E. to T.E. using the
    trapezoidal rule, stacked radially at the L.E. (theta = 0).
    """
    tan_beta = np.tan(np.radians(BETA_BLADE_DEG))
    n_z, n_r = BETA_BLADE_DEG.shape
    theta = np.zeros((n_z, n_r))
    dz = Z_BLADE[1] - Z_BLADE[0]

    for i in range(1, n_z):
        theta[i, :] = theta[i - 1, :] + 0.5 * (tan_beta[i - 1, :] + tan_beta[i, :]) * (dz / RADIUS)

    R_grid, Z_grid = np.meshgrid(RADIUS, Z_BLADE - Z_BLADE[0])
    X = R_grid * np.cos(theta)  # Radial [m]
    Y = R_grid * np.sin(theta)  # Tangential [m]
    Z = Z_grid                  # Axial [m] (0 at L.E., 0.05 at T.E.)
    return X, Y, Z


# ------------------------------------------------------------------------------
# 3. SECTION 8.7 TABULATED PLANE OUTPUT FORMATTER
# ------------------------------------------------------------------------------
def print_plane_table(params, fields, plane_idx, label=""):
    """
    Prints flow properties at a given axial plane in the 3-block format of
    Section 8.7 (Sample Output), ordered from shroud down to hub.
    """
    NSTRM = params["NSTRM"]
    RADIUS = params["RADIUS"]
    OMEGA = params["OMEGA"]
    CP = params["CP"]
    z_val = params["Z"][plane_idx]

    print("\n" + "=" * 92)
    print(f"RESULTS ARE AS FOLLOWS FOR PLANE NO. {plane_idx + 1}  (Z = {z_val:.4f} m)  {label}")
    print("=" * 92)

    # Block 1: Kinematics
    print(f"{'RADIUS':>8} {'PSI':>8} {'CZ':>9} {'CR':>9} {'CM':>9} {'CU':>9} {'VU':>10} {'C':>9} {'V':>9}")
    for j in range(NSTRM - 1, -1, -1):
        r_j = RADIUS[j]
        cz = fields["CZ"][plane_idx, j]
        cr = fields["CR"][plane_idx, j]
        cm = fields["CM"][plane_idx, j]
        cu = fields["CU"][plane_idx, j]
        vu = OMEGA * r_j - cu
        c_mag = np.hypot(cm, cu)
        v_mag = np.hypot(cm, vu)
        print(
            f"{r_j:8.4f} {fields['PSI'][plane_idx, j]:8.4f} "
            f"{cz:9.3f} {cr:9.3f} {cm:9.3f} {cu:9.3f} {vu:10.3f} {c_mag:9.3f} {v_mag:9.3f}"
        )

    # Block 2: Pressures (kPa), Temperatures (K), Flow Angles (deg)
    print("\n" + f"{'PSTATIC':>10} {'TSTATIC':>9} {'TTOTAL':>9} {'TOREL':>9} {'PTOTAL':>10} {'PREL':>10} {'BETA(Z)':>9} {'ALPHA(Z)':>9}")
    for j in range(NSTRM - 1, -1, -1):
        cm = fields["CM"][plane_idx, j]
        cu = fields["CU"][plane_idx, j]
        t_tot = fields["HTOTAL"][plane_idx, j] / CP
        t_orel = fields["HOREL"][plane_idx, j] / CP
        alpha_deg = np.degrees(np.arctan2(cu, cm))
        print(
            f"{fields['PSTATIC'][plane_idx, j] / 1e3:10.3f} {fields['TSTATIC'][plane_idx, j]:9.3f} "
            f"{t_tot:9.3f} {t_orel:9.3f} "
            f"{fields['PTOTAL'][plane_idx, j] / 1e3:10.3f} {fields['PREL'][plane_idx, j] / 1e3:10.3f} "
            f"{fields['BETA'][plane_idx, j]:9.3f} {alpha_deg:9.3f}"
        )

    # Block 3: Swirl, Wheel Speed, Density, Z, Entropy
    print("\n" + f"{'RCU':>9} {'U':>9} {'DENSITY':>10} {'Z':>8} {'ENTROPY':>10}")
    for j in range(NSTRM - 1, -1, -1):
        r_j = RADIUS[j]
        rcu = r_j * fields["CU"][plane_idx, j]
        u_spd = OMEGA * r_j
        print(
            f"{rcu:9.3f} {u_spd:9.3f} {fields['DENSITY'][plane_idx, j]:10.5f} "
            f"{z_val:8.4f} {fields['ENTROPY'][plane_idx, j]:10.5f}"
        )


# ------------------------------------------------------------------------------
# 4. MAIN RUNNER & GRAPHING DATA EXPORT
# ------------------------------------------------------------------------------
def main():
    print("=================================================================")
    print("RUNNING PART 1: COMPRESSIBLE FLOW WITH LOSSES (OMEGLOS_TE = 0.03)")
    print("=================================================================")
    params, fields_part1 = run_ree_solver(compressible=True, with_losses=True)

    print("\n=================================================================")
    print("RUNNING PART 2: INCOMPRESSIBLE LOSSLESS FLOW (RHO=RHO_0, LOSS=0) ")
    print("=================================================================")
    _, fields_part2 = run_ree_solver(compressible=False, with_losses=False)

    print("\n=================================================================")
    print("COMPUTING CASE 3: ANALYTICAL QUASI-1D SOLUTION (DIXON)           ")
    print("=================================================================")
    ana_sol = compute_analytical_solution(params)

    i_LE = params["i_LE"]
    i_TE = params["i_TE"]
    i_MID = (i_LE + i_TE) // 2
    NSTATN = params["NSTATN"]

    # Print key Section 8.7 plane tables
    print("\n>>> PART 1 (COMPRESSIBLE + LOSSES) KEY PLANES <<<")
    print_plane_table(params, fields_part1, 0, label="[INLET]")
    print_plane_table(params, fields_part1, i_LE, label="[ROTOR L.E.]")
    print_plane_table(params, fields_part1, i_MID, label="[ROTOR MID-CHORD]")
    print_plane_table(params, fields_part1, i_TE, label="[ROTOR T.E.]")
    print_plane_table(params, fields_part1, NSTATN - 1, label="[EXIT]")

    print("\n>>> PART 2 (INCOMPRESSIBLE LOSSLESS) TRAILING EDGE <<<")
    print_plane_table(params, fields_part2, i_TE, label="[ROTOR T.E. - INCOMPRESSIBLE]")

    # Quick sanity-check comparison at Hub & Shroud (T.E.)
    print("\n--- QUICK TRAILING EDGE SUMMARY (HUB -> SHROUD) ---")
    print(f"Analytical : CZ={ana_sol['CZ_TE'][0]:.2f} -> {ana_sol['CZ_TE'][-1]:.2f} m/s | "
          f"BETA={ana_sol['BETA_TE'][0]:.2f} -> {ana_sol['BETA_TE'][-1]:.2f} deg | "
          f"RHO={ana_sol['DENSITY_TE'][0]:.3f} -> {ana_sol['DENSITY_TE'][-1]:.3f} kg/m^3")
    print(f"Part 2 (In): CZ={fields_part2['CZ'][i_TE, 0]:.2f} -> {fields_part2['CZ'][i_TE, -1]:.2f} m/s | "
          f"BETA={fields_part2['BETA'][i_TE, 0]:.2f} -> {fields_part2['BETA'][i_TE, -1]:.2f} deg | "
          f"RHO={fields_part2['DENSITY'][i_TE, 0]:.3f} -> {fields_part2['DENSITY'][i_TE, -1]:.3f} kg/m^3")
    print(f"Part 1 (Cp): CZ={fields_part1['CZ'][i_TE, 0]:.2f} -> {fields_part1['CZ'][i_TE, -1]:.2f} m/s | "
          f"BETA={fields_part1['BETA'][i_TE, 0]:.2f} -> {fields_part1['BETA'][i_TE, -1]:.2f} deg | "
          f"RHO={fields_part1['DENSITY'][i_TE, 0]:.3f} -> {fields_part1['DENSITY'][i_TE, -1]:.3f} kg/m^3")

    # Compute 3D blade coordinates for Deliverable 4
    RADIUS = params["RADIUS"]
    Z_BLADE = params["Z"][i_LE : i_TE + 1]
    X1, Y1, Z1 = compute_blade_3d_xyz(RADIUS, Z_BLADE, fields_part1["BETA"][i_LE : i_TE + 1, :])
    X2, Y2, Z2 = compute_blade_3d_xyz(RADIUS, Z_BLADE, fields_part2["BETA"][i_LE : i_TE + 1, :])
    X3, Y3, Z3 = compute_blade_3d_xyz(RADIUS, Z_BLADE, ana_sol["BETA_BLADE"])

    # Export everything needed for graphing (Deliverables 3 & 4)
    np.savez(
        "project1_graphing_data_Gemini.npz",
        RADIUS=RADIUS,
        Z=params["Z"],
        Z_BLADE=Z_BLADE,
        i_LE=i_LE,
        i_TE=i_TE,
        # Deliverable 3: Trailing Edge radial distributions
        CZ_TE_P1=fields_part1["CZ"][i_TE, :],
        CR_TE_P1=fields_part1["CR"][i_TE, :],
        CR_MID_P1=fields_part1["CR"][i_MID, :],
        BETA_TE_P1=fields_part1["BETA"][i_TE, :],
        DENSITY_TE_P1=fields_part1["DENSITY"][i_TE, :],
        CZ_TE_P2=fields_part2["CZ"][i_TE, :],
        CR_TE_P2=fields_part2["CR"][i_TE, :],
        BETA_TE_P2=fields_part2["BETA"][i_TE, :],
        DENSITY_TE_P2=fields_part2["DENSITY"][i_TE, :],
        CZ_TE_ANA=ana_sol["CZ_TE"],
        CR_TE_ANA=ana_sol["CR_TE"],
        BETA_TE_ANA=ana_sol["BETA_TE"],
        DENSITY_TE_ANA=ana_sol["DENSITY_TE"],
        # Full 2D grids (for streamline contours if desired)
        PSI_P1=fields_part1["PSI"],
        PSI_P2=fields_part2["PSI"],
        # Deliverable 4: 3D blade coordinates
        X1=X1, Y1=Y1, Z1=Z1,
        X2=X2, Y2=Y2, Z2=Z2,
        X3=X3, Y3=Y3, Z3=Z3,
    )
    print("\nSaved all graphing & 3D blade arrays to 'project1_graphing_data_Gemini.npz'.")


if __name__ == "__main__":
    main()
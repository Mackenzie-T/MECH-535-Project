import copy
import numpy as np
from fluids import (
    initialize_project,
    update_vorticity,
    update_streamfunction_psi,
    update_velocities,
)


def locate_streamline_origin(psi_target, psi_upstream_col, j_curr):
    """
    Finds the upstream origin interval at station i-1 where PSI == psi_target
    and returns the lower index k and linear interpolation weights (a, b) such that:
    psi_target = a * Psi_{i-1, k+1} + b * Psi_{i-1, k}, with a + b = 1.
    """
    NSTRM = len(psi_upstream_col)
    if j_curr == 0:
        return 0, 0.0, 1.0
    if j_curr == NSTRM - 1:
        return NSTRM - 2, 1.0, 0.0

    # Search monotonic array psi_upstream_col for bracket [k, k+1]
    k = np.searchsorted(psi_upstream_col, psi_target) - 1
    k = int(np.clip(k, 0, NSTRM - 2))
    denom = psi_upstream_col[k + 1] - psi_upstream_col[k]
    if abs(denom) < 1e-14:
        a = 0.5
    else:
        a = (psi_target - psi_upstream_col[k]) / denom
    a = float(np.clip(a, 0.0, 1.0))
    b = 1.0 - a
    return k, a, b


def trace_thermodynamics(params, fields, compressible=True, with_losses=True):
    """
    Step 7
    Traces streamlines back from station i to i-1 and applies Rothalpy conservation
    and the Common Calculation Block.
    """
    NSTATN, NSTRM = params["NSTATN"], params["NSTRM"]
    i_LE, i_TE = params["i_LE"], params["i_TE"]
    R = params["RADIUS"]
    OMEGA = params["OMEGA"]
    CP, RGAS, GAMMA = params["CP"], params["RGAS"], params["GAMMA"]
    GAM_EXP = GAMMA / (GAMMA - 1.0)

    PSI = fields["PSI"]
    CM = fields["CM"]
    CU = fields["CU"]
    OMEGLOS = fields["OMEGLOS"]

    DENSITY_OLD = fields["DENSITY"].copy()

    # Reset entropy s_ij = 0 at start of sweep after calculating RHS (Ch. 8 Page 2)
    fields["ENTROPY"].fill(0.0)

    for i in range(1, NSTATN):
        in_rotor = (i > i_LE) and (i <= i_TE)

        for j in range(NSTRM):
            k, a, b = locate_streamline_origin(PSI[i, j], PSI[i - 1, :], j)
            R1 = a * R[k + 1] + b * R[k]
            R2 = R[j]

            CU2 = CU[i, j]
            VU_2 = OMEGA * R2 - CU2
            CM2 = CM[i, j]
            fields["BETA"][i, j] = np.degrees(np.arctan2(VU_2, CM2))

            if not compressible and not with_losses:
                # Task 2 (Project Slide 3): Comment out / bypass density & loss recalculation
                fields["DENSITY"][i, j] = params["rho_inlet_ref"]
                fields["ENTROPY"][i, j] = 0.0
                continue

            # Interpolate upstream state at origin (bullet point at station i-1)
            ENTROPY1 = a * fields["ENTROPY"][i - 1, k + 1] + b * fields["ENTROPY"][i - 1, k]
            PSTATIC1 = a * fields["PSTATIC"][i - 1, k + 1] + b * fields["PSTATIC"][i - 1, k]

            if in_rotor:
                # Both station 1 (i-1) and station 2 (i) are in the rotating frame (Ch. 8 Slide 23 & Page 3)
                # Compute upstream relative stagnation properties at station i-1
                USPEED_COL = OMEGA * R
                VU1_COL = USPEED_COL - CU[i - 1, :]
                V1_SQ_COL = CM[i - 1, :] ** 2 + VU1_COL**2
                h_01rel_col = fields["h"][i - 1, :] + 0.5 * V1_SQ_COL
                p_01rel_col = (
                    fields["PSTATIC"][i - 1, :]
                    * (h_01rel_col / fields["h"][i - 1, :]) ** GAM_EXP
                )

                HOREL1 = a * h_01rel_col[k + 1] + b * h_01rel_col[k]
                PREL1 = a * p_01rel_col[k + 1] + b * p_01rel_col[k]

                # Common Calculation Block (Ch. 8 Slide 27 & Page 3)
                U1 = OMEGA * R1
                U2 = OMEGA * R2
                HOREL2 = HOREL1 - 0.5 * U1**2 + 0.5 * U2**2
                V_2_sq = CM2**2 + VU_2**2
                HSTATIC2 = HOREL2 - 0.5 * V_2_sq
                HTOTAL2 = HSTATIC2 + 0.5 * (CM2**2 + CU2**2)

                PREL2_ISEN = PREL1 * (HOREL2 / HOREL1) ** GAM_EXP
                LOSS_STEP = OMEGLOS[i, j] if with_losses else 0.0
                PREL2 = PREL2_ISEN - LOSS_STEP * (PREL1 - PSTATIC1)

                PTOTAL2 = PREL2 * (HTOTAL2 / HOREL2) ** GAM_EXP
                PSTATIC2 = PTOTAL2 * (HSTATIC2 / HTOTAL2) ** GAM_EXP
                DENSITY2 = (
                    (CP * PSTATIC2) / (RGAS * HSTATIC2)
                    if compressible
                    else params["rho_inlet_ref"]
                )
                ENTROPY2 = (
                    ENTROPY1 + CP * np.log(HOREL2 / HOREL1) - RGAS * np.log(PREL2 / PREL1)
                )

                fields["HOREL"][i, j] = HOREL2
                fields["PREL"][i, j] = PREL2
            else:
                # Upstream or downstream stationary duct (OMEGA = 0, HTOTAL and PTOTAL conserved along streamline)
                HTOTAL1 = a * fields["HTOTAL"][i - 1, k + 1] + b * fields["HTOTAL"][i - 1, k]
                PTOTAL1 = a * fields["PTOTAL"][i - 1, k + 1] + b * fields["PTOTAL"][i - 1, k]

                HTOTAL2 = HTOTAL1
                PTOTAL2 = PTOTAL1
                C_2_sq = CM2**2 + CU2**2
                HSTATIC2 = HTOTAL2 - 0.5 * C_2_sq
                PSTATIC2 = PTOTAL2 * (HSTATIC2 / HTOTAL2) ** GAM_EXP
                DENSITY2 = (
                    (CP * PSTATIC2) / (RGAS * HSTATIC2)
                    if compressible
                    else params["rho_inlet_ref"]
                )
                ENTROPY2 = ENTROPY1

            fields["h"][i, j] = HSTATIC2
            fields["TSTATIC"][i, j] = HSTATIC2 / CP
            fields["HTOTAL"][i, j] = HTOTAL2
            fields["PTOTAL"][i, j] = PTOTAL2
            fields["PSTATIC"][i, j] = PSTATIC2
            fields["DENSITY"][i, j] = DENSITY2
            fields["ENTROPY"][i, j] = ENTROPY2

    EPS_DENSITY = np.max(np.abs(fields["DENSITY"] - DENSITY_OLD))
    return EPS_DENSITY


def run_ree_solver(compressible=True, with_losses=True, min_sweeps=50, max_sweeps=500):
    """
    Steps 8 & 9
    Main iterative field-sweep loop. Runs at least 50 sweeps and checks convergence:
    eps_1 (PSI) <= 1e-5, eps_2 (DENSITY) <= 1e-3, eps_3 (RHS) <= 1e-3.
    """
    params, fields = initialize_project(compressible=compressible)

    for sweep in range(1, max_sweeps + 1):
        omega_old = fields["RHS"].copy()

        # Step 4: Calculate vorticity omega_ij
        if compressible or with_losses:
            update_vorticity(params, fields)
        else:
            fields["RHS"].fill(0.0)

        EPS_RHS = np.max(np.abs(fields["RHS"] - omega_old))

        # Step 5: Update Psi_ij
        EPS_PSI = update_streamfunction_psi(params, fields)

        # Step 6: Calculate velocities CZ, CR, CM
        update_velocities(params, fields)

        # Step 7: Trace back thermodynamic variables along streamlines
        EPS_DENSITY = trace_thermodynamics(
            params, fields, compressible=compressible, with_losses=with_losses
        )

        # Step 8: Check convergence after at least 50 sweeps (Ch. 8 Page 3-4)
        if sweep >= min_sweeps:
            if (
                (EPS_PSI <= 1e-5)
                and (EPS_DENSITY <= 1e-3)
                and (EPS_RHS <= 1e-3)
                ):
                print(
                    f"[Converged in {sweep} sweeps] "
                    f"dPsi={EPS_PSI:.2e}, drho={EPS_DENSITY:.2e}, domega={EPS_RHS:.2e}"
                )
                break
    else:
        print(
        f"[Reached {max_sweeps} sweeps] "
        f"dPsi={EPS_PSI:.2e}, drho={EPS_DENSITY:.2e}, domega={EPS_RHS:.2e}"
        )

    return params, copy.deepcopy(fields)

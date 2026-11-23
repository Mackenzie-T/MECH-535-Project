"""
Streamline tracing of thermodynamic properties and the main iterative solver loop.
"""

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
    Find where the streamline with value psi_target crossed station i-1.

    Returns the lower node index k and linear interpolation weights (a, b) such that
        psi_target = a * PSI[i-1, k+1] + b * PSI[i-1, k],  with a + b = 1.
    The hub and shroud streamlines map directly onto the hub and shroud nodes.
    """
    NSTRM = len(psi_upstream_col)
    if j_curr == 0:
        return 0, 0.0, 1.0
    if j_curr == NSTRM - 1:
        return NSTRM - 2, 1.0, 0.0

    # PSI increases monotonically from hub to shroud, so bracket the target value
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
    Trace each streamline from station i-1 to station i and update the
    thermodynamic state.

    - Outside the rotor: stagnation enthalpy and pressure are conserved.
    - Inside the rotor: rothalpy is conserved and the loss coefficient OMEGLOS
      reduces the relative stagnation pressure.

    Returns the largest change in density during the sweep.
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

    # Entropy is rebuilt from zero on every sweep (the RHS has already been computed)
    fields["ENTROPY"].fill(0.0)

    for i in range(1, NSTATN):
        in_rotor = i_LE < i <= i_TE

        for j in range(NSTRM):
            k, a, b = locate_streamline_origin(PSI[i, j], PSI[i - 1, :], j)
            R1 = a * R[k + 1] + b * R[k]  # radius of the streamline at station i-1
            R2 = R[j]

            CU2 = CU[i, j]
            VU_2 = OMEGA * R2 - CU2  # relative tangential velocity
            CM2 = CM[i, j]
            fields["BETA"][i, j] = np.degrees(np.arctan2(VU_2, CM2))

            if not compressible and not with_losses:
                # Incompressible, lossless case: density and entropy stay fixed
                fields["DENSITY"][i, j] = params["rho_inlet_ref"]
                fields["ENTROPY"][i, j] = 0.0
                continue

            # Upstream state at the streamline origin (interpolated at station i-1)
            ENTROPY1 = a * fields["ENTROPY"][i - 1, k + 1] + b * fields["ENTROPY"][i - 1, k]
            PSTATIC1 = a * fields["PSTATIC"][i - 1, k + 1] + b * fields["PSTATIC"][i - 1, k]

            if in_rotor:
                # Relative stagnation properties at station i-1
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

                # Rothalpy conservation: h_0,rel - U^2/2 is constant along the streamline
                U1 = OMEGA * R1
                U2 = OMEGA * R2
                HOREL2 = HOREL1 - 0.5 * U1**2 + 0.5 * U2**2
                V_2_sq = CM2**2 + VU_2**2
                HSTATIC2 = HOREL2 - 0.5 * V_2_sq
                HTOTAL2 = HSTATIC2 + 0.5 * (CM2**2 + CU2**2)

                # Isentropic relative stagnation pressure, then subtract the loss
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
                # Stationary duct: stagnation enthalpy and pressure are conserved
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
    Run the iterative field sweeps until convergence and return (params, fields).

    Each sweep: vorticity -> stream function -> velocities -> thermodynamics.
    At least min_sweeps are run, then the loop stops once
        max change in PSI <= 1e-5, in density <= 1e-3, and in RHS <= 1e-3.
    """
    params, fields = initialize_project(compressible=compressible)

    for sweep in range(1, max_sweeps + 1):
        rhs_old = fields["RHS"].copy()

        # Vorticity source term (zero for the incompressible, lossless case)
        if compressible or with_losses:
            update_vorticity(params, fields)
        else:
            fields["RHS"].fill(0.0)

        EPS_RHS = np.max(np.abs(fields["RHS"] - rhs_old))

        EPS_PSI = update_streamfunction_psi(params, fields)
        update_velocities(params, fields)
        EPS_DENSITY = trace_thermodynamics(
            params, fields, compressible=compressible, with_losses=with_losses
        )

        if sweep >= min_sweeps:
            if EPS_PSI <= 1e-5 and EPS_DENSITY <= 1e-3 and EPS_RHS <= 1e-3:
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

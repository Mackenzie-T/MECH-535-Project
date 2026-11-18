import copy
import numpy as np
# TODO: import the other files


def locate_streamline_origin(psi_target, psi_upstream_col, j_curr):
    """
    Finds the upstream origin interval at station i-1 where Psi == psi_target
    and returns the lower index k and linear interpolation weights (a, b) such that:
    psi_target = a * Psi_{i-1, k+1} + b * Psi_{i-1, k}, with a + b = 1.
    """
    M = len(psi_upstream_col)
    if j_curr == 0:
        return 0, 0.0, 1.0
    if j_curr == M - 1:
        return M - 2, 1.0, 0.0

    # Search monotonic array psi_upstream_col for bracket [k, k+1]
    k = np.searchsorted(psi_upstream_col, psi_target) - 1
    k = int(np.clip(k, 0, M - 2))
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
    N, M = params["N"], params["M"]
    i_LE, i_TE = params["i_LE"], params["i_TE"]
    r = params["r"]
    Omega = params["Omega"]
    cp, R_gas, gamma = params["cp"], params["R"], params["gamma"]
    g_exp = gamma / (gamma - 1.0)

    Psi = fields["Psi"]
    Cm = fields["Cm"]
    C_theta = fields["C_theta"]
    varpi_step = fields["varpi_step"]

    rho_old = fields["rho"].copy()

    # Reset entropy s_ij = 0 at start of sweep after calculating omega (Ch. 8 Page 2)
    fields["s"].fill(0.0)

    for i in range(1, N):
        in_rotor = (i > i_LE) and (i <= i_TE)

        for j in range(M):
            k, a, b = locate_streamline_origin(Psi[i, j], Psi[i - 1, :], j)
            r_1 = a * r[k + 1] + b * r[k]
            r_2 = r[j]

            C_th_2 = C_theta[i, j]
            V_th_2 = Omega * r_2 - C_th_2
            Cm_2 = Cm[i, j]
            fields["beta"][i, j] = np.degrees(np.arctan2(V_th_2, Cm_2))

            if not compressible and not with_losses:
                # Task 2 (Project Slide 3): Comment out / bypass density & loss recalculation
                fields["rho"][i, j] = params["rho_inlet_ref"]
                fields["s"][i, j] = 0.0
                continue

            # Interpolate upstream state at origin (bullet point at station i-1)
            s_1 = a * fields["s"][i - 1, k + 1] + b * fields["s"][i - 1, k]
            p_1 = a * fields["p"][i - 1, k + 1] + b * fields["p"][i - 1, k]

            if in_rotor:
                # Both station 1 (i-1) and station 2 (i) are in the rotating frame (Ch. 8 Slide 23 & Page 3)
                # Compute upstream relative stagnation properties at station i-1
                U_vec = Omega * r
                V_th_up = U_vec - C_theta[i - 1, :]
                V_sq_up = Cm[i - 1, :] ** 2 + V_th_up**2
                h_01rel_col = fields["h"][i - 1, :] + 0.5 * V_sq_up
                p_01rel_col = (
                    fields["p"][i - 1, :]
                    * (h_01rel_col / fields["h"][i - 1, :]) ** g_exp
                )

                h_01rel = a * h_01rel_col[k + 1] + b * h_01rel_col[k]
                p_01rel = a * p_01rel_col[k + 1] + b * p_01rel_col[k]

                # Common Calculation Block (Ch. 8 Slide 27 & Page 3)
                U_1 = Omega * r_1
                U_2 = Omega * r_2
                h_02rel = h_01rel - 0.5 * U_1**2 + 0.5 * U_2**2
                V_2_sq = Cm_2**2 + V_th_2**2
                h_2 = h_02rel - 0.5 * V_2_sq
                h_02 = h_2 + 0.5 * (Cm_2**2 + C_th_2**2)

                p_02rel_i = p_01rel * (h_02rel / h_01rel) ** g_exp
                loss_coeff = varpi_step[i, j] if with_losses else 0.0
                p_02rel = p_02rel_i - loss_coeff * (p_01rel - p_1)

                p_02 = p_02rel * (h_02 / h_02rel) ** g_exp
                p_2 = p_02 * (h_2 / h_02) ** g_exp
                rho_2 = (
                    (cp * p_2) / (R_gas * h_2)
                    if compressible
                    else params["rho_inlet_ref"]
                )
                s_2 = (
                    s_1
                    + cp * np.log(h_02rel / h_01rel)
                    - R_gas * np.log(p_02rel / p_01rel)
                )

                fields["h_o_rel"][i, j] = h_02rel
                fields["p_o_rel"][i, j] = p_02rel
            else:
                # Upstream or downstream stationary duct (Omega = 0, h_0 and p_0 conserved along streamline)
                h_01 = a * fields["h_o"][i - 1, k + 1] + b * fields["h_o"][i - 1, k]
                p_01 = a * fields["p_o"][i - 1, k + 1] + b * fields["p_o"][i - 1, k]

                h_02 = h_01
                p_02 = p_01
                C_2_sq = Cm_2**2 + C_th_2**2
                h_2 = h_02 - 0.5 * C_2_sq
                p_2 = p_02 * (h_2 / h_02) ** g_exp
                rho_2 = (
                    (cp * p_2) / (R_gas * h_2)
                    if compressible
                    else params["rho_inlet_ref"]
                )
                s_2 = s_1

            fields["h"][i, j] = h_2
            fields["T"][i, j] = h_2 / cp
            fields["h_o"][i, j] = h_02
            fields["p_o"][i, j] = p_02
            fields["p"][i, j] = p_2
            fields["rho"][i, j] = rho_2
            fields["s"][i, j] = s_2

    max_diff_rho = np.max(np.abs(fields["rho"] - rho_old))
    return max_diff_rho


def run_ree_solver(compressible=True, with_losses=True, min_sweeps=50, max_sweeps=500):
    """
    Steps 8 & 9
    Main iterative field-sweep loop. Runs at least 50 sweeps and checks convergence:
    eps_1 (Psi) <= 1e-5, eps_2 (rho) <= 1e-3, eps_3 (omega) <= 1e-3.
    """
    params, fields = initialize_project()

    for sweep in range(1, max_sweeps + 1):
        omega_old = fields["omega"].copy()

        # Step 4: Calculate vorticity omega_ij
        if compressible or with_losses:
            update_vorticity(params, fields)
        else:
            fields["omega"].fill(0.0)

        max_diff_omega = np.max(np.abs(fields["omega"] - omega_old))

        # Step 5: Update Psi_ij
        max_diff_psi = update_streamfunction_psi(params, fields)

        # Step 6: Calculate velocities Cx, Cr, Cm
        update_velocities(params, fields)

        # Step 7: Trace back thermodynamic variables along streamlines
        max_diff_rho = trace_thermodynamics(
            params, fields, compressible=compressible, with_losses=with_losses
        )

        # Step 8: Check convergence after at least 50 sweeps (Ch. 8 Page 3-4)
        if sweep >= min_sweeps:
            if (
                (max_diff_psi <= 1e-5)
                and (max_diff_rho <= 1e-3)
                and (max_diff_omega <= 1e-3)
            ):
                print(
                    f"[Converged in {sweep} sweeps] "
                    f"dPsi={max_diff_psi:.2e}, drho={max_diff_rho:.2e}, domega={max_diff_omega:.2e}"
                )
                break
        else:
            print(
                f"[Reached {max_sweeps} sweeps] "
                f"dPsi={max_diff_psi:.2e}, drho={max_diff_rho:.2e}, domega={max_diff_omega:.2e}"
            )

    return params, copy.deepcopy(fields)

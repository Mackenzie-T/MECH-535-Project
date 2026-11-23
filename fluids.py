"""
Grid set-up and flow-field updates for the axial-flow rotor throughflow solver.

All 2D fields are NumPy arrays of shape (NSTATN, NSTRM):
    index i -> axial station (z direction)
    index j -> radial node from hub (j = 0) to shroud (j = NSTRM - 1)
"""

import math
import numpy as np


def initialize_project(
    NSTATN=51,
    NSTRM=11,
    RSHROUD=0.5,
    RHUB=0.45,
    rpm=6000,
    rho_inlet=1.5,
    T_inlet=288,
    XMASS=30.4,
    DZ=0.005,
    DR=0.005,
    RCU_1=39.3,
    RCU_2=117.8,
    compressible=True,
):
    """
    Build the parameter dictionary and the initial guess for every flow field.

    The rotor occupies stations i_LE < i <= i_TE. Upstream of the rotor the swirl
    is rCu = RCU_1, downstream it is rCu = RCU_2, and inside the rotor it varies
    linearly between the two. A total loss of 0.03 is spread evenly over the rotor.
    """
    A_inlet = math.pi * (RSHROUD**2 - RHUB**2)

    params = {
        "NSTATN": NSTATN,
        "NSTRM": NSTRM,
        "i_LE": 2 * (NSTRM - 1),  # rotor leading edge station index
        "i_TE": 3 * (NSTRM - 1),  # rotor trailing edge station index
        "RADIUS": np.linspace(RHUB, RSHROUD, NSTRM),  # radial nodes, hub to shroud
        "OMEGA": rpm * math.pi / 30,  # rotational speed [rad/s]
        "CP": 1005,  # specific heat [J/kg/K]
        "RGAS": 287.0,  # gas constant [J/kg/K]
        "GAMMA": 1.4,  # ratio of specific heats
        "rho_inlet_ref": rho_inlet,  # reference density [kg/m^3]
        "T_inlet_ref": T_inlet,  # reference temperature [K]
        "DZ": DZ,
        "DR": DR,
        "A_inlet": A_inlet,
        "Cx_inlet": XMASS / (rho_inlet * A_inlet),  # uniform inlet axial velocity
        "XMASS": XMASS,  # mass flow rate [kg/s]
        "RHUB": RHUB,
        "RSHROUD": RSHROUD,
        "Z": np.linspace(0.0, (NSTATN - 1) * DZ, NSTATN),  # axial stations
        "RCU_1": RCU_1,  # inlet swirl rCu [m^2/s]
        "RCU_2": RCU_2,  # exit swirl rCu [m^2/s]
    }

    shape = (NSTATN, NSTRM)
    fields = {
        "PSI": np.ones(shape),  # normalised stream function
        "CM": np.ones(shape) * params["Cx_inlet"],  # meridional velocity
        "CZ": np.ones(shape) * params["Cx_inlet"],  # axial velocity
        "CR": np.zeros(shape),  # radial velocity
        "CU": np.zeros(shape),  # tangential (whirl) velocity
        "C": np.ones(shape),  # absolute velocity magnitude
        "OMEGLOS": np.zeros(shape),  # loss coefficient per station
        "DENSITY": np.ones(shape) * params["rho_inlet_ref"],
        "ENTROPY": np.zeros(shape),
        "BETA": np.zeros(shape),  # relative flow angle [deg]
        "PSTATIC": np.ones(shape) * 101325.0,
        "h": np.ones(shape) * 300000.0,  # static enthalpy
        "HOREL": np.ones(shape) * 310000.0,  # relative stagnation enthalpy
        "PREL": np.ones(shape) * 105000.0,  # relative stagnation pressure
        "HTOTAL": np.ones(shape) * 310000.0,  # stagnation enthalpy
        "PTOTAL": np.ones(shape) * 105000.0,  # stagnation pressure
        "TSTATIC": np.ones(shape) * T_inlet,
        "RHS": np.zeros(shape),  # vorticity source term
    }

    g_exp = params["GAMMA"] / (params["GAMMA"] - 1.0)
    # Inlet stagnation conditions: T_0 = T_inlet and rho_0 = rho_inlet
    h_o_in = params["CP"] * T_inlet
    p_o_in = rho_inlet * params["RGAS"] * T_inlet

    for i in range(NSTATN):
        in_rotor = params["i_LE"] < i <= params["i_TE"]
        for j in range(NSTRM):
            r_ij = params["RADIUS"][j]
            fields["PSI"][i, j] = (r_ij**2 - RHUB**2) / (RSHROUD**2 - RHUB**2)

            # Swirl distribution and losses
            if i <= params["i_LE"]:
                fields["CU"][i, j] = params["RCU_1"] / r_ij
            elif in_rotor:
                frac = (i - params["i_LE"]) / (params["i_TE"] - params["i_LE"])
                fields["CU"][i, j] = (
                    params["RCU_1"] + (params["RCU_2"] - params["RCU_1"]) * frac
                ) / r_ij
                fields["OMEGLOS"][i, j] = 0.03 / (params["i_TE"] - params["i_LE"])
            else:
                fields["CU"][i, j] = params["RCU_2"] / r_ij

            fields["C"][i, j] = math.sqrt(
                fields["CM"][i, j] ** 2 + fields["CU"][i, j] ** 2
            )

            # Static properties from the stagnation state
            h_stat = h_o_in - 0.5 * fields["C"][i, j] ** 2
            p_stat = p_o_in * (h_stat / h_o_in) ** g_exp

            fields["HTOTAL"][i, j] = h_o_in
            fields["PTOTAL"][i, j] = p_o_in
            fields["h"][i, j] = h_stat
            fields["TSTATIC"][i, j] = h_stat / params["CP"]
            fields["PSTATIC"][i, j] = p_stat

            if compressible:
                fields["DENSITY"][i, j] = (params["CP"] * p_stat) / (
                    params["RGAS"] * h_stat
                )
            else:
                fields["DENSITY"][i, j] = rho_inlet

            # Relative stagnation properties and initial relative flow angle
            V_u = params["OMEGA"] * r_ij - fields["CU"][i, j]
            V_sq = fields["CM"][i, j] ** 2 + V_u**2
            fields["HOREL"][i, j] = h_stat + 0.5 * V_sq
            fields["PREL"][i, j] = p_stat * (fields["HOREL"][i, j] / h_stat) ** g_exp
            fields["BETA"][i, j] = math.degrees(math.atan2(V_u, fields["CM"][i, j]))

    return params, fields


def update_vorticity(params, fields):
    """
    Compute the vorticity source term RHS at every interior node from the
    radial gradients of swirl (rCu), entropy and stagnation enthalpy.
    """
    NSTATN, NSTRM = params["NSTATN"], params["NSTRM"]
    RADIUS = params["RADIUS"]
    CU = fields["CU"]

    for i in range(1, NSTATN):
        for j in range(1, NSTRM - 1):
            r_ij = RADIUS[j]

            swirl_term = (
                CU[i, j] * (CU[i, j + 1] * RADIUS[j + 1] - CU[i, j - 1] * RADIUS[j - 1])
                / r_ij
            )
            entropy_term = fields["TSTATIC"][i, j] * (
                fields["ENTROPY"][i, j + 1] - fields["ENTROPY"][i, j - 1]
            )
            h0_term = fields["HTOTAL"][i, j + 1] - fields["HTOTAL"][i, j - 1]

            fields["RHS"][i, j] = (
                math.pi
                / (params["DR"] * params["XMASS"] * fields["CZ"][i, j])
                * (swirl_term + entropy_term - h0_term)
            )


def update_streamfunction_psi(params, fields):
    """
    One sweep of the finite-difference update for the stream function PSI at
    every interior node. Returns the largest change in PSI during the sweep.
    """
    NSTATN, NSTRM = params["NSTATN"], params["NSTRM"]
    RADIUS = params["RADIUS"]
    RHO = fields["DENSITY"]
    PSI = fields["PSI"]
    max_diff_psi = 0.0

    for i in range(1, NSTATN):
        # At the exit station, mirror the upstream node (zero axial gradient)
        i_plus = i - 1 if i == NSTATN - 1 else i + 1

        for j in range(1, NSTRM - 1):
            # rho * r averaged on the four cell faces
            rho_r1 = (RHO[i, j] + RHO[i_plus, j]) / 2 * RADIUS[j]
            rho_r2 = (RHO[i, j] + RHO[i - 1, j]) / 2 * RADIUS[j]
            rho_r3 = (RHO[i, j] + RHO[i, j + 1]) / 2 * (RADIUS[j] + RADIUS[j + 1]) / 2
            rho_r4 = (RHO[i, j] + RHO[i, j - 1]) / 2 * (RADIUS[j] + RADIUS[j - 1]) / 2

            A = (1 / rho_r1 + 1 / rho_r2 + 1 / rho_r3 + 1 / rho_r4) ** -1
            B = (
                PSI[i_plus, j] / rho_r1
                + PSI[i - 1, j] / rho_r2
                + PSI[i, j + 1] / rho_r3
                + PSI[i, j - 1] / rho_r4
            )

            psi_new = A * (B + params["DZ"] ** 2 * fields["RHS"][i, j])
            max_diff_psi = max(max_diff_psi, abs(psi_new - PSI[i, j]))
            PSI[i, j] = psi_new

    return max_diff_psi


def update_velocities(params, fields):
    """
    Recover the axial (CZ), radial (CR) and meridional (CM) velocities from
    the gradients of the stream function PSI.
    """
    NSTATN, NSTRM = params["NSTATN"], params["NSTRM"]
    DZ, DR, XMASS = params["DZ"], params["DR"], params["XMASS"]
    PSI = fields["PSI"]

    for i in range(1, NSTATN):
        i_plus = i - 1 if i == NSTATN - 1 else i + 1

        for j in range(NSTRM):
            r_ij = params["RADIUS"][j]
            # One-sided differences at the hub and shroud
            j_plus = j if j == NSTRM - 1 else j + 1
            j_minus = j if j == 0 else j - 1

            coeff = XMASS / (2 * math.pi * fields["DENSITY"][i, j] * r_ij)

            fields["CZ"][i, j] = (
                coeff * (PSI[i, j_plus] - PSI[i, j_minus]) / ((j_plus - j_minus) * DR)
            )

            # CR stays zero on the hub and shroud walls
            if 0 < j < NSTRM - 1:
                fields["CR"][i, j] = -coeff * (PSI[i_plus, j] - PSI[i - 1, j]) / (2 * DZ)

            fields["CM"][i, j] = math.sqrt(fields["CZ"][i, j] ** 2 + fields["CR"][i, j] ** 2)

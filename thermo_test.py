import numpy as np

# 1. Create dummy params for a small test grid (N=5 axial stations, M=3 radial nodes)
N = 5
M = 3
params = {
    "r_hub": 0.45,
    "r_shroud": 0.50,
    "N_rpm": 6000.0,
    "Omega": 6000.0 * 2.0 * np.pi / 60.0,
    "m_dot": 30.4,
    "T_inlet_ref": 288.0,
    "rho_inlet_ref": 1.5,
    "rC_theta_1": 39.3,
    "rC_theta_2": 117.8,
    "loss_TE": 0.03,
    "cp": 1005.0,
    "R": 287.0,
    "gamma": 1.4,
    "N": N,
    "M": M,
    "dz": 0.01,
    "dr": 0.025,
    "i_LE": 1,  # Leading edge at index 1
    "i_TE": 3,  # Trailing edge at index 3
    "r": np.linspace(0.45, 0.50, M),
    "z": np.linspace(0.0, 0.04, N),
    "Cx_inlet": 136.0,
}

# 2. Create dummy fields matching the (N, M) shape arrays
r_grid2d = np.tile(params["r"], (N, 1))

fields = {
    "Psi": np.linspace(0, 1, M)[np.newaxis, :]
    * np.ones((N, M)),  # Streamfunction 0 to 1
    "Cx": np.full((N, M), 136.0),  # Axial velocity
    "Cr": np.zeros((N, M)),  # Radial velocity
    "Cm": np.full((N, M), 136.0),  # Meridional velocity
    "rC_theta": np.full((N, M), 39.3),  # Swirl
    "C_theta": np.full((N, M), 39.3) / r_grid2d,  # Tangential velocity
    "varpi_step": np.zeros((N, M)),  # Loss step
    "rho": np.full((N, M), 1.5),  # Density
    "h": np.full((N, M), 1005.0 * 288.0),  # Enthalpy
    "h_o": np.full((N, M), 1005.0 * 288.0 + 0.5 * 136.0**2),  # Stagnation enthalpy
    "T": np.full((N, M), 288.0),  # Temperature
    "p": np.full((N, M), 100296.0),  # Static pressure
    "p_o": np.full((N, M), 150000.0),  # Stagnation pressure
    "h_o_rel": np.full((N, M), 300000.0),  # Relative stagnation enthalpy
    "p_o_rel": np.full((N, M), 150000.0),  # Relative stagnation pressure
    "s": np.zeros((N, M)),  # Entropy
    "beta": np.zeros((N, M)),  # Relative blade angle
    "omega": np.zeros((N, M)),  # Vorticity RHS
}

# Populate swirl and loss stepping inside the rotor region (i_LE < i <= i_TE)
for i in range(params["i_LE"] + 1, params["i_TE"] + 1):
    fields["varpi_step"][i, :] = params["loss_TE"] / (params["i_TE"] - params["i_LE"])
    fields["rC_theta"][i, :] = params["rC_theta_1"] + (
        params["rC_theta_2"] - params["rC_theta_1"]
    ) * ((i - params["i_LE"]) / (params["i_TE"] - params["i_LE"]))
    fields["C_theta"][i, :] = fields["rC_theta"][i, :] / params["r"]

# --- TEST STEP 7 (TRACE THERMODYNAMICS) ---
from thermodynamics import trace_thermodynamics

print("Testing Step 7 (trace_thermodynamics)...")
max_d_rho = trace_thermodynamics(params, fields, compressible=True, with_losses=True)
print(f"Step 7 executed successfully! Max density change: {max_d_rho:.6f}")
print("Updated density array shape:", fields["rho"].shape)
print("Updated density values at exit (i=4):\n", fields["rho"][-1, :])

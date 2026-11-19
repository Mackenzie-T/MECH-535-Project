import copy
import numpy as np
import math


def initialize_project(N=50, M=10, r_shroud = 0.5, r_hub = 0.45, rpm = 6000, rho_inlet = 1.5, T_inlet =288, m = 30.4, dx = 0.005, dr = 0.005, rC_theta_1 = 39.3, rC_theta_2 = 117.8):
    # Define grid size and geometry markers
    
    A_inlet = math.pi*(r_shroud**2 - r_hub**2)
    N, M = N, M  
    params = {
      "N": N,
      "M": M,
      "i_LE": 2*N,  # Rotor leading edge station index
      "i_TE": (3*N),  # Rotor trailing edge station index
      "r": np.linspace(r_shroud, r_hub, M),  # Radial coordinates from hub to casing
      "Omega": rpm*math.pi/30,  # Rotational speed (rad/s)
      "cp": 1005,  # Specific heat
      "R": 287.0,  # Gas constant
      "gamma": 1.4,  # Ratio of specific heats
      "rho_inlet_ref": rho_inlet,  # Reference density
      "T_inlet_ref": T_inlet,
      "dx": dx,
      "dr": dr,
      "A_inlet": A_inlet,
      "Cx_inlet": m/(rho_inlet* A_inlet),
    }

    # Initialize all 2D fields
               
    fields = {
      "Psi": np.ones((N, M)),
      "Cm": np.ones((N, M)) * params["Cx_inlet"], # axial/meridional 
      "Cx": np.ones((N, M)) * params["Cx_inlet"],
      "Cr": np.zeros((N, M)),
      "C_theta": np.zeros((N, M)),
      "C": np.ones((N, M)),
      "varpi_step": np.zeros((N, M)),  # Loss coefficients (or read from profile)
      "rho": np.ones((N, M)) * params["rho_inlet_ref"],
      "s": np.zeros((N, M)),
      "beta": np.zeros((N, M)),
      "p": np.ones((N, M)) * 101325.0,
      "h": np.ones((N, M)) * 300000.0,
      "h_o_rel": np.ones((N, M)) * 310000.0,
      "p_o_rel": np.ones((N, M)) * 105000.0,
      "h_o": np.ones((N, M)) * 310000.0,
      "p_o": np.ones((N, M)) * 105000.0,
      "T": np.ones((N, M)) * T_inlet,
      "omega": np.zeros((N, M)),
    }
    

    # Initialization loop 
    for i in range(N):
        in_rotor = (i > params["i_LE"]) and (i <= params["i_TE"])
        for j in range(M):
            r_ij = params["r"][j]
            fields["Psi"][i, j] = (r_ij**2 - r_hub**2)/(r_shroud**2 - r_hub**2)
            fields["C_theta"][i, j] = rC_theta_1 / r_ij
            fields["C"][i, j] = math.sqrt(fields["Cm"][i, j]**2 + fields["C_theta"][i, j]**2)
            fields["h_o"][i, j] = params["cp"]*T_inlet + fields["C"][i, j]**2/2

            if in_rotor:
              fields["varpi_step"][i, j] = 0.05/(params["i_TE"]-params["i_LE"])   

    return params, fields

def update_velocities(fields, N, M):

  for i in range(N):
    for j in range(M):
      
      if j == M: 
        j_plus = j
      else:
        j_plus = j+1
    
      if j == 0:
        j_minus = j
      else:
        j_minus = j-1
    
      if i == N: 
        i_plus = i-1
      else:
        j_plus = i+1
                
      if i == 0:
        i_minus = i
      else:
        i_minus = i-1
    
      fields["Cx"][i, j] = m/(math.pi * 2 * fields["rho"][i, j] * r_ij) * (fields["Psi"][i, j_plus] - fields["Psi"][i, j_minus])/(2*dr)
      fields["Cr"][i, j] = -m/(math.pi * 2 * fields["rho"][i, j] * r_ij) * (fields["Psi"][i_plus, j] - fields["Psi"][i_minus, j])/(2*dx)
      fields["Cm"][i, j] = math.sqrt(fields["Cx"][i, j]**2 + fields["Cr"][i, j]**2)

  return fields 

print(initialize_project())

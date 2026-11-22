import copy
import numpy as np
import math


def initialize_project(NSTATN=51, NSTRM=11, RSHROUD = 0.5, RHUB = 0.45, rpm = 6000, rho_inlet = 1.5, T_inlet =288, XMASS = 30.4, DZ = 0.005, DR = 0.005, RCU_1 = 39.3, RCU_2 = 117.8, compressible=True):
    # Define grid size and geometry markers
    
    A_inlet = math.pi*(RSHROUD**2 - RHUB**2)
    NSTATN, NSTRM = NSTATN, NSTRM
    params = {
      "NSTATN": NSTATN,
      "NSTRM": NSTRM,
      "i_LE": 2*(NSTRM-1),  # Rotor leading edge station index
      "i_TE": (3*(NSTRM-1)),  # Rotor trailing edge station index
      "RADIUS": np.linspace(RHUB, RSHROUD, NSTRM),  # Radial coordinates from hub to shroud
      "OMEGA": rpm*math.pi/30,  # Rotational speed (rad/ENTROPY)
      "CP": 1005,  # Specific heat
      "RGAS": 287.0,  # Gas constant
      "GAMMA": 1.4,  # Ratio of specific heats
      "rho_inlet_ref": rho_inlet,  # Reference density
      "T_inlet_ref": T_inlet,
      "DZ": DZ,
      "DR": DR,
      "A_inlet": A_inlet,
      "Cx_inlet": XMASS/(rho_inlet* A_inlet),
      "XMASS": XMASS,
      "RHUB": RHUB,
      "RSHROUD": RSHROUD,
      "Z": np.linspace(0.0, (NSTATN - 1) * DZ, NSTATN),
      "RCU_1": RCU_1,
      "RCU_2": RCU_2,
      "DZ": DZ
    }

    # Initialize all 2D fields
               
    fields = {
      "PSI": np.ones((NSTATN, NSTRM)),
      "CM": np.ones((NSTATN, NSTRM)) * params["Cx_inlet"], # axial/meridional 
      "CZ": np.ones((NSTATN, NSTRM)) * params["Cx_inlet"],
      "CR": np.zeros((NSTATN, NSTRM)),
      "CU": np.zeros((NSTATN, NSTRM)),
      "C": np.ones((NSTATN, NSTRM)),
      "OMEGLOS": np.zeros((NSTATN, NSTRM)),  # Loss coefficients (or read from profile)
      "DENSITY": np.ones((NSTATN, NSTRM)) * params["rho_inlet_ref"],
      "ENTROPY": np.zeros((NSTATN, NSTRM)),
      "BETA": np.zeros((NSTATN, NSTRM)),
      "PSTATIC": np.ones((NSTATN, NSTRM)) * 101325.0,
      "h": np.ones((NSTATN, NSTRM)) * 300000.0,
      "HOREL": np.ones((NSTATN, NSTRM)) * 310000.0,
      "PREL": np.ones((NSTATN, NSTRM)) * 105000.0,
      "HTOTAL": np.ones((NSTATN, NSTRM)) * 310000.0,
      "PTOTAL": np.ones((NSTATN, NSTRM)) * 105000.0,
      "TSTATIC": np.ones((NSTATN, NSTRM)) * T_inlet,
      "RHS": np.zeros((NSTATN, NSTRM)),
      #"A": np.ones((NSTATN, NSTRM)),
      #"B": np.ones((NSTATN, NSTRM)),
    }
    

    # Initialization loop 
    for i in range(NSTATN):
        in_rotor = (i > params["i_LE"]) and (i <= params["i_TE"])
        for j in range(NSTRM):
            r_ij = params["RADIUS"][j]
            fields["PSI"][i, j] = (r_ij**2 - RHUB**2)/(RSHROUD**2 - RHUB**2)

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

            # Given stagnation conditions T_o = T_inlet (288 K) and rho_o = rho_inlet (1.5 kg/m^3)
            g_exp = params["GAMMA"] / (params["GAMMA"] - 1.0)
            h_o_in = params["CP"] * T_inlet
            p_o_in = rho_inlet * params["RGAS"] * T_inlet

            # Static properties derived from stagnation state
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

            # Relative stagnation properties and initial blade angle BETA
            V_u = params["OMEGA"] * r_ij - fields["CU"][i, j]
            V_sq = fields["CM"][i, j] ** 2 + V_u**2
            fields["HOREL"][i, j] = h_stat + 0.5 * V_sq
            fields["PREL"][i, j] = (
                p_stat * (fields["HOREL"][i, j] / h_stat) ** g_exp
            )
            fields["BETA"][i, j] = math.degrees(
                math.atan2(V_u, fields["CM"][i, j])
            )

    return params, fields



def update_vorticity(params, fields):
    """
    Update the vorticity field based on the current flow properties.
    This function calculates the vorticity at each grid point using the velocity components.
    """
    NSTATN, NSTRM = params["NSTATN"], params["NSTRM"]
    RADIUS = params["RADIUS"]
    OMEGA = params["OMEGA"]

    for i in range(1, NSTATN):
        for j in range(1, NSTRM - 1):
            # Calculate the vorticity

            r_ij = RADIUS[j]

            C_theta_term = fields["CU"][i, j]*(fields["CU"][i, j + 1]*RADIUS[j + 1] - fields["CU"][i, j - 1]*RADIUS[j - 1] )/ r_ij
            s_term  = fields["TSTATIC"][i, j] * (fields["ENTROPY"][i, j + 1] - fields["ENTROPY"][i, j - 1]) 
            h0_term =  fields["HTOTAL"][i, j + 1] - fields["HTOTAL"][i, j - 1]

            fields["RHS"][i, j] = math.pi/(params["DR"]*params["XMASS"]*fields["CZ"][i, j]) * (C_theta_term + s_term - h0_term)

    #return fields

def update_streamfunction_psi(params, fields):
    """
    Update the streamfunction PSI based on the current flow properties.
    This function calculates the streamfunction at each grid point using the velocity components.
    """
    NSTATN, NSTRM = params["NSTATN"], params["NSTRM"]
    RADIUS = params["RADIUS"]
    max_diff_psi = 0.0

    for i in range(1, NSTATN):
        for j in range(1, NSTRM - 1):

          if i == NSTATN -1:
            i_plus = i-1
          else:
            i_plus = i+1
            
          rho_r1 = (fields["DENSITY"][i, j]+fields["DENSITY"][i_plus, j])/2  * RADIUS[j]
          rho_r2 = (fields["DENSITY"][i, j]+fields["DENSITY"][i-1, j])/2  * RADIUS[j]

          rho_r3 = (fields["DENSITY"][i, j]+fields["DENSITY"][i, j+1])/2  * (RADIUS[j] + RADIUS[j+1])/2
          rho_r4 = (fields["DENSITY"][i, j]+fields["DENSITY"][i, j-1])/2  * (RADIUS[j] + RADIUS[j-1])/2

          A = (rho_r1**-1 + rho_r2**-1 + rho_r3**-1 + rho_r4**-1)**-1 
          B = fields["PSI"][i_plus, j]/rho_r1 + fields["PSI"][i-1, j]/rho_r2 + fields["PSI"][i, j+1]/rho_r3 + fields["PSI"][i, j-1]/rho_r4

          psi_new = A * (B + params["DZ"] ** 2 * fields["RHS"][i, j])
          max_diff_psi = max(max_diff_psi, abs(psi_new - fields["PSI"][i, j]))
          fields["PSI"][i, j] = psi_new

    return max_diff_psi
          



def update_velocities(params, fields):

  NSTATN, NSTRM = params["NSTATN"], params["NSTRM"]

  for i in range(1, NSTATN):
    for j in range(NSTRM):

      r_ij = params["RADIUS"][j]
      DZ = params["DZ"]
      DR = params["DR"]
      XMASS = params["XMASS"]
      
      if j == NSTRM-1: 
        j_plus = j
      else:
        j_plus = j+1
    
      if j == 0:
        j_minus = j
      else:
        j_minus = j-1
    
      if i == NSTATN-1: 
        i_plus = i-1
      else:
        i_plus = i+1
                
    
      fields["CZ"][i, j] = XMASS/(math.pi * 2 * fields["DENSITY"][i, j] * r_ij) * (fields["PSI"][i, j_plus] - fields["PSI"][i, j_minus])/((j_plus - j_minus)*DR)
      
      if j != 0 and j != NSTRM-1:
        fields["CR"][i, j] = -XMASS/(math.pi * 2 * fields["DENSITY"][i, j] * r_ij) * (fields["PSI"][i_plus, j] - fields["PSI"][i-1, j])/(2*DZ)

      fields["CM"][i, j] = math.sqrt(fields["CZ"][i, j]**2 + fields["CR"][i, j]**2)

  return fields 



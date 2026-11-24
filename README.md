MECH 535 Project - Axial Rotor Throughflow Solver

Authored by:
Del Raso, Sofia 261182062
Finch, Max 261083266
Troffe, Mackenzie 261178333
Rumohr, Liam 261155845

==================================================

A 2D (axial-radial) throughflow solver for the design of an axial-flow rotor.
It solves for the stream function, velocity components and thermodynamic state
on a grid running from the inlet, through the rotor, to the exit, for two cases:

  - Part 1: compressible flow with losses
  - Part 2: incompressible, lossless flow

The results are compared with a quasi-1D analytical solution.


Files
-----

  fluids.py
      Sets up the grid, parameters and initial flow fields. Contains the
      vorticity, stream-function and velocity update steps.

  thermodynamics.py
      Traces each streamline from one station to the next to update enthalpy,
      pressure, density and entropy. Contains the main iterative solver,
      run_ree_solver().

  TE_Values.py
      Prints tables of Cz, Cr, beta and density at the rotor trailing edge for
      both cases.

  plots_and_3d.py
      Plots the trailing-edge distributions for Part 1, Part 2 and the
      analytical solution, plus a 3D view of the designed blade.

  field_heatmaps.py
      Saves heatmaps over the whole domain, with Part 1 and Part 2 side by
      side, for: radial velocity Cr, axial velocity Cz, whirl velocity C_theta,
      density and entropy.


Requirements
------------

Python 3 with NumPy and Matplotlib:

    pip install numpy matplotlib


Running
-------

Run each script from the folder that contains all five files:

    python TE_Values.py        trailing-edge tables printed to the console
    python plots_and_3d.py     trailing-edge plots and 3D blade (opens windows)
    python field_heatmaps.py   writes one PNG per field to ./heatmaps/

Each script runs the solver for both cases, which takes a few seconds. When the
solver converges it prints the number of sweeps it took and the final residuals.


Setup
-----

Default inputs are set in initialize_project() in fluids.py:

    Hub / shroud radius        0.45 m / 0.50 m
    Rotor speed                6000 rpm
    Mass flow                  30.4 kg/s
    Inlet stagnation state     T0 = 288 K, rho0 = 1.5 kg/m^3
    Swirl rC_theta             39.3 m^2/s (inlet) -> 117.8 m^2/s (exit),
                               linear through the rotor
    Total loss coefficient     0.03, spread evenly over the rotor
    Grid                       51 axial stations x 11 radial nodes,
                               dz = dr = 0.005 m
    Rotor                      leading edge at station 20,
                               trailing edge at station 30


Method
------

Each solver sweep does the following, in order:

  1. Compute the vorticity source term from the radial gradients of swirl,
     entropy and stagnation enthalpy.
  2. Update the stream function psi with a finite-difference update.
  3. Recover Cz, Cr and Cm from the gradients of psi.
  4. Trace each streamline back to the previous station and update the
     thermodynamic state:
       - Outside the rotor, stagnation enthalpy and pressure are conserved.
       - Inside the rotor, rothalpy is conserved and the loss coefficient
         lowers the relative stagnation pressure.

The solver runs at least 50 sweeps. After that it stops when the largest change
per sweep is at most 1e-5 in psi, 1e-3 in density and 1e-3 in the vorticity
term.

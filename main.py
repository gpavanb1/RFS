import matplotlib.pyplot as plt
import numpy as np
import os
from flamelet_problem import FlameletProblem
from tabulation_module import FPVTabulation, PV_H2O

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

LAMBDA_0 = 8.3
NPTS = 20
T_OX = 750.0
T_F = 290.0
P_PSI = 250.0
P_PA = P_PSI * 6894.757

# Define the problem
# Use plot_verbose=True to show flamelets each time a new solution is found
problem = FlameletProblem(LAMBDA_0, NPTS, t_ox=T_OX, t_f=T_F, p=P_PA, 
                          lmbda_max=LAMBDA_0+1.0, lmbda_threshold=1e-3, 
                          mech='burke-hydrogen.yaml', plot_verbose=False)

# Continue the problem
problem.continuation(
    newton_tol=1.0e-3,
    verbose=False,
    max_newton_steps=40,
    stepsize0=1.0e-1,
    stepsize_max=0.5,
    stepsize_aggressiveness=1e3,
    theta0=1e-1,
    adaptive_theta=True,
)

# Tabulation
tabulation = FPVTabulation(problem, pv_definition=PV_H2O)
tabulation.build_table(n_c=100)
tabulation.save_to_file('output/fpv_table.npz')

# S-curve plot (Quick check of simulation results)
plt.figure()
plt.plot(problem.chi_list, problem.Tmax_list)
plt.xlabel('Chi_st')
plt.ylabel('Tmax')
plt.title('S-curve')
plt.grid(True)
plt.savefig('output/s_curve.png')

print("Simulation complete. FPV table and S-curve plot saved to output/ folder.")
print("Use post_process.py for detailed analysis.")

import matplotlib.pyplot as plt
from flamelet_problem import FlameletProblem

LAMBDA_0 = -3.
NPTS = 30

# Define the problem
# Use plot_verbose=True to show flamelets each time a new solution is found
problem = FlameletProblem(LAMBDA_0, NPTS)

# Continue the problem
problem.continuation(
    newton_tol=1.0e-3,
    verbose=False,
    max_steps=3000,
    max_newton_steps=20,
    stepsize0=1.0e-1,
    stepsize_max=1.0e0,
    stepsize_aggressiveness=1e6,
)

plt.plot(problem.chi_list, problem.Tmax_list)
plt.show()

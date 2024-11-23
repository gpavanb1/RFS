# RFS

[Counterflow diffusion flames](https://cefrc.princeton.edu/sites/g/files/toruqf1071/files/Files/2014%20Lecture%20Notes/Pitsch/Lecture6_LamDiffFlames_2014.pdf) are a canonical problem in combustion simulations

Solution to these system of equations are used to perform large-scale calculations in a faster manner

RFS (Robust Flamelet Solver) is a mixture-fraction space based flamelet solver with arc-length continuation
capabilities

Solving in mixture-fraction space with logarithmic arc-length continuation allows for quick traversal of solution space

## How to install and execute?

* Use `conda` to setup [Cantera 3.0](https://cantera.org/install/conda-install.html#sec-install-conda) as suggested on its website

* Install the dependencies using `pip install -r requirements.txt`

Just run 
```
python main.py
```

The following program illustrates a basic example
```python
import matplotlib.pyplot as plt
from flamelet_problem import FlameletProblem

LAMBDA_0 = -3.
NPTS = 30

# Define the problem
# Use plot_verbose=True to show flamelets each time a new solution is found
problem = FlameletProblem(LAMBDA_0, NPTS)

# Continue the problem
# Entire list of arguments in `flamelet_problem.py` and same as `pacopy.euler_newton`
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
```

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1)
for any questions.

## Acknowledgements

This would not have been possible without the immense efforts of [Spitfire](https://github.com/sandialabs/Spitfire) and [Pacopy](https://github.com/sigma-py/pacopy)

Sample Hydrogen mechanism is the same one used from [Spitfire](https://github.com/sandialabs/Spitfire)

# RFS

![Made with Love in India](https://madewithlove.org.in/badge.svg)

[Counterflow diffusion flames](https://cefrc.princeton.edu/sites/g/files/toruqf1071/files/Files/2014%20Lecture%20Notes/Pitsch/Lecture6_LamDiffFlames_2014.pdf) are a canonical problem in combustion simulations

Solution to these system of equations are used to perform large-scale calculations in a faster manner

RFS (Robust Flamelet Solver) is a mixture-fraction space based flamelet solver with arc-length continuation
capabilities

Solving in mixture-fraction space with logarithmic arc-length continuation allows for quick traversal of solution space

## How to install and execute?

* Download and install [Spitfire](https://github.com/sandialabs/Spitfire) and [Pacopy 0.1.0](https://github.com/sigma-py/pacopy/tree/branch-switching)

* Install the other dependencies using `pip install -r requirements.txt`

Just run 
```
python flamelet.py
```

The following program illustrates a basic example
```python
import pacopy
import matplotlib.pyplot as plt
from flamelet_problem import FlameletProblem

LAMBDA_0 = -3.
NPTS = 30

# Define the problem
problem = FlameletProblem(LAMBDA_0, NPTS)

# Pacopy formulation
pacopy.euler_newton(
    problem, problem.u0, problem.lmbda0, 
    problem.callback, newton_tol=1.0e-3, 
    verbose=True, max_steps=3000,
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

This would not have been possible with the immense efforts of [Spitfire](https://github.com/sandialabs/Spitfire) and [Pacopy](https://github.com/sigma-py/pacopy)

Sample Hydrogen mechanism is the same one used from [Spitfire](https://github.com/sandialabs/Spitfire)
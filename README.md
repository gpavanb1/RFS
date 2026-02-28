# RFS (Robust Flamelet Solver)

[Counterflow diffusion flames](https://cefrc.princeton.edu/sites/g/files/toruqf1071/files/Files/2014%20Lecture%20Notes/Pitsch/Lecture6_LamDiffFlames_2014.pdf) are a canonical problem in combustion simulations. Solutions to these systems of equations are used to perform large-scale calculations in a faster manner through tabulation.

RFS is a mixture-fraction space based flamelet solver with robust arc-length continuation and Flamelet Progress Variable (FPV) tabulation capabilities.

## Features

- **Robust Continuation**: Mixture-fraction space solver with logarithmic arc-length continuation for quick traversal of the S-curve (from stable burning to extinction).
- **FPV Tabulation**: Build Flamelet Progress Variable (FPV) tables for temperature, mass fractions, and reaction source terms.
- **Visualization**: Tools for plotting S-curves and 2D FPV table visualizations.

## Installation

1.  Use `conda` to setup [Cantera 3.0](https://cantera.org/install/conda-install.html#sec-install-conda) as suggested on its website.
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Make sure you have [pacopy_cython](https://github.com/gpavanb1/pacopy_cython) available in the root directory. Request [here](mailto:gpavanb@gmail.com?subject=Pacopy-Cython%20Required) if required.
4.  Build the Cython extensions for the continuation solver:
    ```bash
    cd pacopy_cython
    python setup.py build_ext --inplace
    cd ..
    ```

## Usage

Run the main simulation and tabulation script:

```bash
python main.py
```

## Testing

To run the Bratu problem test for the continuation solver:

```bash
PYTHONPATH=. python pacopy_cython/test/test_bratu.py
```

### Basic Example

The following snippet (simplified from `main.py`) illustrates how to define a problem, run continuation, and build a table:

```python
from flamelet_problem import FlameletProblem
from tabulation_module import FPVTabulation, PV_H2O

# 1. Define the problem (Hydrogen flame at 250 PSI)
problem = FlameletProblem(
    lmbda0=8.3,
    npts=30,
    t_ox=750.0,
    t_f=290.0,
    p=1.72e6, # 250 PSI in Pa
    mech='burke-hydrogen.yaml'
)

# 2. Run arc-length continuation
problem.continuation(
    newton_tol=1.0e-3,
    max_newton_steps=40,
    stepsize0=1.0e-1,
    adaptive_theta=True
)

# 3. Build and save FPV table
tabulation = FPVTabulation(problem, pv_definition=PV_H2O)
tabulation.build_table(n_c=100)
tabulation.save_to_file('output/fpv_table.npz')
```

### Post-Processing

Visualize the generated tables and source terms using:

```bash
python post_process.py
```

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1).

## Acknowledgements

This would not have been possible without the immense efforts of [Spitfire](https://github.com/sandialabs/Spitfire) and [Pacopy](https://github.com/sigma-py/pacopy).
RFS uses a [Cython-optimized version of Pacopy](https://github.com/gpavanb1/pacopy_cython).
The sample Hydrogen mechanism is from [Spitfire](https://github.com/sandialabs/Spitfire).

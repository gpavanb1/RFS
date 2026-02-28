import numpy as np
import math
import matplotlib.pyplot as plt
import pacopy_cython as pacopy
import scipy.sparse as sp

from helpers.suppressor import suppress_stdout_stderr
from helpers.suppressor import suppress_stdout_stderr
from spitfire import Flamelet, FlameletSpec, ChemicalMechanismSpec


class FlameletProblem():
    def __init__(self, lmbda0, npts, t_ox=298.15, t_f=298.15, p=101325.0,
                 mech='burke-hydrogen.yaml',
                 comp_f='H2:1', plot_verbose=False, 
                 lmbda_max=2.0, lmbda_threshold=1e-4):

        # Set plot verbosity
        self.plot_verbose = plot_verbose

        # Required to reinvoke Flamelet
        self.lmbda0 = lmbda0
        self.p = p
        self.lmbda_max = lmbda_max
        self.lmbda_threshold = lmbda_threshold

        # Variables used in other functions
        self.last_stored_lmbda = -np.inf
        self.internal_step_count = 0

        # Flow details
        mech = ChemicalMechanismSpec(mech, 'gas')
        air = mech.stream(stp_air=True)
        air.TP = t_ox, p
        fuel = mech.stream('TPX', (t_f, p, comp_f))

        # Create base flamelet and steady state
        # Note that exp(lambda) = chi_st
        self.flamelet0 = Flamelet(mech_spec=mech,
                                  initial_condition='equilibrium',
                                  oxy_stream=air,
                                  fuel_stream=fuel,
                                  grid_points=npts,
                                  stoich_dissipation_rate=math.exp(lmbda0))
        self.steady_lib = self.flamelet0.compute_steady_state()
        print(f'Computed steady state for lambda0={lmbda0}')

        # Variables used in other functions
        self.num_equations = self.flamelet0._n_equations
        self.u0 = self.flamelet0._current_state
        self.chi_list = []
        self.Tmax_list = []
        self.solutions = []
        self.grid = self.flamelet0.mixfrac_grid
        self.fuel_y = fuel.Y
        self.air_y = air.Y
        self.air_T = air.T
        self.fuel_T = fuel.T
        self.species_names = mech.species_names
        self.num_species = mech.n_species

    def f(self, u, lmbda):
        """
        Evaluate RHS for adiabatic flamelet
        """
        flamelet = Flamelet(FlameletSpec(
            library_slice=self.steady_lib,
            stoich_dissipation_rate=math.exp(lmbda)))
        return flamelet._adiabatic_rhs(0., u)

    def inner(self, a, b):
        """
        Weighted inner product to balance Temperature (O(1000)) and Mass Fractions (O(1))
        """
        weights = np.ones_like(a)
        weights[::self.num_equations] = 1.0 / 2500.0
        return np.dot(a * weights, b * weights) / len(a)

    def norm2_r(self, a):
        return np.dot(a, a)

    def DD(self, y):
        """
        Second-derivative of y with respect to non-uniform grid
        """
        dydx = np.gradient(y)/np.gradient(self.grid)
        return np.gradient(dydx)/np.gradient(self.grid)

    def df_dlmbda(self, u, lmbda):
        """
        Numerical derivative with respect to lmbda using delta = 1e-5.
        Since f is linear in exp(lmbda), the exact derivative is 
        (f(lmbda + delta) - f(lmbda)) / (exp(delta) - 1).
        """
        delta = 1e-5
        return (self.f(u, lmbda + delta) - self.f(u, lmbda)) / (math.exp(delta) - 1.0)

    def jacobian_solver(self, u, lmbda, rhs):
        """
        Sparse Jacobian is mandatory for solution within reasonable times
        """
        flamelet = Flamelet(FlameletSpec(
            library_slice=self.steady_lib,
            stoich_dissipation_rate=math.exp(lmbda)))
        flamelet._current_state = u  # Set the state before computing Jacobian
        M = flamelet._adiabatic_jac_csc(u)
        
        # Suppress SuperLU/LAPACK singular matrix warnings (dgstrf info 1)
        with suppress_stdout_stderr():
            return sp.linalg.spsolve(M, rhs)

    def callback(self, k, lmbda, sol):
        """
        Callback to append current maximum temperature
        """
        if abs(lmbda - self.last_stored_lmbda) < self.lmbda_threshold:
            self.internal_step_count += 1
            print(f"\r  ... internal step {self.internal_step_count} (lambda={lmbda:.4f})", end='', flush=True)
            return

        self.last_stored_lmbda = lmbda
        self.internal_step_count = 0

        self.chi_list.append(math.exp(lmbda))
        self.solutions.append(sol.copy())
        T_list = sol[::self.num_equations]
        Tmax = T_list.max()
        self.Tmax_list.append(Tmax)

        # Print values - clear internal step line if present and move to next
        print(f"\rFlamelet {len(self.solutions)}: lambda = {lmbda:.4f}, Tmax = {Tmax:.2f}                        ")

        # Visualize current solution
        if self.plot_verbose:
            f = self.flamelet_from_state(sol)
            plt.plot(f.mixfrac_grid, f.current_temperature)
            plt.show()

        if lmbda > self.lmbda_max:
            raise StopIteration("Reached target lambda range.")

    def continuation(
        self,
        newton_tol=1.0e-6,
        max_steps=float("inf"),
        verbose=True,
        max_newton_steps=20,
        predictor_variant="tangent",
        corrector_variant="tangent",
        stepsize0=5.0e-1,
        stepsize_max=float("inf"),
        stepsize_aggressiveness=2,
        cos_alpha_min=0.9,
        theta0=1.0,
        adaptive_theta=False,
    ):
        """
        Pseudo-arclength continuation.

        Uses euler_newton from pacopy
        """
        try:
            pacopy.euler_newton(
                self,
                self.u0,
                self.lmbda0,
                self.callback,
                newton_tol=newton_tol,
                verbose=verbose,
                max_steps=max_steps,
                max_newton_steps=max_newton_steps,
                predictor_variant=predictor_variant,
                corrector_variant=corrector_variant,
                stepsize0=stepsize0,
                stepsize_max=stepsize_max,
                stepsize_aggressiveness=stepsize_aggressiveness,
                cos_alpha_min=cos_alpha_min,
                theta0=theta0,
                adaptive_theta=adaptive_theta,
            )
            print() # Final newline
        except StopIteration as e:
            print(f"Continuation stopped: {e}")

    def flamelet_from_state(self, u):
        """
        Helper function to quickly isolate flamelet object from u vector
        Used to plot 
        """
        self.flamelet0._current_state = u
        return self.flamelet0

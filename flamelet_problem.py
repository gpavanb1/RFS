import numpy as np
import math
import matplotlib.pyplot as plt
import pacopy
import scipy.sparse as sp


from spitfire import Flamelet, FlameletSpec, ChemicalMechanismSpec


class FlameletProblem():
    def __init__(self, lmbda0, npts, tf=372.,
                 mech='burke-hydrogen.yaml',
                 comp_f='H2:1', plot_verbose=False):

        # Set plot verbosity
        self.plot_verbose = plot_verbose

        # Required to reinvoke Flamelet
        self.lmbda0 = lmbda0

        # Flow details
        mech = ChemicalMechanismSpec(mech, 'gas')
        air = mech.stream(stp_air=True)
        fuel = mech.stream('TPX', (tf, air.P, comp_f))

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
        self.grid = self.flamelet0.mixfrac_grid
        self.fuel_y = fuel.Y
        self.air_y = air.Y
        self.air_T = air.T
        self.fuel_T = fuel.T

    def f(self, u, lmbda):
        """
        Evaluate RHS for adiabatic flamelet
        """
        flamelet = Flamelet(FlameletSpec(
            library_slice=self.steady_lib, stoich_dissipation_rate=math.exp(lmbda)))
        return flamelet._adiabatic_rhs(0., u)

    def inner(self, a, b):
        return np.dot(a, b)

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
        Note that $exp(\lambda) = \chi_{st}$
        This implies $df/d\lambda = (df/d\chi_{st})*(d\chi_{st}/d\lambda)$
        That simplifies to $(df/d\chi_{st})*exp(lambda)$

        For $(df/d\chi_{st})$, refer to
        https://cefrc.princeton.edu/sites/g/files/toruqf1071/files/Files/2010%20Lecture%20Notes/Norbert%20Peters/Lecture8.pdf
        Pg. 8.-11

        They trivially simplify to 0.5*T'' and (0.5/Z)*Y''
        """
        neq = self.num_equations

        # Get current temperature (by selecting every Nth element in u where N - number of species)
        # Attach boundary values to the interior array
        T_list = np.hstack((self.air_T, u[::neq], self.fuel_T))
        # We care only about interior derivatives
        Tpp = 0.5*self.DD(T_list)[1:-1]

        # Only N-1 species are solved
        Ypp = []
        for i in range(0, neq-1):
            species_list = np.hstack(
                (self.air_y[i], u[i+1::neq], self.fuel_y[i]))
            current_species_ypp = 0.5 * \
                np.divide(self.DD(species_list)[1:], self.grid[1:])
            Ypp.append(current_species_ypp[:-1])
        Ypp = np.array(Ypp)
        full_mat = np.vstack((Tpp, Ypp))
        return math.exp(lmbda)*full_mat.flatten('F')

    def jacobian_solver(self, u, lmbda, rhs):
        """
        Sparse Jacobian is mandatory for solution within reasonable times
        """
        flamelet = Flamelet(FlameletSpec(
            library_slice=self.steady_lib, stoich_dissipation_rate=math.exp(lmbda)))
        M = flamelet._adiabatic_jac_csc(u)
        return sp.linalg.spsolve(M, rhs)

    def callback(self, k, lmbda, sol):
        """
        Callback to append current maximum temperature
        """
        self.chi_list.append(math.exp(lmbda))
        T_list = sol[::self.num_equations]
        Tmax = T_list.max()
        self.Tmax_list.append(Tmax)

        # Print values
        print(lmbda, Tmax)
        print('-' * 27)

        # Visualize current solution
        if self.plot_verbose:
            f = self.flamelet_from_state(sol)
            plt.plot(f.mixfrac_grid, f.current_temperature)
            plt.show()

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

    def flamelet_from_state(self, u):
        """
        Helper function to quickly isolate flamelet object from u vector
        Used to plot 
        """
        flamelet = Flamelet(FlameletSpec(
            library_slice=self.steady_lib, stoich_dissipation_rate=math.exp(self.lmbda0)))
        flamelet._current_state = u
        return flamelet

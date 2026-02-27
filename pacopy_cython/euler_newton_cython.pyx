# -*- coding: utf-8 -*-
#
import math
import cython
from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as np

from .newton_cython import newton_cython, NewtonConvergenceError

# Type definitions for better performance
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def euler_newton_cython(
    problem,
    np.ndarray[DTYPE_t, ndim=1] u0,
    double lmbda0,
    callback,
    double max_steps=float("inf"),
    bint verbose=True,
    double newton_tol=1.0e-12,
    int max_newton_steps=5,
    str predictor_variant="tangent",
    str corrector_variant="tangent",
    double stepsize0=5.0e-1,
    double stepsize_max=float("inf"),
    double stepsize_aggressiveness=2,
    double cos_alpha_min=0.9,
    double theta0=1.0,
    bint adaptive_theta=False,
    bint converge_onto_zero_eigenvalue=False,
):
    """
    Cython-optimized pseudo-arclength continuation.

    This is a direct port of the original euler_newton function with Cython optimizations
    for better performance in the continuation loops.
    """
    cdef double lmbda = lmbda0
    cdef int k = 0
    cdef np.ndarray[DTYPE_t, ndim=1] u
    cdef np.ndarray[DTYPE_t, ndim=1] du_dlmbda, du_ds, du_ds_current, u_current
    cdef double dlmbda_ds, dlmbda_ds_current, duds2, duds2_current, ds, theta
    cdef double nrm, cos_alpha, r
    cdef double nonzero_eigval = 0.0
    cdef double tol = 1.0e-10
    cdef bint newton_success
    cdef int num_newton_steps

    # Initial Newton solve
    try:
        u, _ = newton_cython(
            lambda u: problem.f(u, lmbda),
            lambda u, rhs: problem.jacobian_solver(u, lmbda, rhs),
            problem.norm2_r,
            u0,
            tol=newton_tol,
            max_iter=max_newton_steps,
            verbose=verbose,
        )
    except NewtonConvergenceError as e:
        if verbose:
            print("No convergence for initial step.".format(lmbda))
        raise e

    if converge_onto_zero_eigenvalue:
        # Track _one_ nonzero eigenvalue.
        nonzero_eigval, _ = problem.jacobian_eigenvalue(u, lmbda)

    ds = fabs(stepsize0)
    theta = theta0

    # tangent predictor for the first step
    du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
    dlmbda_ds = 1.0 if stepsize0 > 0 else -1.0
    du_ds = du_dlmbda * dlmbda_ds

    duds2 = problem.inner(du_ds, du_ds)

    nrm = sqrt(theta ** 2 * duds2 + dlmbda_ds ** 2)
    du_ds /= nrm
    dlmbda_ds /= nrm
    duds2 /= nrm ** 2

    u_current = u.copy()
    lmbda_current = lmbda
    du_ds_current = du_ds.copy()
    dlmbda_ds_current = dlmbda_ds
    duds2_current = duds2

    callback(k, lmbda, u)
    k += 1

    while True:
        if k > max_steps:
            break

        if verbose:
            print()
            print("Step {}, stepsize: {:.3e}".format(k, ds))

        # Predictor
        u = u_current + du_ds_current * ds
        lmbda = lmbda_current + dlmbda_ds_current * ds

        # Newton corrector - call the optimized version
        u, lmbda, num_newton_steps, newton_success = _newton_corrector_cython(
            problem,
            u,
            lmbda,
            theta,
            u_current,
            lmbda_current,
            du_ds_current,
            dlmbda_ds_current,
            ds,
            corrector_variant,
            max_newton_steps,
            newton_tol,
            verbose,
        )

        if not newton_success:
            if verbose:
                print("Newton convergence failure! Restart with smaller step size.")
            ds *= 0.5
            continue

        if converge_onto_zero_eigenvalue:
            eigval, eigvec = problem.jacobian_eigenvalue(u, lmbda)
            is_zero = fabs(eigval) < tol

            if is_zero:
                if verbose:
                    print("Converged onto zero eigenvalue.")
                return eigval, eigvec
            else:
                # Check if the eigenvalue crossed the origin
                if (nonzero_eigval > 0 and eigval > 0) or (
                    nonzero_eigval < 0 and eigval < 0
                ):
                    nonzero_eigval = eigval
                else:
                    # crossed the origin!
                    if verbose:
                        print("Eigenvalue crossed origin! Restart with smaller step size.")
                    # order 1 approximation for the zero eigenvalue
                    ds *= nonzero_eigval / (nonzero_eigval - eigval)
                    continue

        # Approximate dlmbda/ds and du/ds for the next predictor step
        if predictor_variant == "tangent":
            # tangent predictor (like in natural continuation)
            du_dlmbda = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))
            # Make sure the sign of dlambda_ds is correct
            r = theta ** 2 * problem.inner(du_dlmbda, u - u_current) + (
                lmbda - lmbda_current
            )
            dlmbda_ds = 1.0 if r > 0 else -1.0
            du_ds = du_dlmbda * dlmbda_ds
        else:
            # secant predictor
            du_ds = (u - u_current) / ds
            dlmbda_ds = (lmbda - lmbda_current) / ds
            # du_lmbda not necessary here. TODO remove
            du_dlmbda = du_ds / dlmbda_ds

        # At this point, du_ds and dlmbda_ds are still unscaled so they do NOT
        # correspond to the true du/ds and dlmbda/ds yet.

        duds2 = problem.inner(du_ds, du_ds)
        cos_alpha = (
            (problem.inner(du_ds_current, du_ds) + (dlmbda_ds_current * dlmbda_ds))
            / sqrt(duds2_current + dlmbda_ds_current ** 2)
            / sqrt(duds2 + dlmbda_ds ** 2)
        )

        if cos_alpha < cos_alpha_min:
            if verbose:
                print(
                    (
                        "Angle between subsequent predictors too large (cos(alpha) = {} < {}). "
                        "Restart with smaller step size."
                    ).format(cos_alpha, cos_alpha_min)
                )
            ds *= 0.5
            continue

        nrm = sqrt(theta ** 2 * duds2 + dlmbda_ds ** 2)
        du_ds /= nrm
        dlmbda_ds /= nrm

        u_current = u.copy()
        lmbda_current = lmbda
        du_ds_current = du_ds.copy()
        # duds2_current could be retrieved by a simple division
        duds2_current = problem.inner(du_ds, du_ds)
        dlmbda_ds_current = dlmbda_ds

        if adaptive_theta:
            # See LOCA manual, equation (2.23). There are min and max safeguards that
            # prevent numerical instabilities when solving the nonlinear systems. Needs
            # more investigation.
            dlmbda_ds2_target = 0.5
            theta *= (
                fabs(dlmbda_ds)
                / sqrt(dlmbda_ds2_target)
                * sqrt((1 - dlmbda_ds2_target) / (1 - dlmbda_ds ** 2))
            )
            theta = min(1.0e1, theta)
            theta = max(1.0e-1, theta)

        callback(k, lmbda, u)
        k += 1

        # Stepsize update
        ds *= (
            1
            + stepsize_aggressiveness
            * ((max_newton_steps - num_newton_steps) / (max_newton_steps - 1)) ** 2
        )
        ds = min(stepsize_max, ds)

    return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _newton_corrector_cython(
    problem,
    np.ndarray[DTYPE_t, ndim=1] u,
    double lmbda,
    double theta,
    np.ndarray[DTYPE_t, ndim=1] u_current,
    double lmbda_current,
    np.ndarray[DTYPE_t, ndim=1] du_ds,
    double dlmbda_ds,
    double ds,
    str corrector_variant,
    int max_newton_steps,
    double newton_tol,
    bint verbose=True,
):
    """
    Cython-optimized Newton corrector for pseudo-arclength continuation.
    """
    cdef int num_newton_steps = 0
    cdef bint newton_success = False
    cdef np.ndarray[DTYPE_t, ndim=1] r, z1, z2, du
    cdef double q, dlmbda
    cdef double norm_r, cos_alpha_inner

    while True:
        r = problem.f(u, lmbda)
        if corrector_variant == "tangent":
            q = (
                theta ** 2 * problem.inner(u - u_current, du_ds)
                + (lmbda - lmbda_current) * dlmbda_ds
                - ds
            )
        else:
            q = (
                theta ** 2 * problem.inner(u - u_current, u - u_current)
                + (lmbda - lmbda_current) ** 2
                - ds ** 2
            )

        if verbose:
            print(
                "Newton norms: sqrt({:.3e} + {:.3e}) = {:.3e}".format(
                    problem.norm2_r(r), q ** 2, sqrt(problem.norm2_r(r) + q ** 2)
                )
            )
        if problem.norm2_r(r) + q ** 2 < newton_tol ** 2:
            if verbose:
                print("Newton corrector converged after {} steps.".format(num_newton_steps))
                print("lmbda = {}, <u, u> = {}".format(lmbda, problem.inner(u, u)))
            newton_success = True
            break

        if num_newton_steps >= max_newton_steps:
            break

        # Solve bordered system
        z1 = problem.jacobian_solver(u, lmbda, -r)
        z2 = problem.jacobian_solver(u, lmbda, -problem.df_dlmbda(u, lmbda))

        if corrector_variant == "tangent":
            dlmbda = -(q + theta ** 2 * problem.inner(du_ds, z1)) / (
                dlmbda_ds + theta ** 2 * problem.inner(du_ds, z2)
            )
            du = z1 + dlmbda * z2
        else:
            dlmbda = -(q + 2 * theta ** 2 * problem.inner(u - u_current, z1)) / (
                2 * (lmbda - lmbda_current)
                + 2 * theta ** 2 * problem.inner(u - u_current, z2)
            )
            du = z1 + dlmbda * z2

        u += du
        lmbda += dlmbda
        num_newton_steps += 1

    return u, lmbda, num_newton_steps, newton_success
# -*- coding: utf-8 -*-
#
import math
import cython
from libc.math cimport sqrt
import numpy as np
cimport numpy as np

# Type definitions for better performance
ctypedef np.float64_t DTYPE_t


class NewtonConvergenceError(Exception):
    pass


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def newton_cython(f, jacobian_solver, norm2, np.ndarray[DTYPE_t, ndim=1] u0,
                  double tol=1.0e-10, int max_iter=20, bint verbose=True):
    """
    Cython-optimized Newton solver.

    This is a direct port of the original newton function with Cython optimizations.
    """
    cdef np.ndarray[DTYPE_t, ndim=1] u = u0.copy()
    cdef np.ndarray[DTYPE_t, ndim=1] fu
    cdef np.ndarray[DTYPE_t, ndim=1] du
    cdef double nrm
    cdef int k = 0
    cdef bint is_converged

    fu = f(u)
    nrm = sqrt(norm2(fu))
    if verbose:
        print("||F(u)|| = {:e}".format(nrm))

    while k < max_iter:
        if nrm < tol:
            break
        du = jacobian_solver(u, -fu)
        u += du
        fu = f(u)
        nrm = sqrt(norm2(fu))
        k += 1
        if verbose:
            print("||F(u)|| = {:e}".format(nrm))

    is_converged = nrm < tol

    if not is_converged:
        raise NewtonConvergenceError(
            "Newton's method didn't converge after {} steps.".format(k)
        )

    return u, k
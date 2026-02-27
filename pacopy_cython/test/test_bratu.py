import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pacopy_cython as pacopy

class Bratu1d:
    def __init__(self, n):
        self.n = n
        self.h = 1.0 / (n + 1)
        self.x = np.linspace(self.h, 1.0 - self.h, n)
        
        # Laplacian matrix
        data = np.ones((3, n))
        data[1] = -2.0
        self.A = sp.spdiags(data, [-1, 0, 1], n, n) / self.h**2

    def f(self, u, lmbda):
        return self.A @ u + lmbda * np.exp(u)

    def df_dlmbda(self, u, lmbda):
        return np.exp(u)

    def jacobian_solver(self, u, lmbda, rhs):
        # J = A + lmbda * diag(exp(u))
        J = self.A + lmbda * sp.diags([np.exp(u)], [0])
        return spla.spsolve(J, rhs)

    def inner(self, a, b):
        return np.dot(a, b) * self.h

    def norm2_r(self, a):
        return np.dot(a, a) * self.h

def test_bratu():
    n = 100
    problem = Bratu1d(n)
    u0 = np.zeros(n)
    lmbda0 = 0.0

    lmbdas = []
    u_max = []

    def callback(k, lmbda, u):
        lmbdas.append(lmbda)
        u_max.append(np.max(u))
        print(f"Step {k}: lambda = {lmbda:.4f}, max(u) = {np.max(u):.4f}")

    print("Starting Bratu 1D continuation...")
    try:
        pacopy.euler_newton(
            problem, u0, lmbda0, callback,
            max_steps=100,
            stepsize0=0.1,
            stepsize_max=0.5,
            newton_tol=1.0e-10
        )
    except Exception as e:
        print(f"Continuation stopped: {e}")

    # The maximum value of lambda for 1D Bratu is approx 3.5138
    lmbda_max_computed = max(lmbdas)
    print(f"\nComputed max lambda: {lmbda_max_computed:.6f}")
    print("Expected max lambda: ~3.5138")
    
    # Check if we reached near the turning point
    assert abs(lmbda_max_computed - 3.5138) < 1e-2, "Did not reach turning point correctly"
    print("SUCCESS: Bratu continuation reached the turning point correctly.")

if __name__ == "__main__":
    test_bratu()

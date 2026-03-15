"""Stuart-Landau Lagrange Direct solver for QP."""
from scipy.integrate import solve_ivp
import numpy as np
import time
from src.core.base_solver import BaseQPSolver
from typing import Tuple, Dict


class StuartLandauLagrangeDirect(BaseQPSolver):
    """
    Continuous-time QP solver via Stuart-Landau oscillator dynamics
    and Arrow-Hurwicz saddle-point algorithm.
    
    Neuromorphic motivation: the ODE mimics analog neuronal dynamics —
    no matrix inversions, no line searches, purely differential equations.
    
    EXPECTED TIMING: 2000–8000ms wall clock for n=6 QPs.
    This is CORRECT behavior. Do not optimize for speed.
    
    Mathematical form (Arrow-Hurwicz saddle-point on the SL-augmented Lagrangian):
    
        dx/dt     = (μ - |x|²)x/τ_x - (Px + q)/τ_x - Aᵀ(λ_up - λ_lo)/τ_x
        dλ_up/dt  = max(0, Ax - u) / τ_λ
        dλ_lo/dt  = max(0, l - Ax) / τ_λ
    """
    
    def __init__(
        self,
        tau_x: float = 1.0,      # primal time constant
        tau_lam: float = 0.2,    # dual (multiplier) time constant
        mu: float = 0.1,         # SL bifurcation parameter
        T_solve: float = 3.0,    # solver time horizon (seconds of ODE time)
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        self.tau_x = tau_x
        self.tau_lam = tau_lam
        self.mu = mu
        self.T_solve = T_solve
        self.rtol = rtol
        self.atol = atol

    @property
    def name(self) -> str:
        return "StuartLandauLagrange"

    def _ode_rhs(self, t, y, P, q, A, l, u):
        """RHS of the ODE system."""
        n = P.shape[0]
        m = A.shape[0]
        x      = y[:n]
        lam_up = y[n:n+m]
        lam_lo = y[n+m:]

        Ax = A @ x

        # Stuart-Landau restoring term + gradient of quadratic cost
        sl_term = (self.mu - float(x @ x)) * x
        grad_f  = P @ x + q
        dual_force = A.T @ (lam_up - lam_lo)

        dx      = (sl_term - grad_f - dual_force) / self.tau_x
        dlam_up = np.maximum(0.0, Ax - u) / self.tau_lam
        dlam_lo = np.maximum(0.0, l - Ax) / self.tau_lam

        return np.concatenate([dx, dlam_up, dlam_lo])

    def solve(self, P, q, A, l, u) -> Tuple[np.ndarray, Dict]:
        """Solve the QP using ODE integration."""
        n, m = P.shape[0], A.shape[0]
        y0 = np.zeros(n + 2*m)
        
        t_wall_start = time.perf_counter()
        result = solve_ivp(
            fun=self._ode_rhs,
            t_span=(0.0, self.T_solve),
            y0=y0,
            args=(P, q, A, l, u),
            method='RK45',
            rtol=self.rtol,
            atol=self.atol,
            max_step=0.1,
            dense_output=False,
        )
        wall_ms = (time.perf_counter() - t_wall_start) * 1000.0

        x = result.y[:n, -1]
        Ax = A @ x
        viol = float(np.maximum(0, Ax - u).max() + np.maximum(0, l - Ax).max())
        obj  = float(0.5 * x @ P @ x + q @ x)

        status = "optimal" if result.success else "max_iter"
        info = {
            'solve_time_ms':   wall_ms,
            'obj_val':         obj,
            'constraint_viol': viol,
            'status':          status,
            'ode_nfev':        result.nfev,
        }
        return x, info


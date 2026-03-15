
"""OSQP-based Quadratic Programming Solver."""
import osqp
import scipy.sparse as sp
import numpy as np
import time
from src.core.base_solver import BaseQPSolver
from typing import Tuple, Dict


class OSQPSolver(BaseQPSolver):
    """OSQP wrapper. Baseline QP solver. Fast (~5–50ms for n=6)."""
    
    def __init__(self, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000, verbose=False):
        self.settings = dict(eps_abs=eps_abs, eps_rel=eps_rel,
                             max_iter=max_iter, verbose=verbose)

    @property
    def name(self) -> str:
        return "OSQP"

    def solve(self, P, q, A, l, u) -> Tuple[np.ndarray, Dict]:
        """Solve the QP using OSQP."""
        t_start = time.perf_counter()
        
        # Convert to OSQP sparse format
        P_sp = sp.csc_matrix(P)
        A_sp = sp.csc_matrix(A)
        
        # Create and setup solver
        prob = osqp.OSQP()
        prob.setup(P=P_sp, q=q, A=A_sp, l=l, u=u, **self.settings)
        
        # Solve
        result = prob.solve()
        
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        
        x = result.x
        Ax = A @ x
        viol = float(np.maximum(0, Ax - u).max() + np.maximum(0, l - Ax).max())
        obj  = float(0.5 * x @ P @ x + q @ x)
        
        status = "optimal" if result.info.status == "solved" else result.info.status
        
        info = {
            'solve_time_ms':   wall_ms,
            'obj_val':         obj,
            'constraint_viol': viol,
            'status':          status,
            'iter':            result.info.iter,
        }
        return x, info


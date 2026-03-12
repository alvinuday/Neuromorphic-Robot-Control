"""
Benchmarking Solver Wrappers
===========================

Unified interface for comparing different QP solvers:
- OSQP (reference optimal)
- iLQR (iterative local optimization)
- SL+DirectLag (neuromorphic)
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional

try:
    import osqp
    from scipy import sparse
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False
    sparse = None


class QPSolver(ABC):
    """Base class for all QP solvers."""
    
    @abstractmethod
    def solve(self, P, q, C, d, Ac, l, u, **kwargs) -> np.ndarray:
        """
        Solve quadratic program:
          min 0.5 x^T P x + q^T x
          s.t. Cx = d
               l <= Ac x <= u
        
        Returns:
            x_opt: optimal decision variables
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """Return solver diagnostics."""
        pass


class OSQPSolver(QPSolver):
    """
    OSQP solver wrapper - Reference optimal solution.
    
    OSQP is a state-of-the-art QP solver:
    - Guaranteed convergence to optimal solution
    - Handles constraints efficiently
    - Takes ~1-100ms for MPC problems
    """
    
    def __init__(self, alpha=1.0, rho=0.1, sigma=1e-6, verbose=False):
        if not OSQP_AVAILABLE:
            raise ImportError("osqp not installed. Run: pip install osqp")
        
        self.alpha = alpha
        self.rho = rho
        self.sigma = sigma
        self.verbose = verbose
        self.last_info = {}
    
    def solve(self, P, q, C, d, Ac, l, u, **kwargs) -> np.ndarray:
        """Solve using OSQP."""
        n = P.shape[0]
        m_eq = C.shape[0] if C is not None else 0
        m_ineq = Ac.shape[0] if Ac is not None else 0
        
        t_start = time.time()
        
        # Combine constraints: Cx = d, l <= Acx <= u
        # OSQP format: l_combined <= A_combined x <= u_combined
        
        if m_eq > 0 and m_ineq > 0:
            A_combined = np.vstack([C, Ac])
            l_combined = np.hstack([d, l])
            u_combined = np.hstack([d, u])
        elif m_eq > 0:
            A_combined = C
            l_combined = d
            u_combined = d
        else:
            A_combined = Ac
            l_combined = l
            u_combined = u
        
        # Convert to sparse format (OSQP requirement)
        P_sparse = sparse.csc_matrix(P)
        A_sparse = sparse.csc_matrix(A_combined) if A_combined is not None else None
        
        # Create and solve
        prob = osqp.OSQP()
        prob.setup(P=P_sparse, q=q, A=A_sparse, l=l_combined, u=u_combined,
                   alpha=self.alpha, rho=self.rho, sigma=self.sigma,
                   verbose=self.verbose)
        
        result = prob.solve()
        t_elapsed = time.time() - t_start
        
        x_opt = result.x if result.x is not None else np.zeros(n)
        
        # Compute diagnostics
        res_eq = C @ x_opt - d if m_eq > 0 else np.array([])
        res_ineq_up = np.maximum(0, Ac @ x_opt - u) if m_ineq > 0 else np.array([])
        res_ineq_lo = np.maximum(0, l - Ac @ x_opt) if m_ineq > 0 else np.array([])
        
        eq_violation = np.max(np.abs(res_eq)) if m_eq > 0 else 0.0
        ineq_violation = max(np.max(res_ineq_up) if m_ineq > 0 else 0, 
                            np.max(res_ineq_lo) if m_ineq > 0 else 0)
        
        self.last_info = {
            'solve_time': t_elapsed,
            'objective': result.info.obj_val,
            'status': result.info.status,
            'iter': result.info.iter,
            'eq_violation': eq_violation,
            'ineq_violation': ineq_violation,
            'solver': 'OSQP'
        }
        
        return x_opt
    
    def get_info(self) -> Dict:
        return self.last_info.copy()


class ILQRSolver(QPSolver):
    """
    iLQR solver wrapper - Iterative linear quadratic regulator.
    
    Uses trajectory optimization:
    - Fast local convergence
    - Takes ~10-50ms for MPC problems
    - May not be globally optimal
    """
    
    def __init__(self, max_iter=10, tolerance=1e-4):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.last_info = {}
    
    def solve(self, P, q, C, d, Ac, l, u, **kwargs) -> np.ndarray:
        """
        Simple iLQR-like solver for MPC-structured QPs.
        
        This is a simplified version. For full iLQR, use MJPC's implementation.
        """
        n = P.shape[0]
        
        t_start = time.time()
        
        # Initialize with zero
        x = np.zeros(n)
        
        # Iterative refinement
        for iteration in range(self.max_iter):
            # Gradient descent step
            grad = P @ x + q
            
            # Add constraint forces
            if C is not None:
                residual_eq = C @ x - d
                grad += C.T @ residual_eq
            
            if Ac is not None:
                slack_u = np.maximum(0, Ac @ x - u)
                slack_l = np.maximum(0, l - Ac @ x)
                grad += Ac.T @ (slack_u - slack_l)
            
            # Line search step size
            alpha = 0.1 / (1 + iteration * 0.01)
            x_new = x - alpha * grad
            
            # Box constraints
            if Ac is not None and u is not None and l is not None:
                # Clip to bounds (simplified)
                pass
            
            # Check convergence
            dx = np.linalg.norm(x_new - x)
            if dx < self.tolerance:
                x = x_new
                break
            
            x = x_new
        
        t_elapsed = time.time() - t_start
        
        # Compute diagnostics
        m_eq = C.shape[0] if C is not None else 0
        m_ineq = Ac.shape[0] if Ac is not None else 0
        
        res_eq = C @ x - d if m_eq > 0 else np.array([])
        res_ineq_up = np.maximum(0, Ac @ x - u) if m_ineq > 0 else np.array([])
        res_ineq_lo = np.maximum(0, l - Ac @ x) if m_ineq > 0 else np.array([])
        
        eq_violation = np.max(np.abs(res_eq)) if m_eq > 0 else 0.0
        ineq_violation = max(np.max(res_ineq_up) if m_ineq > 0 else 0,
                            np.max(res_ineq_lo) if m_ineq > 0 else 0)
        
        objective = 0.5 * x @ P @ x + q @ x
        
        self.last_info = {
            'solve_time': t_elapsed,
            'objective': objective,
            'iterations': iteration + 1,
            'eq_violation': eq_violation,
            'ineq_violation': ineq_violation,
            'solver': 'iLQR'
        }
        
        return x
    
    def get_info(self) -> Dict:
        return self.last_info.copy()


class NeuromorphicSolver(QPSolver):
    """
    SL+DirectLag neuromorphic solver wrapper.
    
    Our continuous-time ODE-based solver:
    - Takes ~50-200ms for MPC problems
    - Constraint satisfaction to machine precision
    - Naturally hardware-compatible
    """
    
    def __init__(self):
        import sys
        sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')
        from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
        
        self.solver = StuartLandauLagrangeDirect(
            tau_x=1.0,
            tau_lam_eq=0.1,
            tau_lam_ineq=0.5,
            T_solve=60.0,
            convergence_tol=1e-6
        )
        self.last_info = {}
    
    def solve(self, P, q, C, d, Ac, l, u, **kwargs) -> np.ndarray:
        """Solve using SL+DirectLag."""
        t_start = time.time()
        
        x_opt = self.solver.solve(
            (P, q, C, d, Ac, l, u),
            verbose=False
        )
        
        t_elapsed = time.time() - t_start
        
        info = self.solver.get_last_info()
        info['solve_time'] = t_elapsed
        info['solver'] = 'Neuromorphic'
        
        self.last_info = info
        
        return x_opt
    
    def get_info(self) -> Dict:
        return self.last_info.copy()


def create_solver(solver_type: str) -> QPSolver:
    """Factory function to create solvers."""
    if solver_type.lower() == 'osqp':
        if not OSQP_AVAILABLE:
            raise ImportError("OSQP not available. Install: pip install osqp")
        return OSQPSolver(verbose=False)
    elif solver_type.lower() == 'ilqr':
        return ILQRSolver()
    elif solver_type.lower() == 'neuromorphic':
        return NeuromorphicSolver()
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

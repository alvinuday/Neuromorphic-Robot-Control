
import numpy as np
import time
from typing import List, Dict, Optional, Any, Tuple
from src.solver.base_snn_solver import BaseSNNSolver, SNNIteration, SNNResult

class PIPGSNNSolver(BaseSNNSolver):
    """
    SNN implementation of Proportional-Integral Projected Gradient (PIPG).
    Based on Mangalore et al. (2024) and Yue et al. (2021).
    """

    def __init__(self, alpha0: float = 0.5, beta0: float = 0.05, 
                 T_anneal: int = 100, max_iter: int = 100, 
                 tol: float = 1e-4, verbose: bool = False):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.T_anneal = T_anneal
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "SNN-PIPG"

    def _to_standard_ineq(self, A: np.ndarray, l: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert l <= Ax <= u to standard inequality form A_ineq * x <= b_ineq."""
        # Standard: Ax <= u AND -Ax <= -l
        A_ineq = np.vstack([A, -A])
        b_ineq = np.concatenate([u, -l])
        # Filter out infinite constraints
        mask = np.isfinite(b_ineq)
        return A_ineq[mask], b_ineq[mask]

    def solve(self, P: np.ndarray, q: np.ndarray, 
              A: np.ndarray, l: np.ndarray, u: np.ndarray,
              x0: Optional[np.ndarray] = None,
              **kwargs) -> SNNResult:
        """
        Solve QP using PIPG dynamics.
        
        Hyperparameters from kwargs take precedence over __init__ values.
        """
        t_start = time.perf_counter()
        
        # Hyperparameters
        alpha0 = kwargs.get('alpha0', self.alpha0)
        beta0 = kwargs.get('beta0', self.beta0)
        T_ann = kwargs.get('T_anneal', self.T_anneal)
        max_iter = kwargs.get('max_iter', self.max_iter)
        
        # Convert constraints to Ax <= b
        A_pi, b_pi = self._to_standard_ineq(A, l, u)
        AT_pi = A_pi.T
        
        L = P.shape[0]
        M = A_pi.shape[0]
        
        # Initialize states
        x = x0 if x0 is not None else np.zeros(L)
        v = np.zeros(M)
        w = np.zeros(M)
        
        history = []
        
        # Initial stats
        cost = 0.5 * x @ P @ x + q @ x
        viol = np.max(A_pi @ x - b_pi) if M > 0 else 0.0
        history.append(SNNIteration(t=0, x=x.copy(), v=v.copy(), w=w.copy(), 
                                   cost=float(cost), max_viol=float(viol),
                                   alpha=alpha0, beta=beta0))

        for t in range(max_iter):
            # Annealing
            level = t // T_ann
            alpha = alpha0 / (2**level)
            beta = beta0 * (2**level)
            
            # Primal update (Gradient neuron)
            # x_t+1 = proj_X(x_t - alpha * (P*x_t + q + A' * v_t))
            grad = P @ x + q + AT_pi @ v
            x_new = x - alpha * grad
            
            # Note: The 'proj_X' in original PIPG is for box constraints on x.
            # In our condensed MPC, x is Delta U, which already has box constraints in A_ineq.
            # So proj_X is just Identity if we don't have explicit additional bounds.
            # However, for robustness, we can clip to extreme values.
            x_new = np.clip(x_new, -1e6, 1e6) 
            
            # Dual update (Constraint & Integral neurons)
            Ax_new = A_pi @ x_new
            viol_new = Ax_new - b_pi
            
            # w_t+1 = w_t + beta * viol_new
            w_new = w + beta * viol_new
            
            # v_t+1 = relu(v_t + beta * (w_new + beta * viol_new))
            # This is the PI enhancement term
            v_arg = v + beta * (w_new + beta * viol_new)
            v_new = np.maximum(0, v_arg)
            
            # Update states
            x, v, w = x_new, v_new, w_new
            
            # Stats
            cost = 0.5 * x @ P @ x + q @ x
            max_viol = np.max(A_pi @ x - b_pi) if M > 0 else 0.0
            
            history.append(SNNIteration(t=t+1, x=x.copy(), v=v.copy(), w=w.copy(),
                                       cost=float(cost), max_viol=float(max_viol),
                                       alpha=alpha, beta=beta))
            
            # Check convergence (simplified)
            if t > 10 and abs(history[-1].cost - history[-2].cost) < self.tol * 0.01:
                if max_viol < self.tol:
                    break

        solve_time = (time.perf_counter() - t_start) * 1000.0
        
        # Verify solution
        Ax = A @ x
        ineq_viol = float(np.max(np.maximum(0, Ax - u)) + np.max(np.maximum(0, l - Ax)))
        # For condensed MPC, we don't have A_eq, but we check l <= Ax <= u
        passed = bool(ineq_viol < self.tol)

        return SNNResult(
            status="optimal" if passed else "converged_with_violation",
            z_star=x,
            history=history,
            solve_time_ms=solve_time,
            objective=float(cost),
            eq_norm=0.0, # Condensed form has no equality constraints
            ineq_viol=ineq_viol,
            passed=passed
        )

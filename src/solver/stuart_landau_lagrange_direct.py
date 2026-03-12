"""
Stuart-Landau + Unconstrained Lagrange Multipliers (Simplified Phase 2-3)
=========================================================================

Simplified approach using direct (unbounded) Lagrange multipliers for both
equality and inequality constraints. This is more stable than phase encoding.

Equations:
  τ_x dx/dt = (μ_x - x²)x - Px - q - C^T λ^eq - A_c^T(λ^up - λ^lo)
  τ_eq dλ_m^eq/dt = (Cx - d)_m  [direct multiplier, unbounded]
  τ_ineq dλ_k^up/dt = max(0, (A_c x - u)_k)  [one-sided, non-negative]
  τ_ineq dλ_k^lo/dt = max(0, (l - A_c x)_k)  [one-sided, non-negative]

This implements continuous-time Arrow-Hurwicz saddle-point algorithm WITHOUT
the phase encoding complexity. Direct multipliers are simpler and more stable.
"""

import numpy as np
import time
from scipy.integrate import solve_ivp
from typing import Optional, Dict


class StuartLandauLagrangeDirect:
    """
    SL+Lagrange solver with DIRECT (unbounded) Lagrange multipliers.
    
    Simpler and more stable than phase encoding for equality constraints.
    Direct multipliers give Arrow-Hurwicz saddle-point discrete system.
    """
    
    def __init__(self,
                 tau_x: float = 1.0,
                 tau_lam_eq: float = 0.1,
                 tau_lam_ineq: float = 0.5,
                 mu_x: float = 0.0,
                 T_solve: float = 30.0,
                 convergence_tol: float = 1e-6,
                 adaptive_annealing: bool = True):
        """
        Initialize SL+Direct Lagrange solver.
        
        Args:
            tau_x: Decision variable time constant
            tau_lam_eq: Equality Lagrange multiplier time constant (FAST)
            tau_lam_ineq: Inequality Lagrange multiplier time constant (FAST)
            mu_x: SL bifurcation parameter (use 0 for pure gradient flow)
            T_solve: Total solve time
            convergence_tol: Convergence threshold
            adaptive_annealing: Time-varying time constants
        """
        self.tau_x = tau_x
        self.tau_lam_eq = tau_lam_eq
        self.tau_lam_ineq = tau_lam_ineq
        self.mu_x = mu_x
        self.T_solve = T_solve
        self.convergence_tol = convergence_tol
        self.adaptive_annealing = adaptive_annealing
        self.last_info = {}
    
    def _ode_dynamics(self, t: float, state: np.ndarray, P, q, C, d, Ac, l_vec, u_vec) -> np.ndarray:
        """
        ODE dynamics with direct Lagrange multipliers.
        
        State: [x_{1..n}, λ_m^eq_{1..m_eq}, λ_k^up_{1..m}, λ_k^lo_{1..m}]
        """
        n = P.shape[0]
        m_eq = C.shape[0] if C is not None else 0
        m = Ac.shape[0] if Ac is not None else 0
        
        # Unpack state
        x = state[:n]
        lam_eq = state[n:n+m_eq] if m_eq > 0 else np.array([])
        lam_up = state[n+m_eq:n+m_eq+m] if m > 0 else np.array([])
        lam_lo = state[n+m_eq+m:] if m > 0 else np.array([])
        
        lam_net = lam_up - lam_lo if m > 0 else np.array([])
        
        # ─────────────────────────────────────────────────────────────────────────
        # Decision Variable Dynamics
        # ─────────────────────────────────────────────────────────────────────────
        SL_restore = (self.mu_x - x**2) * x
        cost_grad = P @ x + q
        
        eq_force = np.zeros(n)
        if m_eq > 0:
            eq_force = C.T @ lam_eq
        
        ineq_force = np.zeros(n)
        if m > 0:
            ineq_force = Ac.T @ lam_net
        
        # Adaptive annealing
        tau_x_eff = self.tau_x
        if self.adaptive_annealing:
            annealing_step = int(t / 3.0)  # Anneal every 3 seconds
            tau_x_eff = self.tau_x / (1.0 + 0.1 * annealing_step)
        
        dx = (1.0 / tau_x_eff) * (SL_restore - cost_grad - eq_force - ineq_force)
        
        # ─────────────────────────────────────────────────────────────────────────
        # Equality Lagrange Multiplier Dynamics (DIRECT, unbounded)
        # ─────────────────────────────────────────────────────────────────────────
        dlam_eq = np.array([])
        if m_eq > 0:
            residual_eq = C @ x - d
            tau_eq_eff = self.tau_lam_eq
            if self.adaptive_annealing:
                tau_eq_eff = self.tau_lam_eq / (1.0 + 0.1 * annealing_step)  # Faster annealing
            dlam_eq = (1.0 / tau_eq_eff) * residual_eq
        
        # ─────────────────────────────────────────────────────────────────────────
        # Inequality Lagrange Multiplier Dynamics (amplitude, one-sided)
        # ─────────────────────────────────────────────────────────────────────────
        dlam_up = np.array([])
        dlam_lo = np.array([])
        if m > 0:
            violation_up = np.maximum(0.0, Ac @ x - u_vec)
            violation_lo = np.maximum(0.0, l_vec - Ac @ x)
            
            tau_ineq_eff = self.tau_lam_ineq
            if self.adaptive_annealing:
                tau_ineq_eff = self.tau_lam_ineq / (1.0 + 0.1 * annealing_step)
            
            dlam_up = (1.0 / tau_ineq_eff) * violation_up
            dlam_lo = (1.0 / tau_ineq_eff) * violation_lo
        
        # Concatenate derivatives
        dydt = np.concatenate([dx, dlam_eq, dlam_up, dlam_lo])
        return dydt
    
    def solve(self, qp_matrices, x0: Optional[np.ndarray] = None,
              verbose: bool = False, return_diagnostics: bool = False) -> np.ndarray:
        """
        Solve the QP using SL + Direct Lagrange Multipliers.
        
        Args:
            qp_matrices: (P, q, A_eq, b_eq, A_ineq, k_ineq) or (P, q, C, d, Ac, l, u)
            x0: Warm-start for x
            verbose: Print info
            return_diagnostics: Return full solver info
        
        Returns:
            x_star: Optimal decision variables
        """
        
        # Parse QP format
        if len(qp_matrices) == 6:
            P, q, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
            C = A_eq
            d = b_eq
            Ac = A_ineq
            l_vec = -np.inf * np.ones(len(k_ineq))
            u_vec = k_ineq
        elif len(qp_matrices) == 7:
            P, q, C, d, Ac, l_vec, u_vec = qp_matrices
        else:
            raise ValueError(f"Expected 6 or 7-tuple, got {len(qp_matrices)}")
        
        n = P.shape[0]
        m_eq = C.shape[0] if C is not None else 0
        m = Ac.shape[0] if Ac is not None else 0
        
        # Initial condition
        if x0 is None:
            x0 = np.zeros(n)
        
        lam_eq_0 = np.zeros(m_eq)
        lam_up_0 = np.zeros(m)
        lam_lo_0 = np.zeros(m)
        
        state0 = np.concatenate([x0, lam_eq_0, lam_up_0, lam_lo_0])
        
        # Convergence event
        def converged(t, y, *args):
            dydt = self._ode_dynamics(t, y, *args)
            return np.linalg.norm(dydt) - self.convergence_tol
        
        converged.terminal = True
        converged.direction = -1
        
        if verbose:
            print(f"[SL+DirectLag] n={n}, m_eq={m_eq}, m={m}")
            print(f"[SL+DirectLag] tau_x={self.tau_x}, tau_lam_eq={self.tau_lam_eq}, tau_lam_ineq={self.tau_lam_ineq}")
        
        t_start = time.time()
        
        # Solve ODE
        sol = solve_ivp(
            self._ode_dynamics,
            [0, self.T_solve],
            state0,
            args=(P, q, C, d, Ac, l_vec, u_vec),
            method='RK45',
            events=converged,
            dense_output=False,
            rtol=1e-5,
            atol=1e-7,
            max_step=0.05
        )
        
        t_elapsed = time.time() - t_start
        
        # Extract solution
        x_star = sol.y[:n, -1]
        
        # Compute diagnostics
        residual_eq = C @ x_star - d if m_eq > 0 else np.array([])
        residual_ineq_up = np.maximum(0.0, Ac @ x_star - u_vec) if m > 0 else np.array([])
        residual_ineq_lo = np.maximum(0.0, l_vec - Ac @ x_star) if m > 0 else np.array([])
        
        constraint_eq_violation = np.max(np.abs(residual_eq)) if m_eq > 0 else 0.0
        constraint_ineq_violation = max(
            np.max(residual_ineq_up) if m > 0 else 0.0,
            np.max(residual_ineq_lo) if m > 0 else 0.0
        )
        
        objective_value = 0.5 * x_star @ P @ x_star + q @ x_star
        converged_flag = (sol.status == 0)
        
        if verbose:
            print(f"[SL+DirectLag] Solved in {t_elapsed:.2f}s ({len(sol.t)} steps)")
            print(f"[SL+DirectLag] Converged: {converged_flag}")
            print(f"[SL+DirectLag] Objective: {objective_value:.6e}")
            print(f"[SL+DirectLag] Eq violation: {constraint_eq_violation:.6e}")
            print(f"[SL+DirectLag] Ineq violation: {constraint_ineq_violation:.6e}")
        
        self.last_info = {
            'objective_value': objective_value,
            'constraint_eq_violation': constraint_eq_violation,
            'constraint_ineq_violation': constraint_ineq_violation,
            'converged': converged_flag,
            'time_to_solution': t_elapsed,
            'num_steps': len(sol.t),
        }
        
        return x_star
    
    def get_last_info(self) -> Dict:
        """Return diagnostic info from last solve."""
        return self.last_info.copy()

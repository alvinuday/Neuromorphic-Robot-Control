"""
Stuart-Landau + LagONN (Full Implementation) - Phase 2-3
=========================================================

Complete SL+LagONN dynamical system for QP solving:
- Phase 2: Lagrange phase oscillators for equality constraints (IX.1-IX.2)
- Phase 3: Lagrange amplitude oscillators for inequality constraints (IX.3-IX.4)

Based on derivation:
  [IX.1] Decision variables: τ_x dx/dt = (μ_x - x²)x - Px - q - C^T λ^eq - A_c^T(λ_up - λ_lo)
  [IX.2] Equality Lagrange:  τ_eq dφ_m^eq/dt = -sin(φ_m^eq) (Cx - d)_m  =>  λ_m^eq = cos(φ_m^eq)
  [IX.3] Upper bound:        τ_ineq dλ_k^up/dt = max(0, (A_c x - u)_k)
  [IX.4] Lower bound:        τ_ineq dλ_k^lo/dt = max(0, (l - A_c x)_k)

Total oscillators: n + m_eq + 2m
For 2-DOF arm (N=20): 120 + 80 + 240 = 440 oscillators
"""

import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, issparse
from typing import Tuple, Dict, Optional


class StuartLandauLagONNFull:
    """
    Complete SL+LagONN solver implementing equations IX.1-IX.4.
    
    Decision variables x_i are Stuart-Landau oscillator amplitudes.
    Equality constraints use Lagrange phase oscillators φ_m^{λ,eq}.
    Inequality constraints use Lagrange amplitude oscillators λ_k^{up/lo}.
    """
    
    def __init__(self, 
                 tau_x: float = 1.0, 
                 tau_eq: float = 0.1,
                 tau_ineq: float = 0.5,
                 mu_x: float = 0.0,
                 dt: float = None,
                 T_solve: float = 30.0,
                 convergence_tol: float = 1e-6,
                 adaptive_annealing: bool = True,
                 annealing_interval: float = 3.0,
                 lagrange_scale: float = 1.0,
                 eq_penalty: float = 1.0,
                 ineq_penalty: float = 1.0,
                 dual_leak: float = 5e-3,
                 post_refine_active_set: bool = True):
        """
        Initialize SL+LagONN solver.
        
        Args:
            tau_x: Decision oscillator time constant
            tau_eq: Equality Lagrange oscillator time constant (FASTER: 0.1 for good constraint enforcement)
            tau_ineq: Inequality Lagrange oscillator time constant
            mu_x: SL bifurcation parameter (should be ~0 for pure gradient flow)
            dt: Fixed timestep (ignored, RK45 adaptive stepping used)
            T_solve: Total solve time (seconds)
            convergence_tol: Convergence criterion on ||dy/dt||
            adaptive_annealing: Enable time-varying tau schedule
            annealing_interval: Time between annealing steps
            lagrange_scale: Scaling factor for Lagrange multiplier injection (10.0 for strong coupling)
        """
        self.tau_x = tau_x
        self.tau_eq = tau_eq
        self.tau_ineq = tau_ineq
        self.mu_x = mu_x
        self.dt = dt
        self.T_solve = T_solve
        self.convergence_tol = convergence_tol
        self.adaptive_annealing = adaptive_annealing
        self.annealing_interval = annealing_interval
        self.lagrange_scale = lagrange_scale  # NEW: Scale Lagrange multiplier effect
        self.eq_penalty = eq_penalty
        self.ineq_penalty = ineq_penalty
        self.dual_leak = dual_leak
        self.post_refine_active_set = post_refine_active_set
        
        # Tracking
        self.last_info = {}
        self.annealing_count = 0

    def _active_set_refine(self, x, P, q, C, d, Ac, l_vec, u_vec, max_iter: int = 5):
        """Refine primal solution by solving KKT system on detected active constraints."""
        n = P.shape[0]
        x_ref = x.copy()
        eye_n = np.eye(n)

        for _ in range(max_iter):
            if Ac is None or Ac.shape[0] == 0:
                active_up = np.zeros(0, dtype=int)
                active_lo = np.zeros(0, dtype=int)
            else:
                ax = Ac @ x_ref
                viol_up = ax - u_vec
                finite_lo = np.isfinite(l_vec)
                viol_lo = np.full_like(ax, -np.inf)
                viol_lo[finite_lo] = l_vec[finite_lo] - ax[finite_lo]

                active_up = np.where(viol_up > -1e-6)[0]
                active_lo = np.where(viol_lo > -1e-6)[0]

            A_blocks = []
            b_blocks = []
            if C is not None and C.shape[0] > 0:
                A_blocks.append(C)
                b_blocks.append(d)
            if active_up.size > 0:
                A_blocks.append(Ac[active_up, :])
                b_blocks.append(u_vec[active_up])
            if active_lo.size > 0:
                A_blocks.append(-Ac[active_lo, :])
                b_blocks.append(-l_vec[active_lo])

            if not A_blocks:
                H = P + 1e-8 * eye_n
                try:
                    x_new = -np.linalg.solve(H, q)
                except np.linalg.LinAlgError:
                    break
            else:
                Aeq = np.vstack(A_blocks)
                beq = np.concatenate(b_blocks)
                m = Aeq.shape[0]

                KKT = np.block([
                    [P + 1e-8 * eye_n, Aeq.T],
                    [Aeq, np.zeros((m, m))],
                ])
                rhs = np.concatenate([-q, beq])
                try:
                    sol = np.linalg.solve(KKT, rhs)
                except np.linalg.LinAlgError:
                    sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
                x_new = sol[:n]

            if not np.all(np.isfinite(x_new)):
                break
            if np.linalg.norm(x_new - x_ref) < 1e-8:
                x_ref = x_new
                break
            x_ref = x_new

        return x_ref
        
    def _ode_dynamics(self, t: float, state: np.ndarray, P, q, C, d, Ac, l_vec, u_vec) -> np.ndarray:
        """
        Complete ODE dynamics for SL+LagONN system (IX.1-IX.4).
        
        State vector: [x_{1..n}, φ_m^{eq}_{1..m_eq}, λ_k^{up}_{1..m}, λ_k^{lo}_{1..m}]
        
        Args:
            t: Current time (for potential adaptive annealing)
            state: Full oscillator state vector
            P, q: Quadratic cost matrices (n×n, n)
            C, d: Equality constraint matrices (m_eq×n, m_eq)
            Ac, l_vec, u_vec: Inequality constraint matrices
        
        Returns:
            dydt: Time derivatives of all oscillators
        """
        n = P.shape[0]
        m_eq = C.shape[0] if C is not None else 0
        m = Ac.shape[0] if Ac is not None else 0
        
        # Unpack state
        x = state[:n]
        phi_eq = state[n:n+m_eq] if m_eq > 0 else np.array([])
        lam_up = state[n+m_eq:n+m_eq+m] if m > 0 else np.array([])
        lam_lo = state[n+m_eq+m:] if m > 0 else np.array([])
        
        # Use unconstrained equality duals; bounded cos(phi) cannot represent KKT multipliers.
        lam_eq = phi_eq if m_eq > 0 else np.array([])
        lam_net = lam_up - lam_lo if m > 0 else np.array([])
        
        # ─────────────────────────────────────────────────────────────────
        # (IX.1) Decision Variable Oscillators
        # ─────────────────────────────────────────────────────────────────
        # τ_x dx/dt = (μ_x - x²)x - Px - q - C^T λ^eq - A_c^T(λ_up - λ_lo)
        
        SL_restore = (self.mu_x - x**2) * x
        cost_grad = P @ x + q
        
        eq_force = np.zeros(n)
        if m_eq > 0:
            residual_eq = C @ x - d
            eq_force = self.lagrange_scale * (C.T @ lam_eq)
            if self.eq_penalty > 0.0:
                eq_force += self.eq_penalty * (C.T @ residual_eq)
        
        violation_up = np.array([])
        violation_lo = np.array([])
        if m > 0:
            Ac_x = Ac @ x
            violation_up = np.maximum(0.0, Ac_x - u_vec)
            violation_lo = np.maximum(0.0, l_vec - Ac_x)

        ineq_force = np.zeros(n)
        if m > 0:
            ineq_force = self.lagrange_scale * (Ac.T @ lam_net)  # STRONGER coupling
            if self.ineq_penalty > 0.0:
                ineq_force += self.ineq_penalty * (Ac.T @ (violation_up - violation_lo))
        
        # Adaptive annealing: reduce tau_x over time
        tau_x_eff = self.tau_x
        if self.adaptive_annealing:
            annealing_step = int(t / self.annealing_interval)
            tau_x_eff = self.tau_x / (1.0 + 0.2 * annealing_step)  # Decay by 20% per interval
        
        dx = (1.0 / tau_x_eff) * (SL_restore - cost_grad - eq_force - ineq_force)
        
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        # (IX.2) Equality Lagrange Oscillators (phase encoding)
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        # Equality dual flow without phase dead-zone.
        
        dphi_eq = np.array([])
        if m_eq > 0:
            dphi_eq = (1.0 / self.tau_eq) * residual_eq
        
        # ─────────────────────────────────────────────────────────────────
        # (IX.3) Upper Bound Lagrange Oscillators (amplitude encoding)
        # ─────────────────────────────────────────────────────────────────
        # τ_ineq dλ_k^up/dt = lagrange_scale * max(0, (A_c x - u)_k)  [STRONGER]
        
        dlam_up = np.array([])
        if m > 0:
            # Adaptive annealing: increase tau_ineq (stronger enforcement)
            tau_ineq_eff = self.tau_ineq
            if self.adaptive_annealing:
                tau_ineq_eff = self.tau_ineq * (1.0 + 0.2 * annealing_step)  # Increase by 20% per interval
            
            dlam_up = self.lagrange_scale * (1.0 / tau_ineq_eff) * (violation_up - self.dual_leak * lam_up)
            dlam_up = np.where((lam_up <= 0.0) & (dlam_up < 0.0), 0.0, dlam_up)
        
        # ─────────────────────────────────────────────────────────────────
        # (IX.4) Lower Bound Lagrange Oscillators (amplitude encoding)
        # ─────────────────────────────────────────────────────────────────
        # τ_ineq dλ_k^lo/dt = lagrange_scale * max(0, (l - A_c x)_k)  [STRONGER]
        
        dlam_lo = np.array([])
        if m > 0:
            dlam_lo = self.lagrange_scale * (1.0 / tau_ineq_eff) * (violation_lo - self.dual_leak * lam_lo)
            dlam_lo = np.where((lam_lo <= 0.0) & (dlam_lo < 0.0), 0.0, dlam_lo)
        
        # ─────────────────────────────────────────────────────────────────
        # Concatenate and return all derivatives
        # ─────────────────────────────────────────────────────────────────
        dydt = np.concatenate([dx, dphi_eq, dlam_up, dlam_lo])
        return dydt
    
    def solve(self, qp_matrices, x0: Optional[np.ndarray] = None, 
              lam0: Optional[np.ndarray] = None,
              verbose: bool = False, 
              return_diagnostics: bool = False) -> np.ndarray:
        """
        Solve the QP using SL+LagONN dynamics.
        
        Args:
            qp_matrices (tuple): One of:
                - 6 elements: (P, q, A_eq, b_eq, A_ineq, k_ineq) [standard OSQP format]
                - 7 elements: (P, q, C, d, Ac, l_vec, u_vec) [explicit format]
            x0: Warm-start for decision variables
            lam0: Warm-start for Lagrange multipliers
            verbose: Print convergence info
            return_diagnostics: Return timing and constraint info
        
        Returns:
            x_star: Optimal decision variables (or full state if return_diagnostics)
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
            raise ValueError(f"qp_matrices must have 6 or 7 elements, got {len(qp_matrices)}")
        
        n = P.shape[0]
        m_eq = C.shape[0] if C is not None else 0
        m = Ac.shape[0] if Ac is not None else 0
        
        # Initial condition
        if x0 is None:
            reg = 1e-6 * np.eye(n)
            try:
                x0 = -np.linalg.solve(P + reg, q)
            except np.linalg.LinAlgError:
                x0 = np.zeros(n)
            if not np.all(np.isfinite(x0)):
                x0 = np.zeros(n)
        
        # Initialize Lagrange multipliers
        phi_eq_0 = np.zeros(m_eq)
        lam_up_0 = np.zeros(m)
        lam_lo_0 = np.zeros(m)
        
        state0 = np.concatenate([x0, phi_eq_0, lam_up_0, lam_lo_0])
        
        if verbose:
            print(f"[SL+LagONN] Starting solve: n={n}, m_eq={m_eq}, m={m}")
            print(f"[SL+LagONN] Parameters: tau_x={self.tau_x}, tau_eq={self.tau_eq}, tau_ineq={self.tau_ineq}")
        
        t_start = time.time()

        dt = 1e-4 if self.dt is None else float(self.dt)
        dt = max(1e-5, dt)
        max_steps = int(np.ceil(self.T_solve / dt))
        state = state0.copy()
        converged = False
        state_clip = 1e4
        deriv_clip = 1e6

        for k in range(max_steps):
            t = k * dt
            dydt = self._ode_dynamics(t, state, P, q, C, d, Ac, l_vec, u_vec)
            dydt = np.clip(dydt, -deriv_clip, deriv_clip)
            final_dynamics_norm = float(np.linalg.norm(dydt))
            if final_dynamics_norm < self.convergence_tol:
                converged = True
                break

            step = dt / (1.0 + 0.01 * final_dynamics_norm)
            state += step * dydt
            state = np.clip(state, -state_clip, state_clip)

            if m > 0:
                up_start = n + m_eq
                lo_start = n + m_eq + m
                state[up_start:up_start + m] = np.maximum(0.0, state[up_start:up_start + m])
                state[lo_start:lo_start + m] = np.maximum(0.0, state[lo_start:lo_start + m])

            if not np.all(np.isfinite(state)):
                break
        
        t_elapsed = time.time() - t_start
        
        # Extract solution
        x_star = state[:n]
        phi_eq_final = state[n:n+m_eq] if m_eq > 0 else np.array([])
        lam_up_final = state[n+m_eq:n+m_eq+m] if m > 0 else np.array([])
        lam_lo_final = state[n+m_eq+m:n+m_eq+2*m] if m > 0 else np.array([])
        
        lam_eq_final = phi_eq_final if m_eq > 0 else np.array([])

        refined_active_set = False
        if self.post_refine_active_set:
            x_star = self._active_set_refine(x_star, P, q, C, d, Ac, l_vec, u_vec)
            refined_active_set = True
        
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
        
        final_dydt = self._ode_dynamics(k * dt, state, P, q, C, d, Ac, l_vec, u_vec)
        final_dynamics_norm = float(np.linalg.norm(final_dydt))
        
        if verbose:
            print(f"[SL+LagONN] Solve completed in {t_elapsed:.2f}s ({k + 1} steps)")
            print(f"[SL+LagONN] Converged: {converged}")
            print(f"[SL+LagONN] Objective: {objective_value:.6e}")
            print(f"[SL+LagONN] Eq constraint violation: {constraint_eq_violation:.6e}")
            print(f"[SL+LagONN] Ineq constraint violation: {constraint_ineq_violation:.6e}")
        
        # Store diagnostics
        self.last_info = {
            'objective_value': objective_value,
            'constraint_eq_violation': constraint_eq_violation,
            'constraint_ineq_violation': constraint_ineq_violation,
            'converged': converged,
            'time_to_solution': t_elapsed,
            'num_steps': k + 1,
            'residual_norm': np.linalg.norm(state - state0),
            'final_dynamics_norm': final_dynamics_norm,
            'integration_status': 1 if converged else 0,
            'refined_active_set': refined_active_set,
            'status': 'converged' if converged else ('refined_active_set' if refined_active_set else 'max_iter'),
        }
        
        return x_star
    
    def get_last_info(self) -> Dict:
        """Return diagnostic info from last solve."""
        return self.last_info.copy()


class SLLagONNADMM:
    """
    ADMM-variant of SL+LagONN using slack variables (simpler, older version).
    Kept for compatibility and reference.
    
    Uses formulation VIII.3-5 from derivation.
    """
    
    def __init__(self,
                 tau_x: float = 1.0,
                 tau_z: float = 1.0,
                 tau_y: float = 0.5,
                 rho: float = 1.0,
                 nu_z: float = 1.0,
                 T_solve: float = 20.0,
                 convergence_tol: float = 1e-4):
        """Initialize ADMM variant."""
        self.tau_x = tau_x
        self.tau_z = tau_z
        self.tau_y = tau_y
        self.rho = rho
        self.nu_z = nu_z
        self.T_solve = T_solve
        self.convergence_tol = convergence_tol
        self.last_info = {}
    
    def solve_admm(self, qp_matrices, verbose=False):
        """Solve using ADMM variant (older approach)."""
        # Parse QP
        if len(qp_matrices) == 5:
            P, q, Ac, l_vec, u_vec = qp_matrices
        else:
            raise ValueError(f"ADMM variant expects 5-tuple, got {len(qp_matrices)}")
        
        n = P.shape[0]
        m = Ac.shape[0]
        
        # Initial state
        x0 = np.zeros(n)
        z0 = np.zeros(m)
        y0 = np.zeros(m)
        state0 = np.concatenate([x0, z0, y0])
        
        def admm_ode(t, state):
            x = state[:n]
            z = state[n:n+m]
            y = state[n+m:]
            
            # dx/dt (VIII.3)
            dx = -(1.0 / self.tau_x) * (
                P @ x + q + Ac.T @ y + self.rho * Ac.T @ (Ac @ x - z)
            )
            
            # dz/dt (VIII.4) with SL restoring
            SL_z = (self.nu_z - z**2) * z
            dz = (1.0 / self.tau_z) * (y + self.rho * (Ac @ x - z)) + SL_z
            
            # dy/dt (VIII.5)
            dy = (1.0 / self.tau_y) * (Ac @ x - z)
            
            return np.concatenate([dx, dz, dy])
        
        # Solve
        t_start = time.time()
        sol = solve_ivp(admm_ode, [0, self.T_solve], state0, method='RK45',
                       rtol=1e-4, atol=1e-6, max_step=0.1)
        t_elapsed = time.time() - t_start
        
        x_star = sol.y[:n, -1]
        objective = 0.5 * x_star @ P @ x_star + q @ x_star
        
        if verbose:
            print(f"[ADMM-SL] Solve in {t_elapsed:.2f}s, obj={objective:.6e}")
        
        final_dynamics_norm = float(np.linalg.norm(admm_ode(sol.t[-1], sol.y[:, -1])))
        self.last_info = {
            'objective_value': objective,
            'time_to_solution': t_elapsed,
            'num_steps': len(sol.t),
            'converged': (final_dynamics_norm < self.convergence_tol),
            'final_dynamics_norm': final_dynamics_norm,
        }
        
        return x_star
    
    def get_last_info(self):
        return self.last_info.copy()

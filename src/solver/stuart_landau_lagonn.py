"""
Stuart-Landau + LagONN Neuromorphic QP Solver

Implements continuous-time oscillator-based QP solver based on:
- Stuart-Landau equations for decision variable encoding
- LagONN (Lagrange Oscillatory Neural Network) for constraint handling
- PIPG (Proportional-Integral Projected Gradient) algorithm flow

Reference: neuromorphic_qp_mpc_derivation.md, Sections IX-XII
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')


class StuartLandauLaGONN:
    """
    Neuromorphic QP Solver using Stuart-Landau oscillators + LagONN constraints.
    
    Solves:
        minimize    (1/2) x^T P x + q^T x
        subject to  C x = d           (equality constraints)
                    l ≤ A_c x ≤ u     (inequality constraints)
    
    State space:
        - x ∈ R^n             : decision variable oscillators
        - φ^eq ∈ R^m_eq       : equality Lagrange phases
        - λ^up ∈ R^m          : upper bound Lagrange amplitudes
        - λ^lo ∈ R^m          : lower bound Lagrange amplitudes
    
    Total oscillators: n + m_eq + 2m
    """
    
    def __init__(self, tau_x=1.0, tau_eq=0.5, tau_ineq=1.0,
                 mu_x=1.0, dt=0.01, T_solve=50.0,
                 convergence_tol=1e-4, max_steps=10000,
                 adaptive_annealing=False,
                 eq_penalty=0.0,
                 ineq_penalty=0.0,
                 dual_leak=5e-3):
        """
        Initialize solver hyperparameters.
        
        Args:
            tau_x (float): Decision oscillator time constant (IX.1)
            tau_eq (float): Equality Lagrange time constant (IX.2)
            tau_ineq (float): Inequality Lagrange time constant (IX.3-4)
            mu_x (float): SL bifurcation parameter (Hopf amplitude)
            dt (float): ODE integrator step size
            T_solve (float): Maximum solve time
            convergence_tol (float): Convergence criterion
            max_steps (int): Maximum integration steps
            adaptive_annealing (bool): Enable time-varying tau for faster convergence
        """
        self.tau_x = tau_x
        self.tau_eq = tau_eq
        self.tau_ineq = tau_ineq
        self.mu_x = mu_x
        self.dt = dt
        self.T_solve = T_solve
        self.convergence_tol = convergence_tol
        self.max_steps = max_steps
        self.adaptive_annealing = adaptive_annealing
        self.eq_penalty = eq_penalty
        self.ineq_penalty = ineq_penalty
        self.dual_leak = dual_leak
        
        # Diagnostic info
        self._last_solve_info = {}
    
    def _ode_dynamics(self, t, state, P, q, C, d, Ac, l_vec, u_vec, params):
        """
        Right-hand side of ODE system (Equations IX.1-IX.4).
        
        State vector structure:
            state[0:n]                    -> x (decision variables)
            state[n:n+m_eq]               -> φ^eq (equality phase angles)
            state[n+m_eq:n+m_eq+m]        -> λ^up (upper bound amplitudes)
            state[n+m_eq+m:]              -> λ^lo (lower bound amplitudes)
        """
        n = params['n']
        m_eq = params['m_eq']
        m = params['m']
        
        # Adaptive time-scale annealing (optional)
        tau_x_t = self.tau_x
        tau_ineq_t = self.tau_ineq
        if self.adaptive_annealing and t > 0:
            anneal_factor = 1.0 / (1.0 + 0.01 * t)  # decay tau_x
            tau_x_t = self.tau_x * anneal_factor
            tau_ineq_t = self.tau_ineq / anneal_factor  # grow tau_ineq penalty
        
        # Extract state components
        x = state[:n]
        phi_eq = state[n:n+m_eq] if m_eq > 0 else np.array([])
        lam_up = state[n+m_eq:n+m_eq+m]
        lam_lo = state[n+m_eq+m:n+m_eq+2*m]
        
        # Treat phi_eq as unconstrained equality duals.
        # The legacy cos(phi) encoding is bounded and cannot represent large KKT multipliers.
        lam_eq = phi_eq if m_eq > 0 else np.array([])
        lam_net = lam_up - lam_lo  # net inequality force
        
        # ─── Equation IX.1: Decision Variable Oscillators ─────────────────────
        # τ_x dx_i/dt = (μ_x - x_i²) x_i - (Px+q)_i - (C^T λ^eq)_i - (A_c^T λ^net)_i
        # 
        # NOTE: For QP solving, use mu_x ≈ 0 to avoid bias (Section X.3 of derivation)
        # This gives pure gradient descent: τ_x dx/dt = -(Px+q) - constraint corrections
        # The SL restoring term is optional; we use it only for box constraint enforcement
        
        if self.mu_x < 1e-6:
            # Pure gradient flow (no SL bias)
            SL_restore = np.zeros(n)
        else:
            SL_restore = (self.mu_x - x**2) * x  # Stuart-Landau restoring term
        
        cost_grad = P @ x + q
        
        eq_correction = np.zeros(n)
        if m_eq > 0:
            eq_residual = C @ x - d
            eq_correction = C.T @ lam_eq + self.eq_penalty * (C.T @ eq_residual)
        
        Ac_x = Ac @ x
        viol_up = np.maximum(0.0, Ac_x - u_vec)
        viol_lo = np.maximum(0.0, l_vec - Ac_x)

        ineq_correction = Ac.T @ lam_net
        if m > 0 and self.ineq_penalty > 0.0:
            ineq_correction += self.ineq_penalty * (Ac.T @ (viol_up - viol_lo))
        
        dx = (1.0 / tau_x_t) * (SL_restore - cost_grad - eq_correction - ineq_correction)
        
        # Equality dual flow without phase dead-zone.
        # Using -sin(phi)*residual can freeze at phi=0 and prevents convergence.
        
        dphi = np.array([])
        if m_eq > 0:
            dphi = (1.0 / self.tau_eq) * eq_residual
        
        # ─── Equations IX.3-4: Inequality Lagrange Oscillators ────────────────
        # τ_ineq dλ_k^up/dt = max(0, (A_c x - u)_k)   [ReLU]
        # τ_ineq dλ_k^lo/dt = max(0, (l - A_c x)_k)   [ReLU]
        
        # Projected-leaky dual dynamics to avoid irreversible windup.
        dlam_up = (1.0 / tau_ineq_t) * (viol_up - self.dual_leak * lam_up)
        dlam_lo = (1.0 / tau_ineq_t) * (viol_lo - self.dual_leak * lam_lo)
        dlam_up = np.where((lam_up <= 0.0) & (dlam_up < 0.0), 0.0, dlam_up)
        dlam_lo = np.where((lam_lo <= 0.0) & (dlam_lo < 0.0), 0.0, dlam_lo)
        
        # ─── Concatenate all derivatives ──────────────────────────────────────
        return np.concatenate([dx, dphi, dlam_up, dlam_lo])
    
    def _convergence_criterion(self, state, dydt, params):
        """
        Check if system has converged (all continuous derivatives near zero).
        """
        norm_dynamics = np.linalg.norm(dydt)
        return norm_dynamics < self.convergence_tol
    
    def solve(self, qp_matrices, x0=None, lam0=None, 
              verbose=False, return_diagnostics=False):
        """
        Solve the QP using Stuart-Landau + LagONN dynamics.
        
        Args:
            qp_matrices (tuple): Can be:
                - 5 elements: (P, q, Ac, l_vec, u_vec) - only inequality constraints
                - 6 elements: (P, q, A_eq, b_eq, A_ineq, k_ineq) - standard OSQP format
                - 7 elements: (P, q, C, d, Ac, l_vec, u_vec) - explicit equality and inequality
            x0 (ndarray): Warm-start for decision variables
            lam0 (ndarray): Warm-start for Lagrange multipliers
            verbose (bool): Print convergence info
            return_diagnostics (bool): Return timing and convergence info
        
        Returns:
            x_star (ndarray): Optimal decision variables
            lam_star (ndarray): Optimal Lagrange multipliers (if return_diagnostics)
            info (dict): Convergence info (if return_diagnostics)
        """
        
        # Parse QP matrices - support multiple formats
        if len(qp_matrices) == 5:
            # Format: (P, q, Ac, l_vec, u_vec) - inequality only
            P, q, Ac, l_vec, u_vec = qp_matrices
            C = np.zeros((0, P.shape[0]))
            d = np.zeros(0)
            m_eq = 0
        elif len(qp_matrices) == 6:
            # Format: (P, q, A_eq, b_eq, A_ineq, k_ineq) - standard OSQP format
            P, q, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
            C = A_eq
            d = b_eq
            Ac = A_ineq
            l_vec = -np.inf * np.ones(len(k_ineq))  # One-sided inequality: Ac x <= k_ineq
            u_vec = k_ineq
            m_eq = C.shape[0]
        elif len(qp_matrices) == 7:
            # Format: (P, q, C, d, Ac, l_vec, u_vec) - both equality and inequality
            P, q, C, d, Ac, l_vec, u_vec = qp_matrices
            m_eq = C.shape[0]
        else:
            raise ValueError(f"qp_matrices must have 5, 6, or 7 elements, got {len(qp_matrices)}")
        
        n = P.shape[0]
        m = Ac.shape[0]
        
        # Initialize state
        if x0 is None:
            # Warm start from unconstrained quadratic minimizer.
            reg = 1e-6 * np.eye(n)
            try:
                x0 = -np.linalg.solve(P + reg, q)
            except np.linalg.LinAlgError:
                x0 = np.zeros(n)
            if not np.all(np.isfinite(x0)):
                x0 = np.zeros(n)
        else:
            x0 = np.asarray(x0)
        
        if lam0 is None:
            phi_eq0 = np.zeros(m_eq)
            lam_up0 = np.zeros(m)
            lam_lo0 = np.zeros(m)
        else:
            # Unpack: lam0 = [phi_eq, lam_up, lam_lo]
            lam0 = np.asarray(lam0)
            if len(lam0) != m_eq + 2*m:
                raise ValueError(f"lam0 has wrong length: {len(lam0)} vs {m_eq + 2*m}")
            phi_eq0 = lam0[:m_eq]
            lam_up0 = lam0[m_eq:m_eq+m]
            lam_lo0 = lam0[m_eq+m:]
        
        state0 = np.concatenate([x0, phi_eq0, lam_up0, lam_lo0])
        
        # Prepare ODE parameters
        params = {'n': n, 'm_eq': m_eq, 'm': m}
        
        # Fixed-step projected integration is much faster and avoids RK dense-output overhead.
        dt = max(1e-5, min(1e-3, float(self.dt)))
        max_steps = min(self.max_steps, int(np.ceil(self.T_solve / dt)))
        state = state0.copy()
        converged = False
        final_dyn_norm = np.inf
        state_clip = 1e4
        deriv_clip = 1e6

        for k in range(max_steps):
            t = k * dt
            dydt = self._ode_dynamics(t, state, P, q, C, d, Ac, l_vec, u_vec, params)
            dydt = np.clip(dydt, -deriv_clip, deriv_clip)
            final_dyn_norm = float(np.linalg.norm(dydt))
            if final_dyn_norm < self.convergence_tol:
                converged = True
                break

            step = dt / (1.0 + 0.01 * final_dyn_norm)
            state += step * dydt
            state = np.clip(state, -state_clip, state_clip)

            # Project inequality dual amplitudes to nonnegative orthant.
            if m > 0:
                up_start = n + m_eq
                lo_start = n + m_eq + m
                state[up_start:up_start + m] = np.maximum(0.0, state[up_start:up_start + m])
                state[lo_start:lo_start + m] = np.maximum(0.0, state[lo_start:lo_start + m])

            if not np.all(np.isfinite(state)):
                break

        # Extract final state
        x_star = state[:n]
        if m_eq > 0:
            phi_eq_star = state[n:n + m_eq]
        else:
            phi_eq_star = np.array([])

        lam_up_star = state[n + m_eq:n + m_eq + m]
        lam_lo_star = state[n + m_eq + m:n + m_eq + 2 * m]

        num_steps = k + 1
        time_to_solution = num_steps * dt

        # Compute final constraint violations and objective
        constr_eq_viol = np.linalg.norm(C @ x_star - d) if m_eq > 0 else 0.0
        constr_ineq_viol = np.max(np.concatenate([
            np.maximum(0.0, Ac @ x_star - u_vec),
            np.maximum(0.0, l_vec - Ac @ x_star)
        ])) if m > 0 else 0.0

        objective = 0.5 * x_star @ P @ x_star + q @ x_star

        # Store diagnostics
        self._last_solve_info = {
            'time_to_solution': time_to_solution,
            'num_steps': num_steps,
            'converged': converged,
            'constraint_eq_violation': constr_eq_viol,
            'constraint_ineq_violation': constr_ineq_viol,
            'objective_value': objective,
            'final_dynamics_norm': final_dyn_norm,
            'integration_status': 1 if converged else 0,
        }

        if verbose:
            print(f"✓ Solved in {time_to_solution:.4f}s ({num_steps} steps)")
            print(f"  Converged: {converged}")
            print(f"  Objective: {objective:.6e}")
            print(f"  Eq constraint violation: {constr_eq_viol:.6e}")
            print(f"  Ineq constraint violation: {constr_ineq_viol:.6e}")

        if return_diagnostics:
            lam_star = np.concatenate([phi_eq_star, lam_up_star, lam_lo_star])
            return x_star, lam_star, self._last_solve_info
        return x_star
    
    def get_last_info(self):
        """Return diagnostic information from last solve."""
        return self._last_solve_info.copy()


class SLLaGONNADMM(StuartLandauLaGONN):
    """
    ADMM-aligned variant (simpler, recommended for initial testing).
    
    Uses slack variables z = A_c x with soft box constraints via SL restoring.
    Augmented Lagrangian: L_ρ(x,z,y) = f(x) + y^T(A_c x - z) + (ρ/2)||A_c x - z||²
    
    State: x, z (slack with SL), y (ADMM dual)
    """
    
    def __init__(self, tau_x=1.0, tau_z=1.0, tau_y=0.5, 
                 rho=1.0, nu_z=1.0, **kwargs):
        """
        Initialize ADMM variant.
        
        Args:
            tau_x (float): Decision variable time constant
            tau_z (float): Slack variable time constant
            tau_y (float): Dual variable time constant
            rho (float): ADMM penalty parameter
            nu_z (float): SL bifurcation for slack variables
            **kwargs: Passed to parent StuartLandauLaGONN
        """
        super().__init__(**kwargs)
        self.tau_z = tau_z
        self.tau_y = tau_y
        self.rho = rho
        self.nu_z = nu_z
    
    def _ode_dynamics_admm(self, t, state, P, q, Ac, l_vec, u_vec, params):
        """
        ADMM-aligned ODE (Equations VIII.3-5 from derivation).
        
        State: [x, z, y]
        """
        n = params['n']
        m = params['m']
        
        # Extract state
        x = state[:n]
        z = state[n:n+m]
        y = state[n+m:]
        
        # ─── (VIII.3): x gradient flow + ADMM coupling ────────────────────────
        # τ_x dx/dt = -(Px + q + A_c^T y + ρ A_c^T (A_c x - z))
        #           = -(Px + q + A_c^T (y + ρ(A_c x - z)))
        
        cost_grad = P @ x + q
        dual_coupling = Ac.T @ (y + self.rho * (Ac @ x - z))
        dx = -(1.0 / self.tau_x) * (cost_grad + dual_coupling)
        
        # ─── (VIII.4): z SL oscillator with soft box constraint ──────────────
        # z has soft bounds [l_k, u_k] via SL restoring: (ν_z - z²) z
        # τ_z dz/dt = (ν_z - z²)z + (y + ρ(A_c x - z))
        
        SL_z = (self.nu_z - z**2) * z  # soft box restoring
        dz_dual = y + self.rho * (Ac @ x - z)
        dz = (1.0 / self.tau_z) * (SL_z + dz_dual)
        
        # ─── (VIII.5): y dual ascent (pure integral) ──────────────────────────
        # τ_y dy/dt = A_c x - z
        
        dy = (1.0 / self.tau_y) * (Ac @ x - z)
        
        return np.concatenate([dx, dz, dy])
    
    def solve_admm(self, qp_matrices, x0=None, z0=None, y0=None,
                   verbose=False, return_diagnostics=False):
        """
        Solve ADMM-variant QP.
        
        Args:
            qp_matrices (tuple): (P, q, Ac, l_vec, u_vec)
            x0, z0, y0: Warm-start values
            verbose, return_diagnostics: As in parent solve()
        
        Returns:
            x_star: Optimal decision variables
        """
        P, q, Ac, l_vec, u_vec = qp_matrices
        
        n = P.shape[0]
        m = Ac.shape[0]
        
        # Initialize state
        if x0 is None:
            x0 = np.zeros(n)
        if z0 is None:
            z0 = np.zeros(m)
        if y0 is None:
            y0 = np.zeros(m)
        
        state0 = np.concatenate([x0, z0, y0])
        
        # Event for convergence: ||x - z|| + ||dx/dt|| small
        def event_converged_admm(t, state, *args):
            dydt = self._ode_dynamics_admm(t, state, *args)
            n_vars = len(dydt) // 3 * 1  # extract x size
            Ac_x = Ac @ state[:n]
            residual_coupling = np.linalg.norm(Ac_x - state[n:n+m])
            residual_dynamics = np.linalg.norm(dydt)
            return residual_coupling + residual_dynamics - self.convergence_tol
        
        event_converged_admm.terminal = True
        event_converged_admm.direction = -1
        
        # Solve ODE
        params = {'n': n, 'm': m}
        
        try:
            sol = solve_ivp(
                self._ode_dynamics_admm,
                [0, self.T_solve],
                state0,
                args=(P, q, Ac, l_vec, u_vec, params),
                method='RK45',
                events=event_converged_admm,
                dense_output=True,
                rtol=1e-4,
                atol=1e-6,
                max_step=self.dt
            )
            
            x_star = sol.y[:n, -1]
            z_star = sol.y[n:n+m, -1]
            y_star = sol.y[n+m:, -1]
            
            time_to_solution = sol.t[-1]
            num_steps = len(sol.t)
            converged = sol.status == 0
            
            # Diagnostics
            Ac_x = Ac @ x_star
            coupling_residual = np.linalg.norm(Ac_x - z_star)
            ineq_viol = np.max(np.concatenate([
                np.maximum(0.0, Ac_x - u_vec),
                np.maximum(0.0, l_vec - Ac_x)
            ]))
            objective = 0.5 * x_star @ P @ x_star + q @ x_star
            
            self._last_solve_info = {
                'time_to_solution': time_to_solution,
                'num_steps': num_steps,
                'converged': converged,
                'coupling_residual': coupling_residual,
                'constraint_ineq_violation': ineq_viol,
                'objective_value': objective,
            }
            
            if verbose:
                print(f"✓ Solved (ADMM) in {time_to_solution:.4f}s ({num_steps} steps)")
                print(f"  Converged: {converged}")
                print(f"  Objective: {objective:.6e}")
                print(f"  Coupling residual ||A_c x - z||: {coupling_residual:.6e}")
                print(f"  Ineq constraint violation: {ineq_viol:.6e}")
            
            if return_diagnostics:
                lam_star = np.concatenate([z_star, y_star])
                return x_star, lam_star, self._last_solve_info
            else:
                return x_star
        
        except Exception as e:
            if verbose:
                print(f"✗ ADMM solver failed: {e}")
            raise

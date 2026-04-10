"""Stuart-Landau Lagrange Direct solver for QP."""
import numpy as np
import time
import osqp
import scipy.sparse as sp
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
    
        dx/dt      = (μ - |x|²)x/τ_x - (Px + q)/τ_x
                     - A_eqᵀ λ_eq / τ_λ - A_inᵀ(λ_up - λ_lo)/τ_λ
        dλ_eq/dt   = (A_eq x - b_eq) / τ_λ
        dλ_up/dt   = max(0, A_in x - u_in) / τ_λ
        dλ_lo/dt   = max(0, l_in - A_in x) / τ_λ

    Notes:
    - Equality rows are detected from constraints where l[i] and u[i] are both finite
      and |u[i]-l[i]| < eq_tol.
    - This improves stability for MPC QPs that include many hard dynamics equalities.
    """
    
    def __init__(
        self,
        tau_x: float = 1.0,      # primal time constant
        tau_lam: float = 0.2,    # dual (multiplier) time constant
        mu: float = 0.1,         # SL bifurcation parameter
        T_solve: float = 3.0,    # solver time horizon (seconds of ODE time)
        rtol: float = 1e-4,
        atol: float = 1e-6,
        eq_tol: float = 1e-9,
        dt: float = 1e-3,
        convergence_tol: float = 1e-5,
        state_clip: float = 1e3,
        constraint_penalty: float = 10.0,
        damping: float = 0.05,
        use_dual: bool = False,
        use_pipg_ineq: bool = False,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        annealing_interval: float = 0.25,
        alpha_min: float = 1e-3,
        beta_max: float = 1e3,
        fallback_to_osqp: bool = False,
        fallback_violation: float = 1e-2,
    ):
        self.tau_x = tau_x
        self.tau_lam = tau_lam
        self.mu = mu
        self.T_solve = T_solve
        self.rtol = rtol
        self.atol = atol
        self.eq_tol = eq_tol
        self.dt = dt
        self.convergence_tol = convergence_tol
        self.state_clip = state_clip
        self.constraint_penalty = constraint_penalty
        self.damping = damping
        self.use_dual = use_dual
        self.use_pipg_ineq = use_pipg_ineq
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.annealing_interval = annealing_interval
        self.alpha_min = alpha_min
        self.beta_max = beta_max
        self.fallback_to_osqp = fallback_to_osqp
        self.fallback_violation = fallback_violation

    def _solve_with_osqp_fallback(self, P, q, A, l, u):
        """Solve QP with OSQP as a robustness fallback."""
        prob = osqp.OSQP()
        prob.setup(
            P=sp.csc_matrix(P),
            q=q,
            A=sp.csc_matrix(A),
            l=l,
            u=u,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=10000,
            verbose=False,
        )
        res = prob.solve()
        return res.x, res.info.status

    @property
    def name(self) -> str:
        return "StuartLandauLagrange"

    def _ode_rhs(self, t, y, P, q, A_eq, b_eq, A_in, l_in, u_in):
        """RHS of the ODE system."""
        n = P.shape[0]
        m_eq = A_eq.shape[0]
        m_in = A_in.shape[0]

        x      = y[:n]
        lam_eq = y[n:n + m_eq]

        if self.use_pipg_ineq:
            w_up = y[n + m_eq:n + m_eq + m_in]
            w_lo = y[n + m_eq + m_in:n + m_eq + 2 * m_in]
            lam_up = np.maximum(0.0, w_up)
            lam_lo = np.maximum(0.0, w_lo)
        else:
            lam_up = y[n + m_eq:n + m_eq + m_in]
            lam_lo = y[n + m_eq + m_in:]

        grad_scale = getattr(self, "_grad_scale", 1.0)
        alpha_eff = getattr(self, "_alpha_eff", self.alpha0)
        beta_eff = getattr(self, "_beta_eff", self.beta0)

        # Stuart-Landau restoring term + gradient of quadratic cost
        sl_term = (self.mu - float(x @ x)) * x
        grad_f  = (P @ x + q) / grad_scale
        dual_force_eq = (A_eq.T @ lam_eq) / grad_scale if (self.use_dual and m_eq > 0) else 0.0
        dual_force_in = (A_in.T @ (lam_up - lam_lo)) / grad_scale if (self.use_dual and m_in > 0) else 0.0

        eq_res = A_eq @ x - b_eq if m_eq > 0 else np.zeros(0)
        ineq_up = np.maximum(0.0, A_in @ x - u_in) if m_in > 0 else np.zeros(0)
        ineq_lo = np.maximum(0.0, l_in - A_in @ x) if m_in > 0 else np.zeros(0)

        penalty_force = 0.0
        if m_eq > 0:
            penalty_force = penalty_force + A_eq.T @ eq_res
        if m_in > 0:
            penalty_force = penalty_force + A_in.T @ (ineq_up - ineq_lo)
        penalty_force = penalty_force / grad_scale

        dx = (
            sl_term
            - grad_f
            - dual_force_eq
            - dual_force_in
            - self.constraint_penalty * penalty_force
            - self.damping * x
        ) / self.tau_x

        dx *= alpha_eff

        if self.use_dual and m_eq > 0:
            dlam_eq = eq_res / self.tau_lam
        else:
            dlam_eq = np.zeros(m_eq)

        if self.use_pipg_ineq and m_in > 0:
            # Loihi/PIPG-style integral+projection channel:
            # w_dot = beta * violation, v = relu(w + beta * violation)
            res_up = A_in @ x - u_in
            res_lo = l_in - A_in @ x
            v_up = np.maximum(0.0, w_up + beta_eff * res_up)
            v_lo = np.maximum(0.0, w_lo + beta_eff * res_lo)
            v_force = (A_in.T @ (v_up - v_lo)) / grad_scale
            dx -= (alpha_eff / self.tau_x) * v_force

            dlam_up = beta_eff * res_up / self.tau_lam
            dlam_lo = beta_eff * res_lo / self.tau_lam
        elif self.use_dual and m_in > 0:
            dlam_up = ineq_up / self.tau_lam
            dlam_lo = ineq_lo / self.tau_lam
        else:
            dlam_up = np.zeros(m_in)
            dlam_lo = np.zeros(m_in)

        return np.concatenate([dx, dlam_eq, dlam_up, dlam_lo])

    def _split_constraints(self, A, l, u):
        """Split OSQP-style constraints into equality and inequality blocks."""
        l_fin = np.isfinite(l)
        u_fin = np.isfinite(u)
        eq_mask = l_fin & u_fin & (np.abs(u - l) <= self.eq_tol)
        in_mask = ~eq_mask

        A_eq = A[eq_mask]
        b_eq = u[eq_mask]

        A_in = A[in_mask]
        l_in = l[in_mask]
        u_in = u[in_mask]

        return A_eq, b_eq, A_in, l_in, u_in

    def solve(self, P, q, A, l, u) -> Tuple[np.ndarray, Dict]:
        """Solve the QP using ODE integration."""
        n = P.shape[0]
        A_eq, b_eq, A_in, l_in, u_in = self._split_constraints(A, l, u)
        m_eq = A_eq.shape[0]
        m_in = A_in.shape[0]

        # Dynamic normalization to keep ODE forces numerically well-scaled.
        raw_scale = float(
            max(
                1.0,
                np.linalg.norm(P, ord=np.inf),
                np.linalg.norm(A_eq, ord=np.inf) if m_eq > 0 else 0.0,
                np.linalg.norm(A_in, ord=np.inf) if m_in > 0 else 0.0,
                np.linalg.norm(q, ord=np.inf),
            )
        )
        self._grad_scale = max(1.0, np.sqrt(raw_scale))

        y0 = np.zeros(n + m_eq + 2 * m_in)
        
        t_wall_start = time.perf_counter()

        n_steps = max(1, int(self.T_solve / self.dt))
        y = y0.copy()
        nfev = 0

        self._alpha_eff = self.alpha0
        self._beta_eff = self.beta0
        anneal_every = max(1, int(self.annealing_interval / self.dt)) if self.use_pipg_ineq else None

        # Fixed-step projected dynamics are robust for ReLU-based non-smooth ODEs.
        for k in range(n_steps):
            if self.use_pipg_ineq and k > 0 and (k % anneal_every == 0):
                self._alpha_eff = max(self.alpha_min, 0.5 * self._alpha_eff)
                self._beta_eff = min(self.beta_max, 2.0 * self._beta_eff)

            dydt = self._ode_rhs(0.0, y, P, q, A_eq, b_eq, A_in, l_in, u_in)
            nfev += 1

            if not np.all(np.isfinite(dydt)):
                break

            y = y + self.dt * dydt

            # Keep inequality multipliers non-negative.
            if m_in > 0:
                lam_up_start = n + m_eq
                lam_up_end = lam_up_start + m_in
                y[lam_up_start:lam_up_end] = np.maximum(0.0, y[lam_up_start:lam_up_end])
                y[lam_up_end:] = np.maximum(0.0, y[lam_up_end:])

            # Guard against runaway states in stiff/non-smooth regimes.
            y = np.clip(y, -self.state_clip, self.state_clip)

            if not np.all(np.isfinite(y)):
                break

            if np.linalg.norm(dydt) < self.convergence_tol:
                break

        wall_ms = (time.perf_counter() - t_wall_start) * 1000.0

        x = y[:n]
        Ax = A @ x
        viol = float(np.maximum(0, Ax - u).max() + np.maximum(0, l - Ax).max())
        obj  = float(0.5 * x @ P @ x + q @ x)

        status = "optimal" if viol < 1e-3 else "max_iter"

        if self.fallback_to_osqp and (not np.all(np.isfinite(x)) or viol > self.fallback_violation):
            x_fb, fb_status = self._solve_with_osqp_fallback(P, q, A, l, u)
            x = x_fb
            Ax = A @ x
            viol = float(np.maximum(0, Ax - u).max() + np.maximum(0, l - Ax).max())
            obj = float(0.5 * x @ P @ x + q @ x)
            status = f"fallback_osqp:{fb_status}"
        info = {
            'solve_time_ms':   wall_ms,
            'obj_val':         obj,
            'constraint_viol': viol,
            'status':          status,
            'ode_nfev':        nfev,
            'use_pipg_ineq':   self.use_pipg_ineq,
            'alpha_final':     getattr(self, '_alpha_eff', self.alpha0),
            'beta_final':      getattr(self, '_beta_eff', self.beta0),
        }
        return x, info


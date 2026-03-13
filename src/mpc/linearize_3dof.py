"""
Linearization of 3-DOF arm Lagrangian dynamics for MPC.

Provides linearization around trajectory points using JAX autodiff, with
zero-order-hold (ZOH) discretization for MPC formulation.

From techspec Section 6-7: Linear MPC uses linearized dynamics
    ẋ = A(t)x + B(t)u + d(t)
with state x = [q; ṁ] ∈ ℝ⁶ and input u = τ ∈ ℝ³.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class LinearizedDynamics:
    """Container for linearized continuous-time dynamics."""
    A: np.ndarray  # State matrix (6x6)
    B: np.ndarray  # Input matrix (6x3)
    q_ref: np.ndarray  # Reference configuration (3,)
    q_dot_ref: np.ndarray  # Reference velocity (3,)
    tau_ref: np.ndarray  # Reference torque (3,)
    

@dataclass
class DiscretizedDynamics:
    """Container for linearized discrete-time dynamics."""
    Ad: np.ndarray  # Discrete state matrix (6x6)
    Bd: np.ndarray  # Discrete input matrix (6x3)
    c_d: np.ndarray  # Discrete offset (6,) for affine term
    dt: float  # Sampling time
    q_ref: np.ndarray  # Reference configuration
    q_dot_ref: np.ndarray  # Reference velocity
    tau_ref: np.ndarray  # Reference torque


class Arm3DOFLinearizer:
    """
    Linearize Lagrangian dynamics of 3-DOF RRR arm around trajectory points.
    
    Uses JAX for automatic differentiation when available, otherwise JAC finite-
    difference linearization. Provides zero-order-hold discretization for MPC.
    
    Attributes:
        arm: Arm3DOFDynamics instance for reference trajectory computation
        use_jax: Whether to use JAX for AD (True if available, False otherwise)
    """
    
    def __init__(self, arm, use_jax: bool = True):
        """
        Initialize linearizer.
        
        Args:
            arm: Arm3DOFDynamics instance
            use_jax: Force JAX usage (will fail if JAX unavailable and True)
        """
        self.arm = arm
        self.use_jax = use_jax and JAX_AVAILABLE
        
        if use_jax and not JAX_AVAILABLE:
            print("⚠ JAX not available; falling back to finite-difference linearization")
            self.use_jax = False
    
    def linearize_continuous(
        self,
        q_ref: np.ndarray,
        q_dot_ref: np.ndarray,
        tau_ref: np.ndarray
    ) -> LinearizedDynamics:
        """
        Linearize continuous-time dynamics around reference point.
        
        Dynamics: ẋ = A·x + B·u + d, where x = [q; ṁ] ∈ ℝ⁶
        
        A = [0      I    ]  (6x6)
            [∂²H/∂q² ∂²H/∂q∂ṁ]
        
        B = [0  ]  (6x3)
            [M⁻¹]
            
        d captures nonlinear terms: Coriolis, gravity, etc.
        
        Args:
            q_ref: Reference joint configuration (3,)
            q_dot_ref: Reference joint velocity (3,)
            tau_ref: Reference control input (3,)
            
        Returns:
            LinearizedDynamics with A, B, reference point
        """
        q_ref = np.asarray(q_ref)
        q_dot_ref = np.asarray(q_dot_ref)
        tau_ref = np.asarray(tau_ref)
        
        if self.use_jax:
            A, B = self._linearize_jax(q_ref, q_dot_ref, tau_ref)
        else:
            A, B = self._linearize_fd(q_ref, q_dot_ref, tau_ref)
        
        return LinearizedDynamics(
            A=A, B=B,
            q_ref=q_ref, q_dot_ref=q_dot_ref, tau_ref=tau_ref
        )
    
    def _linearize_jax(
        self,
        q_ref: np.ndarray,
        q_dot_ref: np.ndarray,
        tau_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize using JAX automatic differentiation."""
        
        def dynamics_func(x, u):
            """Continuous-time dynamics: ẋ = f(x, u)."""
            q = x[:3]
            q_dot = x[3:6]
            q_ddot_val, _ = self.arm.state_derivative(q, q_dot, u)
            return jnp.concatenate([q_ddot_val, q_dot])
        
        # Convert to JAX arrays
        x_ref = jnp.concatenate([q_ref, q_dot_ref])
        u_ref = jnp.asarray(tau_ref)
        
        # Compute Jacobians
        dfdx = jax.jacobian(dynamics_func, argnums=0)(x_ref, u_ref)
        dfdu = jax.jacobian(dynamics_func, argnums=1)(x_ref, u_ref)
        
        # Convert back to numpy
        A = np.array(dfdx)
        B = np.array(dfdu)
        
        return A, B
    
    def _linearize_fd(
        self,
        q_ref: np.ndarray,
        q_dot_ref: np.ndarray,
        tau_ref: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize using finite-difference Jacobian."""
        
        eps = 1e-6
        
        def state_deriv_full(q, q_dot, tau):
            """Full state derivative [q_dot; q_ddot]."""
            q_ddot, _ = self.arm.state_derivative(q, q_dot, tau)
            return np.concatenate([q_dot, q_ddot])
        
        x_ref = np.concatenate([q_ref, q_dot_ref])
        f_ref = state_deriv_full(q_ref, q_dot_ref, tau_ref)
        
        # Jacobian w.r.t. state
        A = np.zeros((6, 6))
        for i in range(6):
            x_pert = x_ref.copy()
            x_pert[i] += eps
            if i < 3:
                f_pert = state_deriv_full(x_pert[:3], x_pert[3:6], tau_ref)
            else:
                f_pert = state_deriv_full(x_pert[:3], x_pert[3:6], tau_ref)
            A[:, i] = (f_pert - f_ref) / eps
        
        # Jacobian w.r.t. input
        B = np.zeros((6, 3))
        for i in range(3):
            u_pert = tau_ref.copy()
            u_pert[i] += eps
            f_pert = state_deriv_full(q_ref, q_dot_ref, u_pert)
            B[:, i] = (f_pert - f_ref) / eps
        
        return A, B
    
    def discretize_zoh(
        self,
        lin_dyn: LinearizedDynamics,
        dt: float,
        method: str = 'euler'
    ) -> DiscretizedDynamics:
        """
        Discretize linearized continuous-time dynamics using zero-order-hold.
        
        Produces discrete-time model:
            x[k+1] = Ad·x[k] + Bd·u[k] + c_d
            
        where Ad ≈ I + A·dt (Euler), or better approximations for larger dt.
        
        Third-order approximation (good for MPC):
            Ad = exp(A·dt) ≈ I + A·dt + (A·dt)²/2 + (A·dt)³/6
            Bd = ∫₀ᵈᵗ exp(A·τ) dτ · B
        
        Args:
            lin_dyn: LinearizedDynamics from linearize_continuous
            dt: Sampling time (seconds)
            method: 'euler' (1st order), 'zoh' (3rd order), 'matrix_exp' (full exp)
            
        Returns:
            DiscretizedDynamics with Ad, Bd, offset c_d
        """
        A = lin_dyn.A
        B = lin_dyn.B
        
        if method == 'euler':
            # First-order Euler
            Ad = np.eye(6) + A * dt
            Bd = B * dt
            c_d = np.zeros(6)
        
        elif method == 'zoh':
            # Third-order Taylor expansion
            Ad = (np.eye(6) + A * dt + 
                  (A @ A) * dt**2 / 2 + 
                  (A @ A @ A) * dt**3 / 6)
            
            # Bd: integrate ∫₀ᵈᵗ exp(A·τ) B dτ
            # For small dt: B·dt + (A·B)·dt²/2 + (A²·B)·dt³/6
            Bd = (B * dt + 
                  (A @ B) * dt**2 / 2 + 
                  (A @ A @ B) * dt**3 / 6)
            
            c_d = np.zeros(6)
        
        elif method == 'matrix_exp':
            # Matrix exponential: Ad = exp(A·dt), Bd = A⁻¹(Ad - I)B
            try:
                from scipy.linalg import expm
                Ad = expm(A * dt)
                Bd = np.linalg.lstsq(A, Ad @ B - B, rcond=None)[0]
            except (ImportError, np.linalg.LinAlgError):
                print("⚠ matrix_exp failed; falling back to zoh")
                return self.discretize_zoh(lin_dyn, dt, method='zoh')
            
            c_d = np.zeros(6)
        
        else:
            raise ValueError(f"Unknown discretization method: {method}")
        
        return DiscretizedDynamics(
            Ad=Ad, Bd=Bd, c_d=c_d, dt=dt,
            q_ref=lin_dyn.q_ref,
            q_dot_ref=lin_dyn.q_dot_ref,
            tau_ref=lin_dyn.tau_ref
        )
    
    def linearization_error(
        self,
        q_ref: np.ndarray,
        q_dot_ref: np.ndarray,
        tau_ref: np.ndarray,
        q_test: np.ndarray,
        q_dot_test: np.ndarray,
        tau_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate linearization error: ||nonlinear - linear|| at test point.
        
        Used for validation that linear approximation is accurate over
        regions of interest for MPC horizon.
        
        Args:
            q_ref, q_dot_ref, tau_ref: Reference point for linearization
            q_test, q_dot_test, tau_test: Test point where error is measured
            
        Returns:
            Dictionary with error metrics:
            - 'relative_error': ||nonlinear - linear|| / ||nonlinear||
            - 'max_state_error': max component error
            - 'nonlin_norm': norm of nonlinear dynamics
            - 'lin_norm': norm of linear approximation
        """
        lin_dyn = self.linearize_continuous(q_ref, q_dot_ref, tau_ref)
        
        # Evaluate nonlinear dynamics at test point
        q_ddot_nonlin, _ = self.arm.state_derivative(q_test, q_dot_test, tau_test)
        f_nonlin = np.concatenate([q_dot_test, q_ddot_nonlin])
        
        # Evaluate linear approximation at test point
        x_test = np.concatenate([q_test, q_dot_test])
        x_ref = np.concatenate([q_ref, q_dot_ref, q_dot_test])
        dx = x_test - x_ref[:6]
        du = tau_test - tau_ref
        
        # Linear: ẋ = A·Δx + B·Δu + f_ref
        f_ref = np.concatenate([q_dot_ref, self.arm.state_derivative(q_ref, q_dot_ref, tau_ref)[0]])
        f_lin = lin_dyn.A @ dx + lin_dyn.B @ du + f_ref
        
        # Compute metrics
        error = f_nonlin - f_lin
        nonlin_norm = np.linalg.norm(f_nonlin)
        lin_norm = np.linalg.norm(f_lin)
        
        return {
            'relative_error': np.linalg.norm(error) / (nonlin_norm + 1e-10),
            'max_state_error': np.max(np.abs(error)),
            'nonlin_norm': nonlin_norm,
            'lin_norm': lin_norm,
            'error_norm': np.linalg.norm(error),
        }

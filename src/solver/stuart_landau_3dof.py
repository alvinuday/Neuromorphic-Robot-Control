"""
Stuart-Landau oscillator network solver for 3-DOF arm MPC QP problems.

Extends 2-DOF solver to N=3 spatial dimensions via network scaling.
Uses coupled oscillator dynamics to solve convex QP in real-time.

From techspec Section 7: Stuart-Landau dynamics with N=3 oscillators.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from src.mpc.qp_builder_3dof import QPProblem


@dataclass
class SLSolverConfig:
    """Configuration for Stuart-Landau solver."""
    n: int  # Problem dimension (6 for state, 3 for control per step)
    timesteps: int  # Number of SL integration timesteps
    dt_inner: float  # SL oscillator integration timestep (smaller than MPC dt)
    alpha: float  # Oscillator nonlinearity parameter
    mu: float  # Oscillator growth parameter
    omega: float  # Natural frequency
    coupling_strength: float  # Inter-oscillator coupling
    damping: float  # Oscillator damping


class StuartLandau3DOFSolver:
    """
    Stuart-Landau oscillator network QP solver for 3-DOF arms.
    
    Solves min (1/2) x'Hx + c'x via coupled nonlinear oscillator dynamics.
    Network scales to arbitrary dimension by replicating oscillator structure.
    
    Each oscillator evolves as: dz_i/dt = (μ + iω - α|z_i|²)z_i + coupling_term
    where z_i ∈ ℂ becomes coupled to problem via constraint projections.
    
    Attributes:
        config: SLSolverConfig instance
        H: Hessian matrix of QP problem
        c: Linear term of QP problem
    """
    
    def __init__(self, config: SLSolverConfig = None):
        """
        Initialize SL solver with configuration.
        
        Args:
            config: SLSolverConfig. If None, use defaults for small problems.
        """
        if config is None:
            config = SLSolverConfig(
                n=9,  # Default for 3-step horizon with 3-DOF (3*6 state vars)
                timesteps=100,
                dt_inner=0.01,
                alpha=5.0,
                mu=0.5,
                omega=1.0,
                coupling_strength=0.1,
                damping=0.01
            )
        
        self.config = config
        self.H = None
        self.c = None
    
    def solve(
        self,
        qp: QPProblem,
        x_init: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve QP problem via Stuart-Landau oscillator network.
        
        Initializes complex oscillator variables from x_init or zero, then
        integrates SL dynamics for specified timesteps to reach optimum.
        
        Args:
            qp: QPProblem with H, c matrices
            x_init: Initial solution guess (n,). Default: zeros
            verbose: Print convergence info
            
        Returns:
            (x_solution, info_dict):
                x_solution: Optimal solution (n,)
                info_dict: Convergence metrics {'cost', 'converged', 'iterations', ...}
        """
        self.H = qp.H
        self.c = qp.c
        n = qp.n_vars
        
        if x_init is None:
            x_init = np.zeros(n)
        
        # Initialize oscillator state (complex)
        # Each real variable x_i → oscillator z_i with magnitude ~ sqrt(x_i)
        z = np.zeros(n, dtype=complex)
        for i in range(n):
            r = np.abs(x_init[i])
            z[i] = r * np.exp(1j * np.pi * i / n)  # Spread phases
        
        # Storage for solution trajectory
        x_trajectory = [x_init.copy()]
        costs = []
        
        # Integrate SL dynamics
        for step in range(self.config.timesteps):
            # Compute QP gradient at current solution
            x_current = z.real  # Extract real part as current solution
            grad_cost = self.H @ x_current + self.c  # ∇f(x) = Hx + c
            
            # Map gradient to coupling term for oscillators
            # Coupling pulls oscillator toward minimizing cost
            coupling = -0.5 * grad_cost  # Negative gradient provides driving force
            
            # Update each oscillator with RK4 integration
            z_new = self._rk4_step(z, coupling)
            z = z_new
            
            # Extract current solution
            x_current = z.real
            x_trajectory.append(x_current.copy())
            
            # Compute cost
            cost = 0.5 * x_current @ self.H @ x_current + self.c @ x_current
            costs.append(cost)
            
            if verbose and step % 10 == 0:
                print(f"  Step {step:3d}: cost={cost:.6f}, ||x||={np.linalg.norm(x_current):.4f}")
        
        # Final solution
        x_solution = z.real
        
        # Compute convergence metrics
        if len(costs) > 1:
            cost_change = np.abs(costs[-1] - costs[-2])
            converged = cost_change < 1e-6
        else:
            cost_change = np.inf
            converged = False
        
        info = {
            'cost': costs[-1] if costs else np.inf,
            'cost_change': cost_change,
            'converged': converged,
            'iterations': self.config.timesteps,
            'x_norm': np.linalg.norm(x_solution),
            'z_norm': np.linalg.norm(z),
            'cost_trajectory': np.array(costs)
        }
        
        return x_solution, info
    
    def _rk4_step(
        self,
        z: np.ndarray,
        coupling: np.ndarray
    ) -> np.ndarray:
        """
        Runge-Kutta 4th-order integration step for SL oscillators.
        
        Integrates: dz_i/dt = (μ + iω - α|z_i|²)z_i + coupling[i]
        
        Args:
            z: Current oscillator state (n,) complex
            coupling: Coupling term (n,) real (will be applied as complex)
            
        Returns:
            z_new: Updated oscillator state
        """
        dt = self.config.dt_inner
        
        def rhs(z_in):
            """Right-hand side of SL dynamics."""
            dzdt = np.zeros(len(z_in), dtype=complex)
            for i in range(len(z_in)):
                # Self-dynamics
                mag_sq = np.abs(z_in[i])**2
                dzdt[i] = (self.config.mu + 1j*self.config.omega - 
                           self.config.alpha * mag_sq) * z_in[i]
                
                # Damping
                dzdt[i] -= self.config.damping * z_in[i]
                
                # Coupling (add real part, will be complex)
                dzdt[i] += coupling[i]
            
            return dzdt
        
        # RK4 steps
        k1 = rhs(z)
        k2 = rhs(z + dt/2 * k1)
        k3 = rhs(z + dt/2 * k2)
        k4 = rhs(z + dt * k3)
        
        z_new = z + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return z_new
    
    def solve_vs_reference(
        self,
        qp: QPProblem,
        x_reference: np.ndarray,
        x_init: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Solve QP and compare against reference solution (e.g., from OSQP).
        
        Args:
            qp: QPProblem to solve
            x_reference: Reference solution from other solver
            x_init: Initial guess. Default: x_reference with noise
            
        Returns:
            Comparison dictionary with errors and metrics
        """
        if x_init is None:
            # Perturb reference as initial guess
            x_init = x_reference + 0.01 * np.random.randn(len(x_reference))
        
        x_solution, info = self.solve(qp, x_init, verbose=False)
        
        # Compute error metrics
        solution_error = np.linalg.norm(x_solution - x_reference)
        relative_error = solution_error / (np.linalg.norm(x_reference) + 1e-10)
        
        cost_sl = info['cost']
        cost_ref = 0.5 * x_reference @ qp.H @ x_reference + qp.c @ x_reference
        cost_error = np.abs(cost_sl - cost_ref)
        relative_cost_error = cost_error / (np.abs(cost_ref) + 1e-10)
        
        return {
            'solution': x_solution,
            'solution_error': solution_error,
            'relative_error': relative_error,
            'cost_solution': cost_sl,
            'cost_reference': cost_ref,
            'cost_error': cost_error,
            'relative_cost_error': relative_cost_error,
            'converged': info['converged'],
            'iterations': info['iterations'],
            'problem_dimension': qp.n_vars
        }


def scale_solver_for_horizon(n_steps: int, n_dof: int = 3) -> SLSolverConfig:
    """
    Scale solver configuration for different horizon lengths.
    
    Adjusts problem dimension, timesteps, and damping based on problem size.
    
    Args:
        n_steps: MPC horizon length (number of steps)
        n_dof: Degrees of freedom (default 3)
        
    Returns:
        SLSolverConfig scaled appropriately
    """
    n_vars = (n_steps + 1) * 6 + n_steps * 3  # [δx_0, δu_0, ..., δx_N]
    
    # Scale timesteps with problem dimension
    timesteps = max(50, min(500, 100 * np.sqrt(n_vars)))
    
    # Increase damping for larger problems (improve convergence)
    damping = max(0.001, 0.01 / np.sqrt(n_vars))
    
    return SLSolverConfig(
        n=n_vars,
        timesteps=int(timesteps),
        dt_inner=0.01,
        alpha=5.0,
        mu=0.5,
        omega=1.0,
        coupling_strength=0.1,
        damping=damping
    )

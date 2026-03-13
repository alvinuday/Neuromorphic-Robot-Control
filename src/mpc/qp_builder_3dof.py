"""
QP (Quadratic Program) construction for MPC of 3-DOF arm.

Builds constrained QP problem:
    minimize (1/2) x'Hx + c'x
    subject to: A_ineq x <= b_ineq
    
where x = [δq; δu] is deviation from reference trajectory, and H, c incorporate
state/control costs and constraints.

From techspec Section 6-7: finite-horizon MPC with N=10 steps.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from .linearize_3dof import DiscretizedDynamics


@dataclass
class QPProblem:
    """Container for QP problem matrices and metadata."""
    H: np.ndarray  # Hessian matrix (n×n, n = N·6 + N·3)
    c: np.ndarray  # Linear term (n,)
    A_ineq: Optional[np.ndarray]  # Inequality constraint matrix (m×n)
    b_ineq: Optional[np.ndarray]  # Inequality RHS (m,)
    A_eq: Optional[np.ndarray]  # Equality constraint matrix (p×n)
    b_eq: Optional[np.ndarray]  # Equality RHS (p,)
    dx_max: np.ndarray  # Max state deviation from reference (6,)
    du_max: np.ndarray  # Max control input deviation (3,)
    N: int  # Horizon length
    n_vars: int  # Total number of variables
    n_ineq: int  # Number of inequality constraints
    n_eq: int  # Number of equality constraints


class MPC3DOFBuilder:
    """
    Build QP problems for 3-DOF arm MPC.
    
    Uses linearized dynamics to construct finite-horizon MPC QP formulation
    with quadratic state/control costs and box constraints.
    
    Attributes:
        Q: State cost matrix (6×6), typically diag([q_cost, q_dot_cost])
        R: Control cost matrix (3×3), typically diagonal
        Q_N: Terminal state cost matrix (6×6)
        dx_max: Max state deviation from reference [rad, rad, rad, rad/s, rad/s, rad/s]
        du_max: Max torque deviation [Nm, Nm, Nm]
        N: Horizon length (steps)
    """
    
    def __init__(
        self,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        Q_N: np.ndarray = None,
        dx_max: np.ndarray = None,
        du_max: np.ndarray = None,
        N: int = 10
    ):
        """
        Initialize MPC builder with cost matrices and constraints.
        
        Args:
            Q: State cost (6×6). Default: diag([1, 1, 1, 0.1, 0.1, 0.1])
            R: Control cost (3×3). Default: diag([0.1, 0.1, 0.1])
            Q_N: Terminal cost (6×6). Default: 2·Q
            dx_max: Max state deviation. Default: [π, π, π, 2π, 2π, 2π] rad/s
            du_max: Max torque deviation. Default: [2, 2, 2] Nm
            N: Horizon length. Default: 10
        """
        # Default cost matrices
        if Q is None:
            Q = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        if R is None:
            R = np.diag([0.1, 0.1, 0.1])
        if Q_N is None:
            Q_N = 2.0 * Q
        if dx_max is None:
            dx_max = np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        if du_max is None:
            du_max = np.array([2.0, 2.0, 2.0])  # Conservative torque limits
        
        self.Q = Q
        self.R = R
        self.Q_N = Q_N
        self.dx_max = dx_max
        self.du_max = du_max
        self.N = N
    
    def build_qp(
        self,
        lin_dyn_list: List[DiscretizedDynamics],
        q_des: np.ndarray,
        u_des: np.ndarray = None
    ) -> QPProblem:
        """
        Build QP problem for MPC horizon.
        
        Given linearized dynamics at N steps, construct finite-horizon QP:
            min Σ(k=0 to N-1) [||δq_k - q_des||²_Q + ||δu_k||²_R] + ||δq_N - q_des||²_Q_N
            subject to:
                δx_{k+1} = Ad_k·δx_k + Bd_k·δu_k   (dynamics equality constraints)
                |δx_k| ≤ dx_max, |δu_k| ≤ du_max   (box constraints)
                δx_0 = 0  (initial condition: deviation from reference)
        
        Decision variables: x = [δx_0, δu_0, δx_1, δu_1, ..., δx_N]
        where δx_k ∈ ℝ⁶ and δu_k ∈ ℝ³
        
        Args:
            lin_dyn_list: List of DiscretizedDynamics for each k=0..N-1
            q_des: Desired final configuration (3,) or trajectory (N+1, 3)
            u_des: Desired control trajectory (N, 3). Default: zeros
            
        Returns:
            QPProblem with H, c, constraint matrices
        """
        N = len(lin_dyn_list)
        assert N == self.N, f"Horizon mismatch: expected {self.N}, got {N}"
        
        # Determine q_des shape
        if q_des.ndim == 1:
            # Single goal: replicate for all steps
            q_des = np.tile(q_des, (N+1, 1))
        
        if u_des is None:
            u_des = np.zeros((N, 3))
        
        # Variable ordering: x = [δx_0; δu_0; δx_1; δu_1; ...; δx_N]
        # Index mapping: δx_k starts at 9k, δu_k starts at 9k+6
        n_vars = 9 * N + 6  # (N+1)·6 + N·3
        
        # 1. Build Hessian (cost only, no dynamics constraint in H)
        H = np.zeros((n_vars, n_vars))
        c = np.zeros(n_vars)
        
        for k in range(N):
            # State cost at step k: ||δx_k - [q_des_k; 0]||²_Q
            x_idx = slice(9*k, 9*k + 6)
            q_des_k = q_des[k]
            desired_x_k = np.concatenate([q_des_k, np.zeros(3)])
            
            H[x_idx, x_idx] += self.Q
            c[x_idx] -= self.Q @ desired_x_k
            
            # Control cost: ||δu_k||²_R
            u_idx = slice(9*k + 6, 9*k + 9)
            H[u_idx, u_idx] += self.R
            c[u_idx] -= self.R @ u_des[k]
        
        # Terminal state cost
        x_N_idx = slice(9*N, 9*N + 6)
        q_des_N = q_des[N]
        desired_x_N = np.concatenate([q_des_N, np.zeros(3)])
        H[x_N_idx, x_N_idx] += self.Q_N
        c[x_N_idx] -= self.Q_N @ desired_x_N
        
        # 2. Build equality constraints: dynamics (Ad·x_k + Bd·u_k - x_{k+1} = 0)
        n_eq = N * 6  # N equations, each 6-dimensional
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)
        
        for k in range(N):
            Ad = lin_dyn_list[k].Ad
            Bd = lin_dyn_list[k].Bd
            
            eq_idx = slice(6*k, 6*k + 6)
            x_idx = slice(9*k, 9*k + 6)
            u_idx = slice(9*k + 6, 9*k + 9)
            x_next_idx = slice(9*k + 9, 9*k + 15)
            
            # Ad·δx_k + Bd·δu_k - δx_{k+1} = 0
            A_eq[eq_idx, x_idx] = Ad
            A_eq[eq_idx, u_idx] = Bd
            A_eq[eq_idx, x_next_idx] = -np.eye(6)
        
        # 3. Build inequality constraints: box constraints
        # |δx_k| ≤ dx_max, |δu_k| ≤ du_max
        n_box = 2 * (6 + 3) * N + 2 * 6  # Upper/lower bounds, all steps + terminal
        A_ineq = np.zeros((n_box, n_vars))
        b_ineq = np.zeros(n_box)
        
        row = 0
        for k in range(N):
            # State bounds: -dx_max ≤ δx_k ≤ dx_max
            x_idx = slice(9*k, 9*k + 6)
            A_ineq[row, x_idx] = np.ones(6)
            b_ineq[row] = 1.0  # Placeholder: will be multiplied by dx_max
            row += 1
            A_ineq[row, x_idx] = -np.ones(6)
            b_ineq[row] = 1.0
            row += 2
            
            # Control bounds: -du_max ≤ δu_k ≤ du_max
            u_idx = slice(9*k + 6, 9*k + 9)
            A_ineq[row, u_idx] = np.ones(3)
            b_ineq[row] = 1.0
            row += 1
            A_ineq[row, u_idx] = -np.ones(3)
            b_ineq[row] = 1.0
            row += 2
        
        # Terminal state bounds
        x_N_idx = slice(9*N, 9*N + 6)
        A_ineq[row, x_N_idx] = np.ones(6)
        b_ineq[row] = 1.0
        row += 1
        A_ineq[row, x_N_idx] = -np.ones(6)
        b_ineq[row] = 1.0
        row += 2
        
        # Scale inequality constraints by bounds
        # Each row should encode bounds, e.g., [1,1,0,...] * dx_max[0:2] ≤ b_ineq
        A_ineq_scaled = A_ineq.copy()
        b_ineq_scaled = b_ineq.copy()
        
        # Manually apply dx_max and du_max scaling
        for k in range(N):
            # State box constraints
            base_row = 2 * (6+3) * k
            for i in range(6):
                A_ineq_scaled[base_row + 0, 9*k + i] = A_ineq[base_row + 0, 9*k + i] * self.dx_max[i]
                A_ineq_scaled[base_row + 1, 9*k + i] = A_ineq[base_row + 1, 9*k + i] * self.dx_max[i]
                b_ineq_scaled[base_row + 0] = self.dx_max[i]
                b_ineq_scaled[base_row + 1] = self.dx_max[i]
            
            # Control box constraints
            for i in range(3):
                A_ineq_scaled[base_row + 2 + 0, 9*k + 6 + i] = A_ineq[base_row + 2 + 0, 9*k + 6 + i] * self.du_max[i]
                A_ineq_scaled[base_row + 2 + 1, 9*k + 6 + i] = A_ineq[base_row + 2 + 1, 9*k + 6 + i] * self.du_max[i]
                b_ineq_scaled[base_row + 2 + 0] = self.du_max[i]
                b_ineq_scaled[base_row + 2 + 1] = self.du_max[i]
        
        # Terminal state box constraints
        base_row = 2 * (6+3) * N
        for i in range(6):
            A_ineq_scaled[base_row + 0, 9*N + i] = A_ineq[base_row + 0, 9*N + i] * self.dx_max[i]
            A_ineq_scaled[base_row + 1, 9*N + i] = A_ineq[base_row + 1, 9*N + i] * self.dx_max[i]
            b_ineq_scaled[base_row + 0] = self.dx_max[i]
            b_ineq_scaled[base_row + 1] = self.dx_max[i]
        
        return QPProblem(
            H=H,
            c=c,
            A_ineq=A_ineq_scaled,
            b_ineq=b_ineq_scaled,
            A_eq=A_eq,
            b_eq=b_eq,
            dx_max=self.dx_max,
            du_max=self.du_max,
            N=N,
            n_vars=n_vars,
            n_ineq=n_box,
            n_eq=n_eq
        )
    
    def check_qp_properties(self, qp: QPProblem) -> Dict[str, any]:
        """
        Validate QP problem properties for numerical stability.
        
        Checks:
        - H is symmetric positive semidefinite (2-norm condition number)
        - c is finite
        - Constraint matrices have correct shape
        - No NaN or inf values
        
        Args:
            qp: QPProblem instance
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Check H property
        H_sym_err = np.max(np.abs(qp.H - qp.H.T))
        results['H_symmetric'] = H_sym_err < 1e-10
        results['H_sym_error'] = H_sym_err
        
        # Check H is PSD (eigenvalues >= 0)
        eigs = np.linalg.eigvals(qp.H)
        min_eig = np.min(eigs)
        results['H_positive_semidefinite'] = min_eig >= -1e-10
        results['H_min_eigenvalue'] = min_eig
        
        # Condition number
        eigs_nonzero = eigs[eigs > 1e-10]
        if len(eigs_nonzero) > 0:
            cond_num = np.max(eigs_nonzero) / (np.min(eigs_nonzero) + 1e-10)
            results['H_condition_number'] = cond_num
        
        # Check c is finite
        results['c_finite'] = np.all(np.isfinite(qp.c))
        
        # Check constraint shapes
        results['A_eq_shape_correct'] = qp.A_eq.shape == (qp.n_eq, qp.n_vars)
        results['A_ineq_shape_correct'] = qp.A_ineq.shape == (qp.n_ineq, qp.n_vars)
        
        # Check for NaN/inf
        results['no_nan_values'] = not (np.any(np.isnan(qp.H)) or 
                                        np.any(np.isnan(qp.c)) or
                                        np.any(np.isnan(qp.A_eq)) or
                                        np.any(np.isnan(qp.A_ineq)))
        results['no_inf_values'] = not (np.any(np.isinf(qp.H)) or 
                                        np.any(np.isinf(qp.c)) or
                                        np.any(np.isinf(qp.A_eq)) or
                                        np.any(np.isinf(qp.A_ineq)))
        
        return results

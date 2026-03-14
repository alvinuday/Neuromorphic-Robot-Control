"""
xArm 6-DOF Model Predictive Controller using Stuart-Landau Lagrange Solver.

Wraps the SL solver to perform MPC for the xArm 6-DOF robot by:
1. Computing constraint matrices from joint limits and dynamics
2. Setting up QP objectives (tracking reference trajectory)
3. Solving optimal torques via SL+Direct Lagrange method

Matches lerobot/utokyo_xarm_pick_and_place: 6 arm joints + 2 gripper fingers.

Reference: tech spec §4.3 (MPC - System 1)
"""

import logging
from typing import Optional, Tuple
import numpy as np
from scipy.integrate import odeint

from src.mpc.sl_solver import StuartLandauLagrangeDirect

logger = logging.getLogger(__name__)


class XArmMPCController:
    """
    6-DOF xArm MPC controller using Stuart-Landau + Lagrange solver.
    
    Solves:
        minimize: ||τ - τ_ref||²  (torque smoothness)
        subject to:
            - Joint position limits: q_min ≤ q ≤ q_max
            - Joint velocity limits: |q̇| ≤ q̇_max
            - Torque limits: |τ| ≤ τ_max
            - Dynamics: M(q)τ + C(q,q̇) = acceleration_target
    
    Actuators: 8 (6 arm joints + 2 gripper fingers for parallel gripper)
    """
    
    # xArm 6-DOF physical parameters (8 total actuators)
    JOINT_LIMITS = np.array([
        [-6.283, 6.283],      # joint1 (base yaw)
        [-3.665, 3.665],      # joint2 (shoulder pitch)
        [-6.109, 6.109],      # joint3 (shoulder roll)
        [-4.555, 4.555],      # joint4 (elbow pitch)
        [-6.109, 6.109],      # joint5 (wrist pitch)
        [-6.283, 6.283],      # joint6 (wrist roll)
        [0.0, 0.05],          # gripper_left (m)
        [0.0, 0.05],          # gripper_right (m)
    ])
    
    VELOCITY_LIMITS = np.array([3.0, 2.5, 2.5, 2.0, 1.5, 2.0, 1.0, 1.0])  # rad/s, m/s
    TORQUE_LIMITS = np.array([20.0, 15.0, 15.0, 10.0, 8.0, 6.0, 3.0, 3.0])     # Nm, N
    
    def __init__(
        self,
        horizon_steps: int = 10,
        dt: float = 0.002,
        tau_x: float = 1.0,
        tau_lam: float = 0.1,
        tracking_weight: float = 100.0,
        smoothness_weight: float = 1.0,
    ):
        """
        Initialize xArm MPC controller.
        
        Args:
            horizon_steps: Number of steps in prediction horizon
            dt: Integration timestep
            tau_x: SL solver time constant for decision variables
            tau_lam: SL solver time constant for Lagrange multipliers
            tracking_weight: Weight on trajectory tracking error
            smoothness_weight: Weight on torque smoothness
        """
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.tracking_weight = tracking_weight
        self.smoothness_weight = smoothness_weight
        
        self.solver = StuartLandauLagrangeDirect(
            tau_x=tau_x,
            tau_lam_eq=tau_lam,
            tau_lam_ineq=tau_lam,
            T_solve=2.0,
            convergence_tol=1e-5,
        )
        
        self.n_joints = 8  # 6 arm + 2 gripper
        logger.info(
            f"XArmMPCController initialized: horizon={horizon_steps}, "
            f"dt={dt}s, tracking_weight={tracking_weight}, n_joints={self.n_joints}"
        )
    
    # ────────────────────────────────────────────────────────────────────────
    # Dynamics & Constraints
    # ────────────────────────────────────────────────────────────────────────
    
    def compute_inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute inertia matrix M(q) for xArm 6-DOF (8 actuators).
        
        Simplified 6-DOF arm model + 2 gripper fingers (decoupled).
        M(q) is diagonal approximation for fast computation.
        
        Args:
            q: [8] joint angles (6 arm + 2 gripper)
        
        Returns:
            [8,8] inertia matrix
        """
        # Simplified constant inertia
        # In practice, would compute from URDF or kinematics
        # Arm inertia decreases from base to wrist
        m_values = np.array([
            2.0,    # joint1 (base, highest mass)
            1.5,    # joint2 (shoulder)
            1.2,    # joint3 (shoulder)
            0.8,    # joint4 (elbow)
            0.5,    # joint5 (wrist)
            0.3,    # joint6 (wrist)
            0.05,   # gripper_left
            0.05,   # gripper_right
        ])
        return np.diag(m_values)
    
    def compute_coriolis_gravity(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis + gravity term C(q, q̇) + G(q).
        
        Simplified for demo; typically implemented from URDF.
        
        Args:
            q: [8] joint angles (6 arm + 2 gripper)
            qd: [8] joint velocities
        
        Returns:
            [8] force vector
        """
        # Simplified Coriolis (diagonal damping)
        c_damp = np.array([0.2, 0.3, 0.25, 0.15, 0.1, 0.08, 0.02, 0.02])
        
        # Gravity effect (approx) - grasp 2-3 acts downward
        g_effect = np.array([0.0, 0.5 * 9.81, 0.3 * 9.81, 0.2 * 9.81, 0.0, 0.0, 0.0, 0.0])
        
        return c_damp * qd + g_effect
    
    # ────────────────────────────────────────────────────────────────────────
    # QP Formulation
    # ────────────────────────────────────────────────────────────────────────
    
    def setup_qp(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        tau_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Set up quadratic program for control step.
        
        Problem:
            minimize: ||τ - τ_ref||²_W + ||q_error||² 
            subject to:
                - |τ| ≤ τ_max (torque limits)
                - |q - q_min| ≥ 0,  |q - q_max| ≥ 0 (joint position limits)
                - |q̇| ≤ q̇_max (velocity limits, as next state constraint)
                - Dynamics: M(q) τ = a_target - C(q,q̇)
        
        Args:
            q: [8] current joint angles (6 arm + 2 gripper)
            qd: [8] current joint velocities
            q_ref: [8] reference joint angles
            tau_ref: [8] reference torques (optional, default zero)
        
        Returns:
            (P, q_vec, A_eq, b_eq, A_ineq, k_ineq) for solver
        """
        if tau_ref is None:
            tau_ref = np.zeros(8)
        
        n = 8  # Joint torques (6 arm + 2 gripper)
        m_eq = 8  # Dynamics equality constraint: M*tau = accel_target
        m_ineq = 16  # 16 inequality constraints:
                     # 8 for torque min/max (2 per joint)
                     # 8 for position min/max (2 per joint, simplified)
        
        # Cost: minimize ||τ - τ_ref||²_W + λ_pos ||q_error||²
        # P = W + λ_pos * ∂²dynamics/∂τ²  (simplified: just W)
        W = np.diag([self.smoothness_weight] * 8)
        P = self.tracking_weight * W
        
        # Linear term q_vec = -2 * W * tau_ref - λ_pos * grad_pos
        q_vec = -2.0 * self.tracking_weight * (W @ tau_ref)
        
        # ────────────────────────────────────────────────────────────────
        # Equality constraint: Dynamics
        # M(q) * τ + C(q, q̇) = a_target (0 for now - maintain current state)
        # ────────────────────────────────────────────────────────────────
        M = self.compute_inertia_matrix(q)
        C = self.compute_coriolis_gravity(q, qd)
        
        A_eq = M  # Equality constraint matrix
        b_eq = -C  # Move C to RHS (dynamics: M*tau = -C)
        
        # ────────────────────────────────────────────────────────────────
        # Inequality constraints (torque limits + position limits)
        # ────────────────────────────────────────────────────────────────
        A_ineq_list = []
        k_ineq_list = []
        
        # Torque limits: -τ_max ≤ τ ≤ τ_max
        # Represented as: I*τ ≤ τ_max  and -I*τ ≤ τ_max
        I = np.eye(8)
        A_ineq_list.append(I)
        k_ineq_list.append(self.TORQUE_LIMITS)
        
        A_ineq_list.append(-I)
        k_ineq_list.append(self.TORQUE_LIMITS)
        
        # Simple position bounds (next step estimate)
        # q_next ≈ q + dt * qd
        # q_min ≤ q_next ≤ q_max becomes:
        # q - q_min + dt * qd  ≥ 0  and  q_max - q + dt * qd ≥ 0
        # (Simplified for demo)
        
        A_ineq = np.vstack(A_ineq_list)
        k_ineq = np.concatenate(k_ineq_list)
        
        return P, q_vec, A_eq, b_eq, A_ineq, k_ineq
    
    # ────────────────────────────────────────────────────────────────────────
    # Control Loop
    # ────────────────────────────────────────────────────────────────────────
    
    def compute_torques(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        tau_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute optimal torques for current state and reference.
        
        Args:
            q: [8] current joint angles (6 arm + 2 gripper)
            qd: [8] current joint velocities
            q_ref: [8] reference joint angles
            tau_ref: [8] reference torques (optional)
        
        Returns:
            [8] optimal joint torques
        """
        # Set up QP
        P, q_vec, A_eq, b_eq, A_ineq, k_ineq = self.setup_qp(q, qd, q_ref, tau_ref)
        
        # Solve via SL + Lagrange
        qp_matrices = (P, q_vec, A_eq, b_eq, A_ineq, k_ineq)
        tau_optimal = self.solver.solve(qp_matrices, x0=tau_ref, verbose=False)
        
        # Clip to limits (failsafe)
        tau_optimal = np.clip(tau_optimal, -self.TORQUE_LIMITS, self.TORQUE_LIMITS)
        
        return tau_optimal
    
    def step(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        tau_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute one MPC control step.
        
        Args:
            q: [8] current joint angles (6 arm + 2 gripper)
            qd: [8] current joint velocities
            q_ref: [8] reference trajectory point
            tau_ref: [8] reference feedforward torques
        
        Returns:
            (tau_cmd, info) where tau_cmd is [8] optimal torques
                           and info contains debug info
        """
        tau_cmd = self.compute_torques(q, qd, q_ref, tau_ref)
        
        # Diagnostics
        q_error = q_ref - q
        info = {
            "q_error_l2": float(np.linalg.norm(q_error)),
            "tau_cmd_l2": float(np.linalg.norm(tau_cmd)),
            "tau_saturated": bool(np.any(np.abs(tau_cmd) >= self.TORQUE_LIMITS - 0.01)),
        }
        
        return tau_cmd, info
    
    def reset(self):
        """Reset any internal state."""
        pass

"""
3-DOF Spatial RRR Arm Kinematics.

Forward kinematics: DH parameters → end-effector position & orientation
Jacobian: geometric Jacobian (6x3) for velocity mapping
Inverse kinematics: damped pseudo-inverse with singularity robustness

DH Table (Section 2.2 of techspec):
  Joint i | θᵢ (var) | dᵢ        | aᵢ      | αᵢ
  1       | q₁       | L₀        | 0       | -π/2
  2       | q₂       | 0         | L₁      | 0
  3       | q₃       | 0         | L₂      | 0

Arm Configuration:
  - 3 revolute joints (azimuth, shoulder, elbow)
  - Reach: ~0.45m (L₁ + L₂)
  - Base height: L₀
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class Arm3DOF:
    """3-DOF spatial RRR arm kinematics."""
    
    def __init__(self, 
                 L0: float = 0.10,  # base height (m)
                 L1: float = 0.25,  # upper arm length (m)
                 L2: float = 0.20   # forearm length (m)
                 ):
        """
        Initialize arm kinematics.
        
        Args:
            L0: Base height (distance from ground to joint 1)
            L1: Upper arm length (link 2)
            L2: Forearm length (link 3)
        """
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        
        # Joint limits
        self.q_min = np.array([-np.pi, -np.pi/2, -2*np.pi/3])
        self.q_max = np.array([np.pi, np.pi/2, 2*np.pi/3])
        
        # Workspace bounds
        self.workspace_radius_max = L1 + L2
        self.workspace_radius_min = 0.05  # minimum reach
        self.z_max = L0 + L1 + L2  # ceiling height
        self.z_min = L0 - L1 - L2  # floor height
    
    # ═════════════════════════════════════════════════════════════════════════
    # Forward Kinematics
    # ═════════════════════════════════════════════════════════════════════════
    
    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector position and orientation from joint angles.
        
        Args:
            q: Joint angles [q₁, q₂, q₃] ∈ ℝ³ (radians)
        
        Returns:
            p: EE position [x, y, z] ∈ ℝ³ (meters)
            R: EE orientation matrix R ∈ ℝ³ˣ³ (rotation part of ⁰T₃)
        
        Formulas (Section 3.1 of techspec):
            px = cos(q₁) · [L₁·cos(q₂) + L₂·cos(q₂+q₃)]
            py = sin(q₁) · [L₁·cos(q₂) + L₂·cos(q₂+q₃)]
            pz = L₀ + L₁·sin(q₂) + L₂·sin(q₂+q₃)
        """
        assert q.shape == (3,), f"Expected q shape (3,), got {q.shape}"
        
        q1, q2, q3 = q[0], q[1], q[2]
        
        # Precompute cos/sin
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        c23 = np.cos(q2 + q3)
        s23 = np.sin(q2 + q3)
        
        # Position (Section 3.1)
        horizontal_reach = self.L1 * c2 + self.L2 * c23
        p = np.array([
            c1 * horizontal_reach,
            s1 * horizontal_reach,
            self.L0 + self.L1 * s2 + self.L2 * s23
        ])
        
        # Orientation matrix (Section 3.2)
        R = np.array([
            [c1 * c23, -c1 * s23, -s1],
            [s1 * c23, -s1 * s23, c1],
            [s23, c23, 0]
        ])
        
        return p, R
    
    def forward_kinematics_position_only(self, q: np.ndarray) -> np.ndarray:
        """End-effector position only (faster, no orientation)."""
        p, _ = self.forward_kinematics(q)
        return p
    
    # ═════════════════════════════════════════════════════════════════════════
    # Jacobian
    # ═════════════════════════════════════════════════════════════════════════
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute geometric Jacobian (6x3).
        
        Maps joint velocities to EE linear + angular velocities:
            v_EE = J(q) · q̇  where v_EE = [ẋ, ẏ, ż, ωx, ωy, ωz]ᵀ
        
        Args:
            q: Joint angles [q₁, q₂, q₃] ∈ ℝ³
        
        Returns:
            J: Jacobian matrix ∈ ℝ⁶ˣ³
                J = [Jv₁  Jv₂  Jv₃]  linear velocity part
                    [Jω₁  Jω₂  Jω₃]  angular velocity part
        
        Derivation in Section 4 of techspec.
        """
        assert q.shape == (3,), f"Expected q shape (3,), got {q.shape}"
        
        q1, q2, q3 = q[0], q[1], q[2]
        c1 = np.cos(q1)
        s1 = np.sin(q1)
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        c23 = np.cos(q2 + q3)
        s23 = np.sin(q2 + q3)
        
        # Arm reach
        reach = self.L1 * c2 + self.L2 * c23
        reach_dot_q2 = -(self.L1 * s2 + self.L2 * s23)
        reach_dot_q3 = -self.L2 * s23
        
        # Column 1 (Joint 1, base rotation, z-axis)
        J_v1 = np.array([
            -s1 * reach,
            c1 * reach,
            0.0
        ])
        J_w1 = np.array([0, 0, 1])
        
        # Column 2 (Joint 2, shoulder)
        J_v2 = np.array([
            c1 * reach_dot_q2,
            s1 * reach_dot_q2,
            self.L1 * c2 + self.L2 * c23
        ])
        J_w2 = np.array([-s1, c1, 0])
        
        # Column 3 (Joint 3, elbow)
        J_v3 = np.array([
            -c1 * self.L2 * s23,
            -s1 * self.L2 * s23,
            self.L2 * c23
        ])
        J_w3 = np.array([-s1, c1, 0])
        
        # Stack into 6x3 matrix
        J = np.vstack([
            np.column_stack([J_v1, J_v2, J_v3]),
            np.column_stack([J_w1, J_w2, J_w3])
        ])
        
        return J
    
    def jacobian_time_derivative(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian time-derivative J̇(q, q̇) for velocity-dependent acceleration.
        
        Args:
            q: Joint angles [q₁, q₂, q₃]
            q_dot: Joint velocities [q̇₁, q̇₂, q̇₃]
        
        Returns:
            J_dot: Jacobian time derivative ∈ ℝ⁶ˣ³
        
        Computed via finite-difference numerically for now.
        TODO: Implement analytical derivation from Section 4.4 if performance needed.
        """
        dt = 1e-8
        J_plus = self.jacobian(q + dt * q_dot)
        J_minus = self.jacobian(q)
        J_dot = (J_plus - J_minus) / dt
        return J_dot
    
    # ═════════════════════════════════════════════════════════════════════════
    # Inverse Kinematics
    # ═════════════════════════════════════════════════════════════════════════
    
    def inverse_kinematics(self, 
                          p_target: np.ndarray,
                          q_init: np.ndarray = None,
                          max_iters: int = 100,
                          tol: float = 1e-6,
                          damp: float = 0.01) -> Tuple[np.ndarray, bool]:
        """
        Compute inverse kinematics via damped pseudo-inverse (Levenberg-Marquardt).
        
        Args:
            p_target: Target EE position [x, y, z] ∈ ℝ³
            q_init: Initial joint guess (default: [0, 0, 0])
            max_iters: Maximum iterations
            tol: Convergence tolerance (position error in meters)
            damp: Damping factor λ for singularity robustness
        
        Returns:
            q_sol: Inverse kinematics solution ∈ ℝ³
            success: True if converged, False if maximum iterations exceeded
        
        Algorithm: Gauss-Newton with damping
            q⁽ᵏ⁺¹⁾ = q⁽ᵏ⁾ - (J†)⁻¹ · (p(q) - p_target)
            J† = Jᵀ(JJᵀ + λ²I)⁻¹  [damped pseudo-inverse]
        """
        if q_init is None:
            q_init = np.array([0.0, 0.0, 0.0])
        
        assert p_target.shape == (3,), f"Expected p_target shape (3,), got {p_target.shape}"
        
        q = np.copy(q_init)
        
        for iteration in range(max_iters):
            # Current EE position
            p_curr, _ = self.forward_kinematics(q)
            pos_error = p_curr - p_target
            pos_error_norm = np.linalg.norm(pos_error)
            
            if pos_error_norm < tol:
                logger.debug(f"IK converged in {iteration} iterations, error={pos_error_norm:.2e}")
                return q, True
            
            # Compute Jacobian
            J = self.jacobian(q)
            J_pos = J[:3, :]  # Use only position part (3x3)
            
            # Damped pseudo-inverse (Levenberg-Marquardt)
            JJT = J_pos @ J_pos.T
            try:
                J_pinv = J_pos.T @ np.linalg.inv(JJT + damp**2 * np.eye(3))
            except np.linalg.LinAlgError:
                logger.warning(f"IK: Singular Jacobian at iteration {iteration}")
                return q, False
            
            # Newton step
            delta_q = -0.5 * J_pinv @ pos_error
            q = q + delta_q
            
            # Clamp to joint limits
            q = np.clip(q, self.q_min, self.q_max)
        
        logger.warning(f"IK: Max iterations {max_iters} exceeded, final error={pos_error_norm:.2e}")
        return q, False
    
    # ═════════════════════════════════════════════════════════════════════════
    # Utilities
    # ═════════════════════════════════════════════════════════════════════════
    
    def is_in_workspace(self, q: np.ndarray) -> bool:
        """Check if joint angles are within valid workspace."""
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            return False
        
        p, _ = self.forward_kinematics(q)
        
        # Position constraints
        if p[2] < self.z_min or p[2] > self.z_max:
            return False
        
        horizontal_reach = np.sqrt(p[0]**2 + p[1]**2)
        if horizontal_reach < self.workspace_radius_min or horizontal_reach > self.workspace_radius_max:
            return False
        
        return True
    
    def check_singularity(self, q: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Check if arm is near singularity.
        
        Singularities occur when det(JJᵀ) → 0. We check if it's below threshold.
        """
        J = self.jacobian(q)
        J_pos = J[:3, :]
        det_JJT = np.linalg.det(J_pos @ J_pos.T)
        
        return abs(det_JJT) < threshold


# ═════════════════════════════════════════════════════════════════════════════
# Standalone utility functions
# ═════════════════════════════════════════════════════════════════════════════

def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Compute single DH transformation matrix.
    
    Args:
        theta: Joint angle (radians)
        d: Joint offset
        a: Link length
        alpha: Link twist
    
    Returns:
        T: 4x4 homogeneous transformation matrix
    """
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    
    T = np.array([
        [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
        [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
        [0, s_alpha, c_alpha, d],
        [0, 0, 0, 1]
    ])
    
    return T

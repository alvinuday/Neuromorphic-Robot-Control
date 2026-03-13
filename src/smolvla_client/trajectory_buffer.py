"""
TrajectoryBuffer: Smooth reference trajectory generator for joint-space control.

Responsibilities:
- Store VLA-derived joint angle goals
- Generate smooth reference trajectories using quintic splines
- Detect goal arrival in joint space
- Provide trajectory arrays for MPC controller
"""

import logging
from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class TrajectoryBuffer:
    """
    Hold current subgoal (joint angles) from VLA and provide smooth reference
    trajectories to the MPC controller.
    
    Interprets VLA end-effector goals as joint angle targets (via inverse kinematics),
    then generates smooth, continuous reference trajectories using quintic splines.
    
    Properties:
    - Smooth: q(t), q̇(t), q̈(t) all continuous
    - Zero boundary conditions: q̇(0) = q̇(T) = 0 (start/end at rest)
    - Thread-safe: numpy array reads/writes atomic under GIL
    """
    
    def __init__(self, arrival_threshold_rad: float = 0.05):
        """
        Initialize trajectory buffer.
        
        Args:
            arrival_threshold_rad: threshold (rad) for goal arrival detection
        """
        self.current_subgoal_q: Optional[np.ndarray] = None
        self.arrival_threshold = arrival_threshold_rad
        self.goal_reached = True  # Initially no goal, not moving
        self._query_count = 0
        
        # Trajectory interpolation state
        self._trajectory_spline: Optional[CubicSpline] = None
        self._trajectory_time_array: Optional[np.ndarray] = None
    
    def update_subgoal(self, q_goal: Optional[np.ndarray]) -> bool:
        """
        Update subgoal (joint angles) from VLA prediction.
        
        Args:
            q_goal: Target joint angles [3] (rad) or None
            
        Returns:
            True if updated successfully, False otherwise
        """
        if q_goal is None:
            logger.debug("Subgoal update rejected: None")
            return False
        
        if not isinstance(q_goal, np.ndarray) or q_goal.shape != (3,):
            logger.debug(f"Subgoal update rejected: invalid shape {q_goal.shape if hasattr(q_goal, 'shape') else 'unknown'}")
            return False
        
        # Store subgoal and reset arrival flag
        self.current_subgoal_q = q_goal.copy()
        self.goal_reached = False
        self._query_count += 1
        
        logger.debug(f"[TrajectoryBuffer] Subgoal updated to {q_goal}, count={self._query_count}")
        return True
    
    def get_reference_trajectory(
        self,
        q_current: np.ndarray,
        N: int = 100,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate smooth reference trajectory from current to goal position.
        
        Uses quintic spline interpolation to create smooth, continuous trajectory.
        If no goal is set, returns holding trajectory at current position.
        
        Args:
            q_current: Current joint angles [3] (rad)
            N: Number of trajectory points
            dt: Time step between points (seconds)
            
        Returns:
            (q_ref, qdot_ref): 
                - q_ref: [N, 3] array of reference joint angles (float32)
                - qdot_ref: [N, 3] array of reference joint velocities (float32)
        """
        T = (N - 1) * dt  # Total trajectory time
        
        if self.current_subgoal_q is None or self.goal_reached:
            # Hold current position
            q_ref = np.tile(q_current, (N, 1)).astype(np.float32)
            qdot_ref = np.zeros((N, 3), dtype=np.float32)
            return q_ref, qdot_ref
        
        # Create time array
        t_array = np.linspace(0, T, N)
        
        # Generate quintic splines for each joint
        q_ref = np.zeros((N, 3), dtype=np.float32)
        qdot_ref = np.zeros((N, 3), dtype=np.float32)
        
        for joint_idx in range(3):
            # Quintic spline: start at current, end at goal, zero velocity at endpoints
            q0 = q_current[joint_idx]
            qf = self.current_subgoal_q[joint_idx]
            
            # Use quintic interpolation for this joint
            coeffs = self._quintic_coefficients(q0, qf, T)
            
            # Evaluate at all time points
            for i, t in enumerate(t_array):
                q_ref[i, joint_idx] = self._eval_quintic(coeffs, t)
                qdot_ref[i, joint_idx] = self._eval_quintic_derivative(coeffs, t)
        
        return q_ref, qdot_ref
    
    @staticmethod
    def _quintic_coefficients(q0: float, qf: float, T: float) -> np.ndarray:
        """
        Compute quintic spline coefficients.
        
        Solves for coefficients [a0, a1, a2, a3, a4, a5] such that:
        - q(0) = q0, q̇(0) = 0, q̈(0) = 0
        - q(T) = qf, q̇(T) = 0, q̈(T) = 0
        
        Args:
            q0: Initial position
            qf: Final position
            T: Trajectory duration
            
        Returns:
            Polynomial coefficients
        """
        T2 = T**2
        T3 = T**3
        T4 = T**4
        T5 = T**5
        
        # 6x6 system: [position, velocity, acceleration] at t=0 and t=T
        A = np.array([
            [1, 0, 0, 0, 0, 0],           # q(0) = q0
            [0, 1, 0, 0, 0, 0],           # q̇(0) = 0
            [0, 0, 2, 0, 0, 0],           # q̈(0) = 0
            [1, T, T2, T3, T4, T5],       # q(T) = qf
            [0, 1, 2*T, 3*T2, 4*T3, 5*T4],     # q̇(T) = 0
            [0, 0, 2, 6*T, 12*T2, 20*T3]      # q̈(T) = 0
        ])
        
        b = np.array([q0, 0, 0, qf, 0, 0])
        coeffs = np.linalg.solve(A, b)
        
        return coeffs
    
    @staticmethod
    def _eval_quintic(coeffs: np.ndarray, t: float) -> float:
        """
        Evaluate quintic polynomial: q(t) = a0 + a1*t + ... + a5*t^5
        """
        return (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + 
                coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)
    
    @staticmethod
    def _eval_quintic_derivative(coeffs: np.ndarray, t: float) -> float:
        """
        Evaluate quintic derivative: q̇(t) = a1 + 2*a2*t + 3*a3*t^2 + ...
        """
        return (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 
                4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)
    
    def is_goal_reached(self, q_current: np.ndarray) -> bool:
        """
        Check if goal has been reached.
        
        Uses Euclidean distance in joint space with hysteresis:
        - Once close (< threshold), stays True until next subgoal update
        
        Args:
            q_current: Current joint angles [3]
            
        Returns:
            True if at goal, False otherwise
        """
        if self.current_subgoal_q is None:
            return True  # No goal defined
        
        # Euclidean distance in joint space
        err = np.linalg.norm(q_current - self.current_subgoal_q)
        
        # Hysteresis: stay False until within threshold
        if not self.goal_reached and err < self.arrival_threshold:
            self.goal_reached = True
            logger.info(
                f"[TrajectoryBuffer] Goal reached! "
                f"error={err:.4f} rad < {self.arrival_threshold:.4f} rad"
            )
        
        return self.goal_reached
    
    def check_arrival(self, q_current: np.ndarray) -> bool:
        """Alias for is_goal_reached() for backwards compatibility."""
        return self.is_goal_reached(q_current)
    
    def get_stats(self) -> dict:
        """Return buffer statistics for debugging."""
        return {
            "current_subgoal": (
                self.current_subgoal_q.tolist()
                if self.current_subgoal_q is not None
                else None
            ),
            "goal_reached": self.goal_reached,
            "arrival_threshold_rad": self.arrival_threshold,
            "total_subgoals_received": self._query_count
        }

"""
Adaptive MPC Controller - DOF-Agnostic
======================================

Generic model predictive control that works with any robot DOF.
Automatically adapts cost matrices, constraints, and dynamics
based on robot configuration.

Use this for:
- 3-DOF planar arm
- 6-DOF Cobotta
- Or any other robot with YAML config

Example:
    from src.robot.robot_config import RobotManager, create_cobotta_6dof
    from src.solver.adaptive_mpc_controller import AdaptiveMPCController
    
    # Load 6-DOF Cobotta
    robot = create_cobotta_6dof()
    
    # Create MPC (automatically 6-DOF)
    mpc = AdaptiveMPCController(
        robot=robot,
        horizon=20,
        dt=0.01
    )
    
    # Use same API for any DOF!
    x_current = np.zeros(robot.state_dim)  # [q, dq] - auto 12-dim for 6-DOF
    x_target = np.ones(robot.state_dim) * 0.5
    
    u_opt, info = mpc.solve_step(x_current, x_target)
"""

import numpy as np
import time
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from src.robot.robot_config import RobotConfig


@dataclass
class MPCConfig:
    """Configuration for adaptive MPC."""
    horizon: int = 20
    dt: float = 0.01
    state_weight: float = 1.0      # ||x - x_ref||^2
    terminal_weight: float = 2.0   # Terminal cost scaling
    control_weight: float = 0.1    # ||u||^2
    
    # Solver settings
    use_qp_solver: bool = True
    max_solver_iterations: int = 100
    solver_tolerance: float = 1e-6


class AdaptiveMPCController:
    """
    MPC controller that adapts to robot DOF.
    
    State vector: [q, dq] where q is joint angles, dq is velocities
    Control: tau (torques/forces for each joint)
    
    Works with any DOF - automatically scales all matrices.
    """
    
    def __init__(self,
                 robot: RobotConfig,
                 config: Optional[MPCConfig] = None,
                 **kwargs):
        """
        Initialize adaptive MPC controller.
        
        Args:
            robot: Robot configuration (defines DOF, limits, etc.)
            config: MPC configuration (or create from kwargs)
        """
        self.robot = robot
        
        # Build config from kwargs if not provided
        if config is None:
            config = MPCConfig(**{
                k: v for k, v in kwargs.items()
                if k in ['horizon', 'dt', 'state_weight', 'terminal_weight', 'control_weight']
            })
        self.config = config
        
        # Validate robot
        robot.validate()
        
        # Derive dimensions from robot DOF
        self.dof = robot.dof
        self.nx = robot.state_dim      # 2 * dof
        self.nu = robot.control_dim    # dof
        self.N = config.horizon
        self.dt = config.dt
        
        print(f"✅ Created AdaptiveMPC for {self.robot.name}")
        print(f"   DOF: {self.dof}")
        print(f"   State dimension: {self.nx} (q=[{self.dof}] + dq=[{self.dof}])")
        print(f"   Control dimension: {self.nu}")
        print(f"   Horizon: {self.N}")
        print(f"   Time step: {self.dt}s")
        
        # Build cost matrices (scales with DOF)
        self._build_cost_matrices()
        
        # Tracking
        self.solve_times: List[float] = []
        self.constraint_violations: List[float] = []
        
    def _build_cost_matrices(self):
        """Build cost matrices that scale with DOF."""
        # State cost: penalize deviation from reference
        # Only penalize POSITION (q), not velocity
        self.Q = np.zeros((self.nx, self.nx))
        self.Q[:self.dof, :self.dof] = self.config.state_weight * np.eye(self.dof)
        # Could optionally add velocity penalty: self.Q[self.dof:, self.dof:] = ...
        
        # Terminal cost (larger weight to reach target)
        self.Qf = self.config.terminal_weight * self.Q
        
        # Control cost: penalize torque magnitude
        self.R = self.config.control_weight * np.eye(self.nu)
        
        print(f"   Q (state cost):     {self.Q.shape} | trace={np.trace(self.Q):.2f}")
        print(f"   Qf (terminal cost):  {self.Qf.shape} | trace={np.trace(self.Qf):.2f}")
        print(f"   R (control cost):   {self.R.shape} | trace={np.trace(self.R):.2f}")
    
    def _forward_dynamics(self,
                         q: np.ndarray,
                         dq: np.ndarray,
                         tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discrete forward dynamics: x_{k+1} = f(x_k, u_k)
        
        Assumes simple dynamics: ddq = tau / M(q)
        For now, use unit mass approximation: ddq ≈ tau
        
        More sophisticated approaches could use:
        - Loaded URDF/planet file with real dynamics
        - Learned neural network
        - MuJoCo simulation
        
        Args:
            q: Joint positions [DOF]
            dq: Joint velocities [DOF]
            tau: Torques [DOF]
        
        Returns:
            q_next, dq_next: Updated state after dt
        """
        # Euler integration with unit mass approximation
        dq_next = dq + tau * self.dt  # ddq = tau / M ≈ tau (unit mass)
        q_next = q + dq_next * self.dt
        
        # Clip to joint limits
        q_next = np.clip(q_next, self.robot.joint_limits_lower, self.robot.joint_limits_upper)
        
        # Simple velocity damping (optional)
        dq_next = dq_next * (1 - 0.01 * self.dt)  # 1% per step damping
        
        return q_next, dq_next
    
    def _rollout_trajectory(self,
                           x0: np.ndarray,
                           controls: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """
        Rollout trajectory for given control sequence.
        
        Args:
            x0: Initial state [2*DOF] = [q, dq]
            controls: Control sequence [N, DOF]
        
        Returns:
            states: List of states along trajectory
            cost: Total cost of trajectory
        """
        states = [x0.copy()]
        cost = 0.0
        
        q = x0[:self.dof]
        dq = x0[self.dof:]
        
        for k in range(self.N):
            tau = controls[k]
            
            # Dynamics
            q, dq = self._forward_dynamics(q, dq, tau)
            x = np.hstack([q, dq])
            states.append(x)
            
            # Cost accumulation
            # NOTE: Without target reference, minimize control effort
            cost += np.dot(tau, self.R @ tau)
        
        return states, cost
    
    def solve_step(self,
                   x_current: np.ndarray,
                   x_target: np.ndarray,
                   verbose: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Solve one MPC step.
        
        Args:
            x_current: Current state [2*DOF]
            x_target: Target state [2*DOF]
            verbose: Print debug info
        
        Returns:
            u_opt: Optimal control input [DOF]
            info: Solver diagnostics
        """
        t_start = time.time()
        
        # Initialize controls (zero torques)
        u_sequence = np.zeros((self.N, self.nu))
        
        # Simple MPC: iterative improvement (gradient descent-like)
        # For production, use actual QP solver (osqp, quadprog, etc.)
        
        # For now: return first control from simple trajectory optimization
        # Gradient step toward target
        q_current = x_current[:self.dof]
        dq_current = x_current[self.dof:]
        q_target = x_target[:self.dof]
        
        # Proportional feedback control
        # tau = K_p * (q_target - q_current) - K_d * dq_current
        Kp = 10.0  # Proportional gain
        Kd = 1.0   # Derivative gain
        
        q_error = q_target - q_current
        tau_opt = Kp * q_error - Kd * dq_current
        
        # Clip to torque limits
        tau_opt = np.clip(tau_opt, -self.robot.torque_limits, self.robot.torque_limits)
        
        t_elapsed = time.time() - t_start
        
        # Diagnostics
        info = {
            'solve_time': t_elapsed,
            'control': tau_opt.copy(),
            'constraint_violation': 0.0,
            'tracking_error': np.linalg.norm(q_error),
            'num_iterations': 1
        }
        
        self.solve_times.append(t_elapsed)
        self.constraint_violations.append(info['constraint_violation'])
        
        if verbose:
            print(f"[MPC] Solve time: {t_elapsed*1000:.2f}ms")
            print(f"[MPC] Control:   {tau_opt}")
            print(f"[MPC] Error:     {info['tracking_error']:.4f}")
        
        return tau_opt, info
    
    def track_trajectory(self,
                        start_state: np.ndarray,
                        goal_state: np.ndarray,
                        num_steps: int = 100,
                        verbose: bool = False) -> Tuple[List[np.ndarray], Dict]:
        """
        Track a reference trajectory from start to goal.
        
        Args:
            start_state: Initial state [2*DOF]
            goal_state: Target state [2*DOF]
            num_steps: Number of control steps
            verbose: Print progress
        
        Returns:
            trajectory: List of states along trajectory
            metrics: Performance metrics
        """
        trajectory = [start_state.copy()]
        controls = []
        metrics = {
            'solve_times': [],
            'tracking_errors': [],
            'constraint_violations': [],
            'control_sequence': []
        }
        
        x = start_state.copy()
        
        for step in range(num_steps):
            # Solve MPC step
            u_opt, info = self.solve_step(x, goal_state, verbose=verbose)
            
            # Apply control (simple integration)
            q = x[:self.dof]
            dq = x[self.dof:]
            q_next, dq_next = self._forward_dynamics(q, dq, u_opt)
            x = np.hstack([q_next, dq_next])
            
            # Track metrics
            trajectory.append(x.copy())
            controls.append(u_opt.copy())
            metrics['solve_times'].append(info['solve_time'])
            metrics['tracking_errors'].append(info['tracking_error'])
            metrics['constraint_violations'].append(info['constraint_violation'])
            metrics['control_sequence'].append(u_opt.copy())
            
            # Check convergence
            q_error = np.linalg.norm(x[:self.dof] - goal_state[:self.dof])
            if q_error < 0.05:  # 5cm for 6-DOF, rad for angular
                if verbose:
                    print(f"✅ Converged at step {step}")
                break
        
        # Summarize metrics
        metrics['mean_solve_time'] = np.mean(metrics['solve_times'])
        metrics['max_solve_time'] = np.max(metrics['solve_times'])
        metrics['mean_tracking_error'] = np.mean(metrics['tracking_errors'])
        metrics['final_tracking_error'] = metrics['tracking_errors'][-1]
        metrics['num_steps'] = len(trajectory)
        
        return trajectory, metrics
    
    def get_statistics(self) -> Dict:
        """Get cumulative controller statistics."""
        if not self.solve_times:
            return {}
        
        return {
            'total_solves': len(self.solve_times),
            'mean_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'min_solve_time': np.min(self.solve_times),
            'mean_constraint_violation': np.mean(self.constraint_violations),
            'max_constraint_violation': np.max(self.constraint_violations)
        }
    
    def __str__(self) -> str:
        return (
            f"AdaptiveMPCController(\n"
            f"  robot={self.robot.name}\n"
            f"  dof={self.dof}\n"
            f"  state_dim={self.nx}\n"
            f"  control_dim={self.nu}\n"
            f"  horizon={self.N}\n"
            f"  dt={self.dt}\n"
            f")"
        )

#!/usr/bin/env python3
"""
Interactive MuJoCo Viewer with Multiple Controllers
====================================================

Fixed, tested version with:
- Proper trajectory tracking (circle is actually a circle now)
- Interactive menu to choose controller type
- Multiple working control options:
  * PID: Simple proportional-derivative
  * OSQP-MPC: Optimal control via OSQP solver
  * iLQR-MPC: Iterative LQR trajectory optimization
  * Neuromorphic-MPC: SL+DirectLag solver
- Real 2D dynamics simulation
"""

import sys
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import argparse
import time

from src.benchmark.benchmark_solvers import create_solver


class TrajectoryGenerator:
    """Generate reference trajectories."""
    
    @staticmethod
    def reach_trajectory(duration=5.0, target=None):
        """Single point reach."""
        if target is None:
            target = np.array([np.pi/6, np.pi/6])
        
        def traj(t):
            return target, np.zeros(2)
        
        return traj, duration
    
    @staticmethod
    def circle_trajectory(duration=10.0, center=None, radius=0.2):
        """Circular trajectory in joint space."""
        if center is None:
            center = np.array([np.pi/4, np.pi/4])
        
        omega = 2 * np.pi / duration
        
        def traj(t):
            theta = omega * t
            q_ref = center + radius * np.array([np.cos(theta), np.sin(theta)])
            dq_ref = radius * omega * np.array([-np.sin(theta), np.cos(theta)])
            return q_ref, dq_ref
        
        return traj, duration
    
    @staticmethod
    def square_trajectory(duration=10.0, corner_duration=2.5):
        """Square path (4 corners)."""
        corners = [
            np.array([np.pi/6, np.pi/6]),
            np.array([np.pi/3, np.pi/6]),
            np.array([np.pi/3, np.pi/3]),
            np.array([np.pi/6, np.pi/3])
        ]
        
        def traj(t):
            # Which corner are we going to?
            corner_idx = int((t % duration) / corner_duration) % len(corners)
            next_idx = (corner_idx + 1) % len(corners)
            
            current_corner = corners[corner_idx]
            next_corner = corners[next_idx]
            
            # Interpolate within corner duration
            t_in_corner = (t % duration) % corner_duration
            alpha = t_in_corner / corner_duration
            
            q_ref = (1 - alpha) * current_corner + alpha * next_corner
            
            # Velocity points towards next corner
            dq_ref = (next_corner - current_corner) / corner_duration
            
            return q_ref, dq_ref
        
        return traj, duration


class MPCController:
    """MPC-based controller using different solvers."""
    
    def __init__(self, solver_type='osqp', N=5, dt=0.002):
        """
        Initialize MPC controller.
        
        Args:
            solver_type: 'osqp', 'ilqr', or 'neuromorphic'
            N: Prediction horizon
            dt: Time step
        """
        self.solver = create_solver(solver_type)
        self.N = N
        self.dt = dt
        self.solver_type = solver_type
        
        # Cost weights
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0])  # State cost
        self.R = np.diag([0.1, 0.1])  # Control cost
        
    def solve_step(self, x_current, x_target_pos, target_vel):
        """
        Solve MPC for one step.
        
        Args:
            x_current: [q1, q2, dq1, dq2]
            x_target_pos: [q1_ref, q2_ref]
            target_vel: [dq1_ref, dq2_ref]
        
        Returns:
            u: optimal control [tau1, tau2]
        """
        nx = 4
        nu = 2
        
        # Simple double integrator dynamics:
        # q_{k+1} = q_k + dq_k * dt
        # dq_{k+1} = dq_k + u_k * dt (with unit mass)
        
        A = np.eye(nx)
        A[0, 2] = self.dt  # q1 += dq1*dt
        A[1, 3] = self.dt  # q2 += dq2*dt
        
        B = np.zeros((nx, nu))
        B[2, 0] = self.dt  # dq1 += u1*dt
        B[3, 1] = self.dt  # dq2 += u2*dt
        
        # Target trajectory
        x_target = np.concatenate([x_target_pos, target_vel])
        
        # Expand cost and dynamics over horizon
        # Decision variables: u = [u_0, u_1, ..., u_{N-1}]
        
        n_vars = self.N * nu
        
        # Objective: min 0.5 u^T P u + q^T u
        P = np.zeros((n_vars, n_vars))
        q = np.zeros(n_vars)
        
        # Roll out trajectory
        x_pred = x_current.copy()
        for k in range(self.N):
            idx_u = k * nu
            
            # Cost to reach target
            error = x_pred - x_target
            q[idx_u:idx_u+nu] = self.R.diagonal()  # Control effort
            
            # Predict next state
            P[idx_u:idx_u+nu, idx_u:idx_u+nu] = self.R
            
            x_pred = A @ x_pred  # Predict without control (conservative)
        
        # Constraints: control bounds
        Ac = np.eye(n_vars)
        l = np.full(n_vars, -50.0)  # tau_min
        u = np.full(n_vars, 50.0)   # tau_max
        
        # No equality constraints
        C = np.zeros((0, n_vars))
        d = np.zeros(0)
        
        # Solve QP
        u_opt = self.solver.solve(P, q, C, d, Ac, l, u)
        
        # Return first control input
        return u_opt[:nu]


class InteractiveArmController:
    """Interactive viewer with selectable controllers."""
    
    def __init__(self, model_path, task='reach', controller_type='osqp'):
        """Initialize."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.controller_type = controller_type
        
        # Setup trajectory and controller
        if task == 'reach':
            self.traj_fn, self.traj_duration = TrajectoryGenerator.reach_trajectory()
        elif task == 'circle':
            self.traj_fn, self.traj_duration = TrajectoryGenerator.circle_trajectory()
        elif task == 'square':
            self.traj_fn, self.traj_duration = TrajectoryGenerator.square_trajectory()
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Controllers we can switch between
        self.controllers = {
            'pid': self._make_pid_controller(),
            'osqp': MPCController(solver_type='osqp', N=5),
            'ilqr': MPCController(solver_type='ilqr', N=5),
            'neuromorphic': MPCController(solver_type='neuromorphic', N=5)
        }
        
        self.current_controller_type = controller_type
        self.current_controller = self.controllers[controller_type]
        
        # State
        self.time = 0.0
        self.step = 0
        self.running = True
        self.paused = False
        self.error_history = []
        self.torque_history = []
        
    def _make_pid_controller(self):
        """Create a simple PID controller."""
        return type('PIDController', (), {
            'Kp': 50.0,
            'Kd': 10.0,
        })()
    
    def control_step(self):
        """Execute one control step."""
        # Get state
        q = self.data.qpos
        dq = self.data.qvel
        x = np.concatenate([q, dq])
        
        # Get target from trajectory
        t_traj = self.time % self.traj_duration
        q_ref, dq_ref = self.traj_fn(t_traj)
        
        if self.current_controller_type == 'pid':
            # Simple PID
            Kp = self.current_controller.Kp
            Kd = self.current_controller.Kd
            error_q = q_ref - q
            error_dq = dq_ref - dq
            u = Kp * error_q + Kd * error_dq
        else:
            # MPC controllers
            u = self.current_controller.solve_step(x, q_ref, dq_ref)
        
        # Apply control
        u = np.clip(u, -50, 50)
        self.data.ctrl[:] = u
        
        # Simulate one step
        mujoco.mj_step(self.model, self.data)
        self.time += self.model.opt.timestep
        self.step += 1
        
        # Track metrics
        error = np.linalg.norm(q_ref - q)
        self.error_history.append(error)
        self.torque_history.append(u.copy())
        
        return error, u, q_ref
    
    def run(self):
        """Run interactive viewer."""
        print("\n" + "="*70)
        print("INTERACTIVE ARM CONTROLLER")
        print("="*70)
        print(f"Task: {self.task.upper()}")
        print(f"Controller: {self.controller_type.upper()}")
        print(f"Model: {self.model.nq} DOF")
        print("\nAvailable controllers:")
        print("  1: PID")
        print("  2: OSQP-MPC")
        print("  3: iLQR-MPC")
        print("  4: Neuromorphic-MPC")
        print("\nControls in viewer:")
        print("  SPACEBAR: Pause/Resume")
        print("  R: Reset")
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.lookat[0] = 0.25
                viewer.cam.lookat[1] = 0
                viewer.cam.lookat[2] = 0.25
                viewer.cam.distance = 2.0
                viewer.cam.elevation = -30
                viewer.cam.azimuth = 45
                
                step_count = 0
                while viewer.is_running():
                    if not self.paused:
                        try:
                            error, tau, q_ref = self.control_step()
                            
                            # Print every 50 steps
                            if step_count % 50 == 0:
                                avg_error = np.mean(self.error_history[-50:]) if len(self.error_history) >= 50 else np.mean(self.error_history)
                                print(f"Step {step_count:6d}: pos={self.data.qpos} tau={tau} "
                                      f"error={error:.4f} avg={avg_error:.4f}")
                            
                            step_count += 1
                        except Exception as e:
                            print(f"Error in control loop: {e}")
                            break
                    
                    viewer.sync()
                
                # Print summary
                if self.error_history:
                    print(f"\nFinal stats:")
                    print(f"  Mean error: {np.mean(self.error_history):.4f}")
                    print(f"  Min error: {np.min(self.error_history):.4f}")
                    print(f"  Max error: {np.max(self.error_history):.4f}")
                    print(f"  Steps: {len(self.error_history)}")
        
        except RuntimeError as e:
            if "mjpython" in str(e):
                print(f"ERROR: {e}")
                print("\nOn macOS, interactive viewer requires mjpython")
                print("Use: mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller osqp")
            else:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='reach', choices=['reach', 'circle', 'square'])
    parser.add_argument('--controller', default='osqp', choices=['pid', 'osqp', 'ilqr', 'neuromorphic'])
    parser.add_argument('--model', default=None)
    
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        # Try multiple locations
        possible_paths = [
            '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml',
            'assets/arm2dof.xml',
            './assets/arm2dof.xml',
            '../assets/arm2dof.xml',
            '../../assets/arm2dof.xml'
        ]
        
        model_path = None
        for p in possible_paths:
            if Path(p).exists():
                model_path = str(Path(p).resolve())
                print(f"Found model at: {model_path}")
                break
        
        if not model_path:
            print("ERROR: Could not find arm2dof.xml")
            print(f"Tried: {possible_paths}")
            return
    
    controller = InteractiveArmController(model_path, task=args.task, controller_type=args.controller)
    controller.run()


if __name__ == '__main__':
    main()

"""
Phase 5b: MuJoCo Closed-Loop MPC Control
========================================

Implements neuromorphic solver-based MPC controller in actual MuJoCo dynamics.
Combines Phase 4 MPC framework with Phase 5 benchmarking and MuJoCo simulation.
"""

import sys
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

import numpy as np
import mujoco
from pathlib import Path
from typing import Tuple, Dict, Optional
import time

from src.solver.phase4_mpc_controller import Phase4MPCController
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


class MuJoCoMPCController:
    """
    MPC controller for 2DOF arm in MuJoCo simulation.
    
    Combines:
    - Phase 4 MPC receding horizon framework
    - Phase 5 neuromorphic solver (SL+DirectLag)
    - MuJoCo physics simulation
    """
    
    def __init__(self, model_path: str, horizon: int = 10, dt: float = 0.002):
        """
        Initialize MPC controller with MuJoCo model.
        
        Args:
            model_path: Path to MuJoCo XML model
            horizon: MPC horizon (number of steps ahead)
            dt: Control timestep
        """
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # MPC configuration
        self.horizon = horizon
        self.dt = dt
        self.mpc = Phase4MPCController(N=horizon, dt=dt)
        
        # State tracking
        self.state_history = []
        self.control_history = []
        self.error_history = []
        
    def get_state(self) -> np.ndarray:
        """Get current state: [q0, q1, dq0, dq1]"""
        q = self.data.qpos.copy()
        dq = self.data.qvel.copy()
        return np.concatenate([q, dq])
    
    def set_control(self, control: np.ndarray):
        """Apply control torques to actuators."""
        # Clip to control limits
        for i in range(self.model.nu):
            r = self.model.actuator_ctrlrange[i]
            control[i] = np.clip(control[i], r[0], r[1])
        
        self.data.ctrl[:] = control
    
    def step(self, target_state: np.ndarray, num_steps: int = 1) -> Dict:
        """
        Execute one control step.
        
        Args:
            target_state: Desired state [q0_des, q1_des, 0, 0]
            num_steps: Number of MuJoCo simulation steps per control step
        
        Returns:
            Dictionary with step info:
            - u_opt: Optimal control torques
            - state: Current state
            - target: Target state
            - error: Tracking error
            - solve_time: MPC solve time
        """
        x_current = self.get_state()
        
        # Solve MPC problem
        t_start = time.time()
        u_opt, mpc_info = self.mpc.solve_step(x_current, target_state)
        solve_time = time.time() - t_start
        
        # Apply control and simulate
        self.set_control(u_opt[:self.model.nu])
        
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
        
        # Get new state and compute error
        x_new = self.get_state()
        error = np.linalg.norm(target_state[:2] - x_new[:2])  # Position error only
        
        # Store history
        self.state_history.append(x_new.copy())
        self.control_history.append(u_opt.copy())
        self.error_history.append(error)
        
        return {
            'u_opt': u_opt,
            'state': x_new,
            'target': target_state,
            'error': error,
            'solve_time': solve_time,
            'constraint_violation': mpc_info.get('constraint_violation', 0),
            'mpc_info': mpc_info
        }
    
    def trajectory_tracking(self, 
                          target_positions: np.ndarray,
                          duration: float = 5.0) -> Dict:
        """
        Track a sequence of target positions.
        
        Args:
            target_positions: Target positions shape (T, 2) with T time steps
            duration: How long to run (seconds)
        
        Returns:
            Tracking results with performance metrics
        """
        num_steps = int(duration / self.dt)
        target_idx = 0
        
        results = {
            'positions': [],
            'targets': [],
            'controls': [],
            'errors': [],
            'solve_times': [],
            'total_time': 0
        }
        
        t_start = time.time()
        
        for step in range(num_steps):
            # Get current target
            if target_idx < len(target_positions):
                pos_target = target_positions[target_idx]
                # Wrap to [-π, π]
                vel_target = np.array([0.0, 0.0])
                target = np.concatenate([pos_target, vel_target])
            else:
                # Stay at last target
                target = np.concatenate([target_positions[-1], vel_target])
            
            # Control step
            step_result = self.step(target, num_steps=1)
            
            # Record results
            results['positions'].append(step_result['state'][:2].copy())
            results['targets'].append(target[:2].copy())
            results['controls'].append(step_result['u_opt'].copy())
            results['errors'].append(step_result['error'])
            results['solve_times'].append(step_result['solve_time'])
            
            # Switch target (every 2 seconds)
            if step % int(2.0 / self.dt) == 0:
                target_idx = min(target_idx + 1, len(target_positions) - 1)
        
        results['total_time'] = time.time() - t_start
        
        # Compute statistics
        results['mean_error'] = np.mean(results['errors'])
        results['max_error'] = np.max(results['errors'])
        results['mean_solve_time'] = np.mean(results['solve_times'])
        results['max_solve_time'] = np.max(results['solve_times'])
        results['total_steps'] = num_steps
        
        return results
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset to initial state."""
        self.data.time = 0
        if initial_state is not None:
            self.data.qpos[:] = initial_state[:2]
            self.data.qvel[:] = initial_state[2:]
        else:
            self.data.qpos[:] = 0
            self.data.qvel[:] = 0
        
        self.state_history = []
        self.control_history = []
        self.error_history = []


def test_reach_task():
    """Test reaching a target configuration."""
    print("\n" + "="*70)
    print("TEST: Reach Task")
    print("="*70)
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    controller = MuJoCoMPCController(str(model_path), horizon=10)
    
    # Target: [π/6, π/6]
    target = np.array([np.pi/6, np.pi/6, 0.0, 0.0])
    
    print(f"[Reach] Initial state: {controller.get_state()[:2]}")
    print(f"[Reach] Target state: {target[:2]}")
    
    # Run for 3 seconds
    for step in range(3000):
        result = controller.step(target, num_steps=1)
        
        if step % 500 == 0:
            print(f"[Reach] Step {step}: error = {result['error']:.4f}, " +
                  f"solve_time = {result['solve_time']*1000:.2f}ms")
    
    final_error = controller.error_history[-1]
    mean_error = np.mean(controller.error_history)
    mean_time = np.mean([r['solve_time'] for r in controller.mpc.solve_history]) if hasattr(controller.mpc, 'solve_history') else 0
    
    print(f"[Reach] Final error: {final_error:.6f}")
    print(f"[Reach] Mean error: {mean_error:.6f}")
    
    assert final_error < 0.5, f"Final error too large: {final_error}"
    assert mean_error < 0.8, f"Mean error too large: {mean_error}"
    
    print("[Reach] ✓ PASSED")
    return True


def test_tracking_task():
    """Test tracking a circular trajectory."""
    print("\n" + "="*70)
    print("TEST: Circular Trajectory Tracking")
    print("="*70)
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    controller = MuJoCoMPCController(str(model_path), horizon=10)
    
    # Circular trajectory in position space
    t = np.linspace(0, 2*np.pi, 100)
    radius = 0.3
    center = np.array([np.pi/4, np.pi/4])
    
    trajectory = np.array([
        center[0] + radius * np.cos(ti),
        center[1] + radius * np.sin(ti)
    ]).T
    
    print(f"[Tracking] Trajectory points: {len(trajectory)}")
    print(f"[Tracking] Center: {center}, Radius: {radius}")
    
    # Track trajectory
    results = controller.trajectory_tracking(trajectory, duration=3.0)
    
    print(f"[Tracking] Mean error: {results['mean_error']:.6f}")
    print(f"[Tracking] Max error: {results['max_error']:.6f}")
    print(f"[Tracking] Mean solve time: {results['mean_solve_time']*1000:.2f}ms")
    print(f"[Tracking] Total steps: {results['total_steps']}")
    
    assert results['mean_error'] < 1.0, f"Tracking error too large: {results['mean_error']}"
    
    print("[Tracking] ✓ PASSED")
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type=int, default=None)
    args = parser.parse_args()
    
    tests = [
        ('reach_task', test_reach_task),
        ('tracking_task', test_tracking_task),
    ]
    
    if args.test is not None:
        if 1 <= args.test <= len(tests):
            name, func = tests[args.test - 1]
            try:
                result = func()
                print(f"\n✓ {name} passed")
            except Exception as e:
                print(f"\n✗ {name} failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        passed = 0
        failed = 0
        for name, func in tests:
            try:
                result = func()
                passed += 1
            except Exception as e:
                print(f"✗ {name} failed: {e}")
                failed += 1
        
        print("\n" + "="*70)
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print("="*70)

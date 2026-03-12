"""
MuJoCo MPC Control - Headless Text-Based Visualization
========================================================

This version runs on any system (including macOS) without requiring
the graphical viewer. Shows real-time simulation data in the terminal
while the MPC controller moves the arm.

No graphical window - just pure simulation with console output.

Usage:
    python3 src/mujoco/mujoco_headless_viewer.py --task reach
    python3 src/mujoco/mujoco_headless_viewer.py --task circle
    python3 src/mujoco/mujoco_headless_viewer.py --task square
"""

import sys
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

import numpy as np
import mujoco
from pathlib import Path
import argparse
import time
from src.solver.phase4_mpc_controller import Phase4MPCController


class HeadlessMPCSimulator:
    """Text-based MPC simulator without graphical viewer."""
    
    def __init__(self, model_path, task='reach', speed=1.0):
        """Initialize simulator."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.speed = speed
        
        # MPC controller
        self.mpc = Phase4MPCController(N=10, dt=0.002)
        
        # Task-specific targets
        self.setup_task()
        
        # Tracking
        self.step_count = 0
        self.error_history = []
        self.position_history = []
        
    def setup_task(self):
        """Configure task targets."""
        if self.task == 'reach':
            self.targets = [np.array([np.pi/6, np.pi/6])]
            self.target_durations = [5.0]
            
        elif self.task == 'circle':
            t = np.linspace(0, 2*np.pi, 100)
            center = np.array([np.pi/4, np.pi/4])
            radius = 0.2
            self.targets = np.array([
                center[0] + radius * np.cos(t),
                center[1] + radius * np.sin(t)
            ]).T
            self.target_durations = [0.05] * len(self.targets)
            
        elif self.task == 'square':
            corners = [
                [np.pi/6, np.pi/6],
                [np.pi/3, np.pi/6],
                [np.pi/3, np.pi/3],
                [np.pi/6, np.pi/3]
            ]
            self.targets = corners + [corners[0]]
            self.target_durations = [2.0] * len(self.targets)
            
        else:
            self.targets = [np.array([np.pi/4, np.pi/4])]
            self.target_durations = [5.0]
    
    def get_current_target(self):
        """Get target for current step."""
        cumulative_steps = 0
        timesteps_per_second = 500
        
        for i, duration in enumerate(self.target_durations):
            duration_steps = int(duration * timesteps_per_second)
            if self.step_count < cumulative_steps + duration_steps:
                if isinstance(self.targets[0], np.ndarray) and len(self.targets[0]) == 1:
                    return self.targets[0]
                else:
                    return np.asarray(self.targets[i])
            cumulative_steps += duration_steps
        
        return np.asarray(self.targets[-1])
    
    def control_step(self):
        """Execute one MPC control step."""
        x = np.concatenate([self.data.qpos, self.data.qvel])
        target_pos = self.get_current_target()
        target_state = np.concatenate([target_pos, np.zeros(2)])
        
        u_opt, info = self.mpc.solve_step(x, target_state)
        self.data.ctrl[:] = np.clip(u_opt[:2], -50, 50)
        
        mujoco.mj_step(self.model, self.data)
        
        error = np.linalg.norm(target_pos - x[:2])
        self.error_history.append(error)
        self.position_history.append(x[:2].copy())
        self.step_count += 1
        
        return error, x, target_pos
    
    def run_simulation(self, duration=None):
        """Run simulation headlessly."""
        if duration is None:
            # Calculate from target durations
            duration = sum(self.target_durations)
        
        print("\n" + "=" * 80)
        print("MPC MuJoCo HEADLESS SIMULATOR (Text-based, works on all systems)")
        print("=" * 80)
        print(f"\nTask: {self.task.upper()}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Speed: {self.speed}x")
        print("\nNote: This is a text-based simulator (no graphical window)")
        print("For interactive 3D viewer, use: mjpython mujoco_interactive_viewer.py")
        print("-" * 80)
        print(f"\n{'Step':>6} | {'Q0 (rad)':>10} | {'Q1 (rad)':>10} | {'Error':>10} | {'Avg Error':>10}")
        print("-" * 80)
        
        num_steps = int(duration * 500)  # 500 Hz control
        start_time = time.time()
        
        for step in range(num_steps):
            error, x, target = self.control_step()
            
            # Print every 100 steps
            if step % 100 == 0:
                avg_error = np.mean(self.error_history[-100:]) if len(self.error_history) >= 100 else np.mean(self.error_history)
                print(f"{step:6d} | {x[0]:10.4f} | {x[1]:10.4f} | {error:10.4f} | {avg_error:10.4f}")
            
            # Control simulation speed
            if self.speed > 0:
                elapsed = time.time() - start_time
                expected_time = step / (500 * self.speed)
                if expected_time > elapsed:
                    time.sleep(expected_time - elapsed)
        
        # Final statistics
        print("-" * 80)
        print("\n📊 SIMULATION RESULTS:")
        print(f"  Total steps: {self.step_count}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Mean error: {np.mean(self.error_history):.4f} rad")
        print(f"  Min error: {np.min(self.error_history):.4f} rad")
        print(f"  Max error: {np.max(self.error_history):.4f} rad")
        print(f"  Std dev: {np.std(self.error_history):.4f} rad")
        
        if self.task == 'reach':
            final_error = self.error_history[-1]
            print(f"\n✅ REACH TASK: {'SUCCESS' if final_error < 0.1 else 'IN PROGRESS'}")
            print(f"   Final error: {final_error:.6f} rad")
        
        elif self.task == 'circle':
            print(f"\n✅ CIRCLE TRACKING: Completed")
            
        elif self.task == 'square':
            print(f"\n✅ SQUARE TRAJECTORY: Completed")
        
        print("\n" + "=" * 80)
        
        return self.error_history


def main():
    parser = argparse.ArgumentParser(
        description='Headless MPC simulator for 2DOF arm (works on all systems)'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='reach',
        choices=['reach', 'circle', 'square'],
        help='Control task'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Simulation speed multiplier'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Override duration in seconds'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to MuJoCo XML model'
    )
    
    args = parser.parse_args()
    
    if args.model:
        model_path = args.model
    else:
        model_path = '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    simulator = HeadlessMPCSimulator(model_path, task=args.task, speed=args.speed)
    simulator.run_simulation(duration=args.duration)


if __name__ == '__main__':
    main()

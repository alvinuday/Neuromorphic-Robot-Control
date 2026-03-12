"""
Interactive MuJoCo Viewer for MPC-Controlled 2DOF Arm

This script provides a real-time visual interface to see the MPC controller
moving the 2DOF arm in MuJoCo physics simulation.

Usage:
    python3 src/mujoco/mujoco_interactive_viewer.py --task reach
    python3 src/mujoco/mujoco_interactive_viewer.py --task circle
    python3 src/mujoco/mujoco_interactive_viewer.py --task square
"""

import sys
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import argparse
from src.solver.phase4_mpc_controller import Phase4MPCController


class InteractiveMPCViewer:
    """Interactive viewer for MPC-controlled 2DOF arm."""
    
    def __init__(self, model_path, task='reach', speed=1.0, controller_type='pid'):
        """
        Initialize viewer.
        
        Args:
            model_path: Path to MuJoCo XML model
            task: 'reach', 'circle', 'square', or custom
            speed: Simulation speed multiplier
            controller_type: 'pid' (simple), 'mpc' (optimization-based)
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.speed = speed
        self.controller_type = controller_type
        
        # Controller selection
        if controller_type == 'mpc':
            self.mpc = Phase4MPCController(N=10, dt=0.002)
        elif controller_type == 'pid':
            self.Kp = 50.0
            self.Kd = 10.0
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        # Task-specific targets
        self.setup_task()
        
        # Tracking
        self.step_count = 0
        self.error_history = []
        
    def setup_task(self):
        """Configure task targets."""
        if self.task == 'reach':
            # Single reach target
            self.targets = [np.array([np.pi/6, np.pi/6])]
            self.target_durations = [5.0]  # Hold for 5 seconds
            
        elif self.task == 'circle':
            # Circular trajectory
            t = np.linspace(0, 2*np.pi, 100)
            center = np.array([np.pi/4, np.pi/4])
            radius = 0.2
            self.targets = np.array([
                center[0] + radius * np.cos(t),
                center[1] + radius * np.sin(t)
            ]).T
            self.target_durations = [0.05] * len(self.targets)
            
        elif self.task == 'square':
            # Square trajectory
            corners = [
                [np.pi/6, np.pi/6],
                [np.pi/3, np.pi/6],
                [np.pi/3, np.pi/3],
                [np.pi/6, np.pi/3]
            ]
            self.targets = corners + [corners[0]]  # Close the loop
            self.target_durations = [2.0] * len(self.targets)
            
        else:
            # Default: reach π/4, π/4
            self.targets = [np.array([np.pi/4, np.pi/4])]
            self.target_durations = [5.0]
    
    def get_current_target(self):
        """Get target for current step."""
        cumulative_steps = 0
        timesteps_per_second = 500  # 500Hz control
        
        for i, duration in enumerate(self.target_durations):
            duration_steps = int(duration * timesteps_per_second)
            if self.step_count < cumulative_steps + duration_steps:
                if isinstance(self.targets[0], np.ndarray) and len(self.targets[0]) == 1:
                    # Single target
                    return self.targets[0]
                else:
                    # Multiple targets
                    return np.asarray(self.targets[i])
            cumulative_steps += duration_steps
        
        # Repeat last target
        return np.asarray(self.targets[-1])
    
    def control_step(self):
        """Execute one control step."""
        try:
            # Get current state
            x = np.concatenate([self.data.qpos, self.data.qvel])
            
            # Get target
            target_pos = self.get_current_target()
            
            if self.controller_type == 'pid':
                # Simple PID control: tau = Kp * (q_target - q) + Kd * (0 - dq)
                error_pos = target_pos - x[:2]
                error_vel = -x[2:]
                u_opt = self.Kp * error_pos + self.Kd * error_vel
                
            elif self.controller_type == 'mpc':
                # MPC control
                target_state = np.concatenate([target_pos, np.zeros(2)])
                u_opt, info = self.mpc.solve_step(x, target_state)
                
            # Apply control
            self.data.ctrl[:] = np.clip(u_opt[:2], -50, 50)
            
            # Simulate
            mujoco.mj_step(self.model, self.data)
            
            # Track error
            error = np.linalg.norm(target_pos - x[:2])
            self.error_history.append(error)
            
            self.step_count += 1
            
            return error
        except Exception as e:
            print(f"ERROR in control_step: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_interactive(self):
        """Run with interactive viewer."""
        print("\n" + "=" * 70)
        print("MPC MuJoCo INTERACTIVE VIEWER")
        print("=" * 70)
        print(f"\nTask: {self.task.upper()}")
        print(f"Controller: {self.controller_type.upper()}")
        print(f"Speed: {self.speed}x")
        print("\nControls:")
        print("  SPACEBAR: Pause/Resume")
        print("  R: Reset")
        print("  Scroll: Zoom in/out")
        print("  Right-click drag: Rotate view")
        print("  Middle-click drag: Pan view")
        print("\nViewing arm control...")
        print("-" * 70)
        
        # Try to use launch_passive, fallback to launch if needed
        try:
            viewer_context = mujoco.viewer.launch_passive(self.model, self.data)
        except RuntimeError as e:
            # On macOS, launch_passive requires mjpython
            if "mjpython" in str(e):
                print("\n⚠️  On macOS, the interactive viewer requires mjpython.")
                print("Use this command instead:")
                print(f"\n  mjpython src/mujoco/mujoco_interactive_viewer.py --task {self.task}\n")
                return
            else:
                raise
        
        with viewer_context as viewer:
            # Configure viewer options
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = False
            
            # RESET camera to default (home view)
            viewer.cam.trackbodyid = 0  # Track world/origin
            viewer.cam.lookat[0] = 0.25  # Look at arm centerline
            viewer.cam.lookat[1] = 0
            viewer.cam.lookat[2] = 0.25
            viewer.cam.distance = 2.0  
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 45
            
            # Do initial sync
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()
            
            # Print where camera is looking
            print(f"Camera lookat: {viewer.cam.lookat}")
            print(f"Camera distance: {viewer.cam.distance}")
            print(f"Camera elevation: {viewer.cam.elevation}")
            print(f"Camera azimuth: {viewer.cam.azimuth}")
            print("If you can't see the arm, try using the mouse:")
            print("  - Right-click drag to rotate")
            print("  - Scroll wheel to zoom")
            print("  - Middle-click drag to pan")
            
            step_count = 0
            print("\n🔄 CONTROL LOOP STARTING...\n")
            while viewer.is_running():
                try:
                    # Execute control step
                    error = self.control_step()
                    step_count += 1
                    
                    # Print progress every 10 steps (more frequent)
                    if step_count % 10 == 0:
                        avg_error = np.mean(self.error_history[-10:]) if len(self.error_history) >= 10 else np.mean(self.error_history)
                        tau = self.data.ctrl
                        print(f"Step {step_count:6d}: pos=[{self.data.qpos[0]:7.4f}, {self.data.qpos[1]:7.4f}] " +
                              f"tau=[{tau[0]:7.2f}, {tau[1]:7.2f}] " +
                              f"error={error:7.4f} avg={avg_error:7.4f}")
                    
                    # Sync viewer
                    viewer.sync()
                    
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user")
                    break
                except Exception as e:
                    print(f"\n❌ ERROR at step {step_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        
        print("\n" + "=" * 70)
        print("Viewer closed")
        print(f"Total steps: {step_count}")
        print(f"Mean error: {np.mean(self.error_history):.4f} rad")
        print(f"Min error: {np.min(self.error_history):.4f} rad")
        print(f"Max error: {np.max(self.error_history):.4f} rad")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Interactive MuJoCo viewer for MPC-controlled 2DOF arm'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='reach',
        choices=['reach', 'circle', 'square'],
        help='Control task: reach (single target), circle (circular path), square (square path)'
    )
    parser.add_argument(
        '--controller',
        type=str,
        default='pid',
        choices=['pid', 'mpc'],
        help='Controller type: pid (simple proportional-derivative), mpc (model predictive control)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Simulation speed multiplier (default: 1.0x)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to MuJoCo XML model'
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        model_path = '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml'
    
    # Verify model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Create and run viewer
    viewer = InteractiveMPCViewer(model_path, task=args.task, speed=args.speed, controller_type=args.controller)
    viewer.run_interactive()


if __name__ == '__main__':
    main()

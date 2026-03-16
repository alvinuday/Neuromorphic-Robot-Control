"""
Pick and Place Controller with Pinocchio Integration
=====================================================
Complete headless implementation for robot pick and place task.

Run:
  python pick_and_place_controller.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Import our modules
try:
    import mujoco
except ImportError:
    print("✗ MuJoCo not installed")
    sys.exit(1)

try:
    from motion_planning import MotionPlanningSequence
    from gripper_control import GripperController, detect_block_contact
    from pinocchio_utils import PinocchioRobotModel
    from evaluate_task import PickAndPlaceEvaluator
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class PickAndPlaceController:
    """Main controller combining MPC, gripper, and task evaluation"""
    
    def __init__(self):
        """Initialize controller and robot model"""
        
        # Paths
        self.root_dir = Path(__file__).parent.parent
        self.xml_path = str(self.root_dir / "assets/xarm_6dof.xml")
        self.log_path = Path(__file__).parent / "pick_and_place_log.csv"
        
        # Load MuJoCo model
        print(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"✓ Model loaded: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        
        # Configuration
        self.N_ARM = 6
        self.N_HZ_MPC = 50
        self.DT_SIM = 0.002
        self.MPC_EVERY = int(1.0 / (self.N_HZ_MPC * self.DT_SIM))
        
        # Controllers
        self.gripper = GripperController(close_torque=50.0, hold_torque=25.0)  # MASSIVELY increased from 10/5
        self.motion_planner = MotionPlanningSequence(
            home=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            grasp=np.array([0.3, -0.8, -2.0, 0.0, 0.0, 0.0]),  # CHANGED: Simpler, more reachable target
            move_time=8.0,
            hold_time=2.0
        )
        self.pinocchio_model = None
        self.evaluator = PickAndPlaceEvaluator()
        
        # Try to load Pinocchio
        try:
            self.pinocchio_model = PinocchioRobotModel(str(self.root_dir / "assets/xarm_6dof.urdf"))
            if self.pinocchio_model.initialized:
                print("✓ Pinocchio model loaded")
            else:
                print("⚠ Pinocchio available but model init failed")
        except Exception as e:
            print(f"⚠ Pinocchio not available: {e}")
        
        # State tracking
        self.step = 0
        self.t_sim = 0.0
        self.log_data = []
        self.gripper_phase = "IDLE"
        self.block_contact = False
    
    def get_block_position(self):
        """Get current block pose from MuJoCo"""
        # Block is body 2 (red_block) in xarm_6dof.xml
        # Bodies: 0=worldbody, 1=table, 2=red_block, 3=xarm_base, ...
        block_body_id = 2
        block_pos = self.data.xpos[block_body_id]
        return block_pos.copy()
    
    def get_arm_state(self):
        """Extract arm joint state from full robot state"""
        # Arm joints are at qpos[7:13], qvel[6:12]
        q = self.data.qpos[7:13].copy()
        dq = self.data.qvel[6:12].copy()
        return q, dq
    
    def compute_mpc_control(self, q, dq, q_ref, dq_ref):
        """
        Simple proportional control as MPC alternative (for now).
        Real MPC would be more sophisticated.
        
        Args:
            q, dq: Current joint state
            q_ref, dq_ref: Reference trajectory
        
        Returns:
            tau: Control torques [6]
        """
        # HIGH GAINS to move arm firmly toward target
        # These are needed because the arm is heavy and needs strong actuation
        Kp = np.array([200.0, 150.0, 120.0, 80.0, 60.0, 40.0])  # INCREASED from [50, 40, 30, 20, 15, 10]
        Kd = np.array([20.0, 15.0, 12.0, 8.0, 6.0, 4.0])         # INCREASED proportionally
        
        # PD control
        q_error = q_ref - q
        dq_error = dq_ref - dq
        tau = Kp * q_error + Kd * dq_error
        
        # Torque limits (from XML: joint1±20, joint2±15, joint3±15, joint4±10, joint5±8, joint6±6)
        tau_max = np.array([20.0, 15.0, 15.0, 10.0, 8.0, 6.0])  # Match motor limits in XML
        tau = np.clip(tau, -tau_max, tau_max)
        
        return tau
    
    def update_gripper(self, block_contact):
        """
        Update gripper state based on task phase.
        
        Phase 0-10s: Approach (move to grasp, idle gripper)
        Phase 10-11.5s: Close gripper
        Phase 11.5-12s: Grasp confirm
        Phase 12-20.5s: Hold and move
        Phase 20.5-21.5s: Open gripper  
        Phase 21.5-26s: Retreat
        Total cycle: 26s
        """
        phase_cycle = 26.0  # CHANGED from 16.0 to match new timing
        t_phase = self.t_sim % phase_cycle
        
        if t_phase < 10.0:
            command = "IDLE"       # Moving to grasp
        elif t_phase < 11.5:
            command = "CLOSING"    # Close gripper
        elif t_phase < 12.0:
            command = "HOLDING"    # Confirm contact
        elif t_phase < 20.5:
            command = "HOLDING"    # Hold while moving
        elif t_phase < 21.5:
            command = "OPENING"    # Open gripper
        else:
            command = "IDLE"       # Return home
        
        tau_gripper = self.gripper.update(self.t_sim, command, contact_force=1.0 if block_contact else 0.0)
        
        return tau_gripper, command
    
    def step_simulation(self):
        """Execute one simulation step"""
        
        # Get current state
        q, dq = self.get_arm_state()
        block_pos = self.get_block_position()
        
        # Get reference trajectory
        q_ref, dq_ref = self.motion_planner.get_reference(self.t_sim)
        
        # Compute arm control
        tau_arm = self.compute_mpc_control(q, dq, q_ref, dq_ref)
        
        # Update gripper
        self.block_contact, _ = detect_block_contact(self.model, self.data, gripper_body_id=14)
        tau_gripper, gripper_phase = self.update_gripper(self.block_contact)
        self.gripper_phase = gripper_phase
        
        # Set control
        self.data.ctrl[0:6] = tau_arm
        self.data.ctrl[6:8] = tau_gripper
        
        # Record data
        self.evaluator.record_position(block_pos, self.t_sim)
        
        self.log_data.append({
            'time': self.t_sim,
            'q': q.copy(),
            'dq': dq.copy(),
            'q_ref': q_ref.copy(),
            'tau': tau_arm.copy(),
            'tau_gripper': tau_gripper.copy(),
            'block_pos': block_pos.copy(),
            'block_contact': self.block_contact,
            'gripper_phase': gripper_phase,
        })
        
        # Step MuJoCo
        mujoco.mj_step(self.model, self.data)
        
        # Increment
        self.t_sim += self.DT_SIM
        self.step += 1
    
    def run_simulation(self, duration=20.0):
        """Run simulation for specified duration"""
        print(f"\nRunning pick and place simulation for {duration}s...")
        print("-" * 70)
        
        num_steps = int(duration / self.DT_SIM)
        start_time = time.time()
        
        for step_idx in range(num_steps):
            self.step_simulation()
            
            # Print progress
            if step_idx % 500 == 0:
                elapsed = time.time() - start_time
                block_pos = self.get_block_position()
                print(f"Step {step_idx:6d}/{num_steps} | t={self.t_sim:6.2f}s | "
                      f"block=[{block_pos[0]:6.3f}, {block_pos[1]:6.3f}, {block_pos[2]:6.3f}] "
                      f"| gripper={self.gripper_phase:8s} | wall_time={elapsed:6.2f}s")
        
        elapsed_wall = time.time() - start_time
        elapsed_sim = self.t_sim
        
        print("-" * 70)
        print(f"✓ Simulation complete")
        print(f"  Simulated time:  {elapsed_sim:.2f}s")
        print(f"  Wall clock time: {elapsed_wall:.2f}s")
        print(f"  Speed: {elapsed_sim/elapsed_wall:.1f}x real-time")
    
    def evaluate_and_report(self):
        """Evaluate task success and print report"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        metrics = self.evaluator.print_report()
        
        return metrics
    
    def save_logs(self):
        """Save simulation logs to CSV"""
        import csv
        
        log_path = Path(__file__).parent / "pick_and_place_log.csv"
        
        if not self.log_data:
            print("No log data to save")
            return
        
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'time',
                'q0', 'q1', 'q2', 'q3', 'q4', 'q5',
                'dq0', 'dq1', 'dq2', 'dq3', 'dq4', 'dq5',
                'q_ref_0', 'q_ref_1', 'q_ref_2', 'q_ref_3', 'q_ref_4', 'q_ref_5',
                'tau0', 'tau1', 'tau2', 'tau3', 'tau4', 'tau5',
                'tau_gripper0', 'tau_gripper1',
                'block_x', 'block_y', 'block_z',
                'block_contact', 'gripper_phase'
            ])
            
            # Data
            for entry in self.log_data:
                row = [
                    entry['time'],
                    *entry['q'],
                    *entry['dq'],
                    *entry['q_ref'],
                    *entry['tau'],
                    *entry['tau_gripper'],
                    *entry['block_pos'],
                    1 if entry['block_contact'] else 0,
                    entry['gripper_phase']
                ]
                writer.writerow(row)
        
        print(f"\n✓ Logs saved to {log_path}")


def main():
    """Main execution"""
    print("=" * 70)
    print("PICK AND PLACE CONTROLLER - HEADLESS MODE")
    print("=" * 70)
    
    try:
        controller = PickAndPlaceController()
        
        # Run simulation
        controller.run_simulation(duration=40.0)  # INCREASED from 20.0s
        
        # Evaluate
        metrics = controller.evaluate_and_report()
        
        # Save logs
        controller.save_logs()
        
        # Return success status
        if metrics['success']:
            print("\n✓✓✓ PICK AND PLACE SUCCESSFUL ✓✓✓")
            return 0
        else:
            print("\n✗ Pick and place failed - block not in end boundary")
            return 1
    
    except KeyboardInterrupt:
        print("\n✗ Simulation interrupted")
        return 2
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

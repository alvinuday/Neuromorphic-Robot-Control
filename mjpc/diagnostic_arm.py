"""
Diagnostic script to debug arm motion
======================================
Check if arm is moving and gripper is working.
"""

import sys
import numpy as np
from pathlib import Path

try:
    import mujoco
except ImportError:
    print("✗ MuJoCo not installed")
    sys.exit(1)

from motion_planning import MotionPlanningSequence
from gripper_control import GripperController

# Load model
root_dir = Path(__file__).parent.parent
xml_path = str(root_dir / "assets/xarm_6dof.xml")

print(f"Loading model from: {xml_path}")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print(f"✓ Model loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}")
print(f"  Bodies: {model.nbody}")
print(f"  Actuators: {model.nu}")

# Get arm state
def get_arm_state():
    q = data.qpos[7:13].copy()
    dq = data.qvel[6:12].copy()
    return q, dq

# Get gripper state
def get_gripper_state():
    gripper_q = data.qpos[13:15]
    return gripper_q.copy()

# Get block position
def get_block_pos():
    return data.xpos[1].copy()  # red_block is body 1

# Initialize
planner = MotionPlanningSequence(
    home=np.array([0, 0, 0, 0, 0, 0]),
    grasp=np.array([1.23, -1.06, -3.61, 0.05, -1.27, 0.36]),
    move_time=6.0,
    hold_time=4.0
)
gripper = GripperController()

print("\n" + "=" * 80)
print("DIAGNOSTIC TEST: ARM MOTION & GRIPPER")
print("=" * 80)

print("\nInitial state:")
q, dq = get_arm_state()
gripper_q = get_gripper_state()
block_pos = get_block_pos()
print(f"  Arm q:        {q}")
print(f"  Arm dq:       {dq}")
print(f"  Gripper q:    {gripper_q}")
print(f"  Block pos:    {block_pos}")

# Simulate for 16 seconds (one full cycle)
dt = 0.002
duration = 16.0
num_steps = int(duration / dt)

print(f"\nRunning {duration}s simulation ({num_steps} steps)...")
print("\nTime | q[0] (rad) | q[1] (rad) | Gripper[0] | Gripper[1] | Block_X | Block_Z | Arm Control | Gripper Cmd")
print("-" * 110)

for step in range(num_steps):
    t_sim = step * dt
    
    # Get reference trajectory
    q_ref, dq_ref = planner.get_reference(t_sim)
    
    # Simple PD control
    Kp = np.array([50, 40, 30, 20, 15, 10])
    Kd = np.array([5, 4, 3, 2, 1.5, 1]) 
    q, dq = get_arm_state()
    tau = Kp * (q_ref - q) + Kd * (dq_ref - dq)
    tau = np.clip(tau, -np.array([25, 20, 20, 15, 12, 10]), np.array([25, 20, 20, 15, 12, 10]))
    
    # Gripper control
    cycle_t = t_sim % 16.0
    if cycle_t < 2.0:
        grip_cmd = "IDLE"
    elif cycle_t < 3.5:
        grip_cmd = "CLOSING"
    elif cycle_t < 10.0:
        grip_cmd = "HOLDING"
    else:
        grip_cmd = "OPENING"
    
    tau_gripper = gripper.update(t_sim, grip_cmd)
    
    # Set control
    data.ctrl[0:6] = tau
    data.ctrl[6:8] = tau_gripper
    
    # Step
    mujoco.mj_step(model, data)
    
    # Get state
    q, dq = get_arm_state()
    gripper_q = get_gripper_state()
    block_pos = get_block_pos()
    
    # Print every 2 seconds
    if step % 1000 == 0:
        arm_moving = "YES" if np.linalg.norm(dq) > 0.01 else "NO "
        gripper_moving = "YES" if np.linalg.norm(tau_gripper) > 0.1 else "NO "
        print(f"{t_sim:5.1f}s | {q[0]:10.3f} | {q[1]:10.3f} | {gripper_q[0]:10.3f} | {gripper_q[1]:10.3f} | "
              f"{block_pos[0]:7.3f} | {block_pos[2]:7.3f} | {arm_moving} | {gripper_moving} ({grip_cmd})")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

final_q, final_dq = get_arm_state()
final_gripper = get_gripper_state()
final_block = get_block_pos()

print(f"\nFinal arm angles: {final_q}")
print(f"Final gripper angles: {final_gripper}")
print(f"Final block position: {final_block}")

# Check joint movement
if np.max(np.abs(final_q - np.array([0,0,0,0,0,0]))) > 0.1:
    print("\n✓ Arm is moving")
else:
    print("\n✗ Arm NOT moving (stayed at home)")

# Check gripper movement
if np.max(np.abs(final_gripper)) > 0.01:
    print("✓ Gripper is moving")
else:
    print("✗ Gripper NOT moving (no finger closure)")

# Check block movement
initial_block_x = 0.2
if abs(final_block[0] - initial_block_x) > 0.01:
    print(f"✓ Block moved (Δx={final_block[0]-initial_block_x:.3f}m)")
else:
    print(f"✗ Block NOT moved (stayed at x={initial_block_x}m)")

print("\n" + "=" * 80)

"""
Simple Pick and Place: Push Block

Strategy: Push the block along the table instead of trying to grip it
- Move arm to contact block (x=0.2, y=0, z=0.52)
- Push it forward to target (x=0.4, y=0, z=0.52)
- This is simpler than gripping and has guaranteed contact
"""

import sys
import time
import numpy as np
import mujoco
from pathlib import Path

xml_path = Path(__file__).parent.parent / "assets/xarm_6dof.xml"
print(f"Loading: {xml_path}")

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"✓ Model: nq={model.nq}, nv={model.nv}")

# Initialize arm to home
data.qpos[7:13] = [0, 0, 0, 0, 0, 0]

# Simple trajectory: push block from (0.2, 0, 0.52) to (0.4, 0, 0.52)
# This requires reachable joint configs

# Intermediate waypoints (learned by trial)
# Phase 1: Move arm to approach block (first 5s)
approach_q = np.array([0.0, -0.5, -1.5, 0.0, 0.0, 0.0])

# Phase 2: Move arm forward to push block (5-15s)
push_q = np.array([0.0, -0.3, -1.2, 0.0, 0.0, 0.0])

# Phase 3: Retract (15-20s)
retract_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

print("\nRunning push-block simulation (20s)...")
print("Time | Block X | Block Z | Arm q[0] | Arm q[1] | Arm q[2] | Phase")
print("-" * 75)

Kp = np.array([100.0, 100.0, 80.0, 50.0, 40.0, 30.0])
Kd = np.array([10.0, 10.0, 8.0, 5.0, 4.0, 3.0])
tau_max = np.array([20.0, 15.0, 15.0, 10.0, 8.0, 6.0])

dt = 0.002
duration = 20.0
num_steps = int(duration / dt)

final_block_x = None

for step in range(num_steps):
    t = step * dt
    
    # Determine target based on time
    q_current = data.qpos[7:13]
    
    if t < 5.0:
        q_target = approach_q
        phase = "APPROACH"
    elif t < 15.0:
        q_target = push_q
        phase = "PUSH"
    else:
        q_target = retract_q
        phase = "RETRACT"
    
    # Control
    dq = data.qvel[6:12]
    tau = Kp * (q_target - q_current) + Kd * (0 - dq)
    tau = np.clip(tau, -tau_max, tau_max)
    
    data.ctrl[0:6] = tau
    data.ctrl[6:8] = 0  # No gripper
    
    # Step
    mujoco.mj_step(model, data)
    
    # Print progress every 2.5 seconds
    if step % 1250 == 0:
        block_pos = data.xpos[1]
        print(f"{t:5.1f}s | {block_pos[0]:7.3f} | {block_pos[2]:7.3f} | "
              f"{q_current[0]:8.3f} | {q_current[1]:8.3f} | {q_current[2]:8.3f} | {phase}")
    
    final_block_x = data.xpos[1][0]

print("-" * 75)

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

block_final = data.xpos[1]
print(f"\nFinal block position: ({block_final[0]:.3f}, {block_final[1]:.3f}, {block_final[2]:.3f})")
print(f"Initial position:    (0.200, 0.000, 0.520)")
print(f"Target position:     (0.400, 0.000, 0.520)")

displacement = block_final[0] - 0.2
print(f"\nDisplacement: {displacement:.3f}m")

if displacement > 0.15:
    print("✓ SUCCESS: Block moved significantly forward!")
elif displacement > 0.05:
    print("⚠ PARTIAL: Block moved but not far enough")
else:
    print("✗ FAILED: Block did not moveforward")

# Check boundaries
START_X = (0.15, 0.25)
END_X = (0.35, 0.45)

in_start = START_X[0] <= block_final[0] <= START_X[1]
in_end = END_X[0] <= block_final[0] <= END_X[1]

print(f"\nIn start boundary (x=[0.15-0.25]): {in_start}")
print(f"In end boundary (x=[0.35-0.45]): {in_end}")

if in_end:
    print("\n✓✓✓ TASK SUCCESS ✓✓✓")
else:
    print(f"\n✗ Task failed (need x > 0.35, have x={block_final[0]:.3f})")

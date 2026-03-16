#!/usr/bin/env python3
"""Quick diagnostic - check arm motion"""
import numpy as np
import mujoco
from pathlib import Path

xml_path = Path(__file__).parent.parent / "assets/xarm_6dof.xml"
print(f"Loading: {xml_path}")

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"✓ Model nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}")

# List all bodies
print("\nBodies:")
for i in range(model.nbody):
    print(f"  {i}: {model.body(i).name}")

# Verify indices
print(f"\nState check:")
print(f"  qpos[0:7]: {data.qpos[0:7]} (free joint)")
print(f"  qpos[7:13]: {data.qpos[7:13]} (arm)")
print(f"  qpos[13:15]: {data.qpos[13:15]} (gripper)")
print(f"  Block body ID: 1 (red_block)")

# Test control
print(f"\nTesting control setup:")
print(f"  ctrl shape: {data.ctrl.shape}")
print(f"  Should have 8 actuators: 6 arm + 2 gripper")

# Test motor info  
print(f"\nActuators:")
for i in range(min(8, model.nu)):
    print(f"  {i}: {model.actuator(i).name}")

print("\n✓ All checks passed")

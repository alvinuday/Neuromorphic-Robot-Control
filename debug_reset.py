#!/usr/bin/env python3
from simulation.envs.xarm_env import XArmEnv
import numpy as np
import mujoco

env = XArmEnv()

print("Joint structure:")
for i in range(env.model.njnt):
    jnt = env.model.joint(i)
    qpos_adr = jnt.qposadr[0]
    print(f"  {i}: {jnt.name:20s} qpos_adr={qpos_adr}")

print("\nInitial state after env creation:")
print(f"  qpos[7] (gripper_right): {env.data.qpos[7]:.4f}")
print(f"  entire qpos: {env.data.qpos}")

# Step with some control
print("\nAfter step():")
action = np.array([1.0, 0.5, -0.5, 0.2, 0.1, 0.15, 0.01, 0.01])
env.step(action)
print(f"  qpos[7] (gripper_right): {env.data.qpos[7]:.6f}")

# Now reset
print("\nCalling reset()...")
env.reset()

print("\nAfter reset():")
print(f"  qpos[7] (gripper_right): {env.data.qpos[7]:.4f}")
print(f"  entire qpos: {env.data.qpos}")

q = env.get_joint_pos()
print("\nget_joint_pos():", q)

env.close()

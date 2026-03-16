"""Debug arm kinematics"""
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("assets/xarm_6dof.xml")
data = mujoco.MjData(model)

# Print all bodies
print("All bodies in model:")
for i in range(model.nbody):
    print(f"  {i}: {model.body(i).name}")

# Set test joint angles and check all body positions
test_q = [0.0, -1.0, 1.0, -1.0, 0.0, 0.0]
data.qpos[:6] = test_q
mujoco.mj_forward(model, data)

print(f"\nWith q={test_q}:")
print("Body positions:")
for i in range(model.nbody):
    name = model.body(i).name
    pos = data.xpos[i]
    print(f"  {name:20s}: {pos}")

# Get the red_block position 
block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
print(f"\nRed block ID: {block_id}, position: {data.xpos[block_id]}")

# Get gripper_base position with different joint angles
test_q2 = [0.0, -1.5, 0.5, -0.5, 0.0, 0.0]
data.qpos[:6] = test_q2
mujoco.mj_forward(model, data)
gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
print(f"\nWith q={test_q2}:")
print(f"Gripper position: {data.xpos[gripper_id]}")

text_q3 = [0.0, -2.0, -0.5, 0.5, 0.0, 0.0]
data.qpos[:6] = test_q3
mujoco.mj_forward(model, data)
print(f"\nWith q={test_q3}:")
print(f"Gripper position: {data.xpos[gripper_id]}")

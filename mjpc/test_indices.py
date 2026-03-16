"""Quick test with corrected joint indices"""
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("assets/xarm_6dof.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

print("Model: nq={}, nv={}".format(model.nq, model.nv))
print("Target: [0.2, 0.0, 0.52]")
print()

# ARM JOINTS AT: qpos[7:13], qvel[6:12]
configs = [
    ("Neutral", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("Reach 1", [0.0, -1.0, 0.5, -0.5, 0.0, 0.0]),
    ("Reach 2", [0.2, -1.5, 0.3, 0.0, 0.0, 0.0]),
]

gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
target = np.array([0.2, 0.0, 0.52])

for name, q_arm in configs:
    # SET ARM AT CORRECT INDICES
    data.qpos[7:13] = q_arm
    mujoco.mj_forward(model, data)
    
    gripper_pos = data.xpos[gripper_id]
    dist = np.linalg.norm(gripper_pos - target)
    
    print(f"{name:15s} q={q_arm}")
    print(f"  Gripper: {gripper_pos}  Distance: {dist:.4f}m\n")

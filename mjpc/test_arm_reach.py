"""Direct joint command test - no MPC, just test arm reach"""
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "assets/xarm_6dof.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Initialize red_block
red_block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
red_block_jnt_id = model.body_jntadr[red_block_body_id]
red_block_qpos_id = model.jnt_qposadr[red_block_jnt_id]

# Set block position
data.qpos[red_block_qpos_id:red_block_qpos_id+3] = np.array([0.2, 0.0, 0.52])
data.qpos[red_block_qpos_id+3:red_block_qpos_id+7] = np.array([1.0, 0.0, 0.0, 0.0])

# Test different joint configurations
configs = [
    ("Home (neutral)", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("Reach down 1", [0.0, -1.0, 0.5, -0.5, 0.0, 0.0]),
    ("Reach down 2", [0.0, -1.5, 0.3, 0.0, 0.0, 0.0]),
    ("Reach down 3", [0.2, -1.2, 0.2, 0.0, 0.0, 0.0]),
    ("Reach elbow down", [0.0, 0.5, -0.5, -1.0, 0.0, 0.0]),
]

gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
target = np.array([0.2, 0.0, 0.52])

print("\n=== Testing Joint Configurations ===\n")
for name, q in configs:
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    gripper_pos = data.xpos[gripper_body_id]
    dist = np.linalg.norm(gripper_pos - target)
    print(f"{name:20s}")
    print(f"  Angles: {np.array(q).__str__()}")
    print(f"  Gripper: {gripper_pos}")
    print(f"  Distance to target: {dist:.4f}m")
    print()

# Now run interactive viewer with manual joint control
print("\n=== Starting Interactive Viewer ===")
print("Adjust joint1_motor to move the arm")
print("Try to get gripper to box position [0.2, 0.0, 0.52]\n")

mujoco.mj_resetData(model, data)
data.qpos[red_block_qpos_id:red_block_qpos_id+3] = np.array([0.2, 0.0, 0.52])
data.qpos[red_block_qpos_id+3:red_block_qpos_id+7] = np.array([1.0, 0.0, 0.0, 0.0])

# Start with a config that might reach
data.qpos[:6] = [0.0, -1.0, 0.5, -0.5, 0.0, 0.0]
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        # Slowly sweep joint2
        t = step * 0.002 * 50  # sim speed
        sweep_val = -1.5 + 1.5 * (np.sin(t * 0.5) + 1) / 2
        data.qpos[1] = sweep_val
        data.ctrl[:6] = data.qpos[:6]
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if step % 250 == 0:
            gripper_pos = data.xpos[gripper_body_id]
            dist = np.linalg.norm(gripper_pos - target)
            print(f"j2={sweep_val:.2f} | gripper={gripper_pos} | dist={dist:.4f}m")
        
        step += 1

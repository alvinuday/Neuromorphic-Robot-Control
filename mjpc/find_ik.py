"""Find joint angles to reach the red box using optimization"""
import numpy as np
import mujoco
from scipy.optimize import minimize

model = mujoco.MjModel.from_xml_path("assets/xarm_6dof.xml")
data = mujoco.MjData(model)

# Red box is at (0.2, 0.0, 0.52) in world coordinates
target_pos = np.array([0.2, 0.0, 0.52])

def forward_kinematics(q):
    """Compute gripper position given joint angles - ARM AT qpos[7:13]"""
    mujoco.mj_resetData(model, data)
    data.qpos[7:13] = q  # ARM AT CORRECT INDICES
    mujoco.mj_forward(model, data)
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    return data.xpos[gripper_body_id].copy()

def loss(q):
    """Distance from gripper to target"""
    gripper_pos = forward_kinematics(q)
    return np.linalg.norm(gripper_pos - target_pos)

# Try multiple random starting points
best_result = None
best_loss = float('inf')

print("Optimizing IK solution...")
for trial in range(30):
    x0 = np.random.uniform(-3.0, 3.0, 6)
    result = minimize(loss, x0, method='Nelder-Mead', 
                     options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
    
    if result.fun < best_loss:
        best_loss = result.fun
        best_result = result
        print(f"  Trial {trial}: loss={result.fun:.6f}")

print(f"\nBest loss: {best_loss:.6f}m")
print(f"Best joint angles: {best_result.x}")

# Verify
final_pos = forward_kinematics(best_result.x)
print(f"\nGripper position: {final_pos}")
print(f"Target position:  {target_pos}")
print(f"Error: {np.linalg.norm(final_pos - target_pos):.6f}m")

if best_loss < 0.05:
    print(f"\n✓ Good IK solution found!")
    print(f"Use in arm_mpc.py: {best_result.x}")


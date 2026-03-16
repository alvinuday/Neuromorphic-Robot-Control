"""
Debug: Find reachable grasp position by letting arm extend
"""
import numpy as np
from pathlib import Path
import mujoco

xml_path = Path(__file__).parent.parent / "assets/xarm_6dof.xml"
print(f"Loading: {xml_path}")

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"✓ Model loaded: nq={model.nq}, nv={model.nv}")

# High-gain control towards [0.3, -0.8, -2.0, 0, 0, 0]
Kp = np.array([200.0, 150.0, 120.0, 80.0, 60.0, 40.0])
Kd = np.array([20.0, 15.0, 12.0, 8.0, 6.0, 4.0])
tau_max = np.array([20.0, 15.0, 15.0, 10.0, 8.0, 6.0])

target_q = np.array([0.3, -0.8, -2.0, 0.0, 0.0, 0.0])
max_reached_q = np.zeros(6)
max_reached_t = 0

print("\nRunning 15-second reaching test...")
print("Time(s) |  q[0]  |  q[1]  |  q[2]  | Max ever reached | Block approx\n" + "-" * 80)

for step in range(int(15 / 0.002)):
    t = step * 0.002
    
    # Get current state
    q = data.qpos[7:13].copy()
    dq = data.qvel[6:12].copy()
    
    # Track max reached
    for i in range(6):
        if abs(q[i]) > abs(max_reached_q[i]):
            max_reached_q[i] = q[i]
            max_reached_t = t
    
    # Simple reaching control
    error = target_q - q
    tau = Kp * error + Kd * (0 - dq)  # dq_ref = 0 for steady target
    tau = np.clip(tau, -tau_max, tau_max)
    
    data.ctrl[0:6] = tau
    data.ctrl[6:8] = 0  # No gripper
    
    # Step ahead
    mujoco.mj_step(model, data)
    
    # Print progress
    if step % 2500 == 0:  # Every 5 seconds
        block_pos = data.xpos[1]  # Red block
        print(f"{t:7.1f} | {q[0]:6.2f} | {q[1]:6.2f} | {q[2]:6.2f} | "
              f"[{max_reached_q[0]:.2f}, {max_reached_q[1]:.2f}, {max_reached_q[2]:.2f}] | "
              f"({block_pos[0]:.2f}, {block_pos[2]:.2f})")

print("\n" + "=" * 80)
print("FINAL STATE ANALYSIS")
print("=" * 80)

final_q = data.qpos[7:13].copy()
final_block = data.xpos[1].copy()

print(f"\nTarget:     {target_q}")
print(f"Final q:    {final_q}")
print(f"Max reached: {max_reached_q}")
print(f"\nBlock position: ({final_block[0]:.3f}, {final_block[1]:.3f}, {final_block[2]:.3f})")
print(f"Block started:  (0.200, 0.000, 0.520)")
print(f"Block is approx {np.linalg.norm(final_q - target_q):.3f} rad away from target")

print("\n✓ Recommendation: Use max_reached as actual grasp target")
print(f"  Suggested GRASP = {list(np.round(max_reached_q, 3))}")

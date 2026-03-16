"""Check joint indices in model"""
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("assets/xarm_6dof.xml")
data = mujoco.MjData(model)

print("nq (# position DOFs):", model.nq)
print("nv (# velocity DOFs):", model.nv)
print("\nJoints and their qpos indices:")
for i in range(model.njnt):
    jnt = model.jnt(i)
    qpos_adr = model.jnt_qposadr[i]
    print(f"  {i}: {jnt.name:20s} qpos_adr={qpos_adr:2d} type={jnt.type}")

print("\nBodies and their qpos:")
for i in range(model.nbody):
    body = model.body(i)
    jnt_id = model.body_jntadr[i]
    if jnt_id >= 0:
        jnt = model.jnt(jnt_id)
        qpos_id = model.jnt_qposadr[jnt_id]
        print(f"  {i}: {body.name:20s} has joint {jnt.name:20s} at qpos[{qpos_id}]")

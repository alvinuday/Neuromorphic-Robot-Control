#!/usr/bin/env python3
import numpy as np
import mujoco
from arm_mpc import get_dynamics, build_mpc_qp, solve_mpc, N_ARM

model = mujoco.MjModel.from_xml_path('assets/xarm_6dof.xml')
data = mujoco.MjData(model)

q = np.zeros(N_ARM)
qdot = np.zeros(N_ARM)

print('Testing get_dynamics...')
try:
    M_inv, C, G = get_dynamics(model, data, q, qdot)
    print(f'✓ Works!')
    print(f'  M_inv: {M_inv.shape}')
    print(f'  C norm: {np.linalg.norm(C):.6f}')
    print(f'  G norm: {np.linalg.norm(G):.6f}')
except Exception as e:
    print(f'✗ Failed: {e}')
    exit(1)

print('\nTesting build_mpc_qp...')
try:
    q_ref = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.05])
    P, q_vec, A, l, u = build_mpc_qp(model, data, q, qdot, q_ref)
    print(f'✓ Works!')
    print(f'  P shape: {P.shape}')
    print(f'  q_vec norm: {np.linalg.norm(q_vec):.6f}')
except Exception as e:
    print(f'✗ Failed: {e}')
    exit(1)

print('\nTesting solve_mpc...')
try:
    tau = solve_mpc(model, data, q, qdot, q_ref)
    print(f'✓ Works!')
    print(f'  tau: {tau}')
    print(f'  tau norm: {np.linalg.norm(tau):.6f}')
except Exception as e:
    print(f'✗ Failed: {e}')
    exit(1)

print('\n✓ All tests passed!')

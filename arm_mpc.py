"""
xArm 6-DOF  ─  Single-Step MPC Controller
=========================================
Single-step receding horizon torque optimization.

Formulation:
  min_{tau}  (q_next - q_ref)^T Q (q_next - q_ref) + tau^T R tau
  s.t.       |tau_i| <= tau_max_i
  
where q_next = q + dt*qdot + dt^2 * M^-1(tau - C(q,qdot) - G(q))

Run
---
  python arm_mpc.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import osqp
from scipy import sparse
import csv
import pathlib

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
XML_PATH   = "assets/xarm_6dof.xml"
N_ARM      = 6
DT_SIM     = 0.002                # simulation timestep
N_HZ_MPC   = 50                   # MPC rate
MPC_EVERY  = int(1.0 / (N_HZ_MPC * DT_SIM))  # every 10 sim steps

# Cost weights
Q_POS      = 100.0                # position tracking
Q_VEL      = 1.0                  # velocity penalty
R_TORQUE   = 0.01                 # control effort

# Torque limits
U_MAX = np.array([20.0, 15.0, 15.0, 10.0, 8.0, 6.0])
U_MIN = -U_MAX

# Logger
LOG_PATH = pathlib.Path("mpc_log.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Reference trajectory
# ─────────────────────────────────────────────────────────────────────────────

# Pre-defined waypoints for pick-and-place (joint angles in radians)
# Red box is at (0.2, 0.0, 0.52) on the platform
WAYPOINTS = {
    'home':           np.array([0.0,   0.3,   0.0,   -0.3,   0.0,   0.0]),      # Default position
    'pick_approach':  np.array([0.0,  -0.8,   1.2,   -0.5,   0.0,   0.0]),      # Approach box from above
    'pick_grasp':     np.array([0.0,  -0.9,   1.3,   -0.6,   0.0,   0.0]),      # Grasp box
    'place_approach': np.array([1.0,  -0.8,   1.2,   -0.5,   0.0,   0.0]),      # Move to opposite side (x=1.0)
    'place_release':  np.array([1.0,  -1.0,   1.4,   -0.6,   0.0,   0.0]),      # Release at new location
}

# Timing phases (in seconds)
PHASE_DURATION = 2.0  # seconds per phase
TOTAL_CYCLE = 10 * PHASE_DURATION  # 20 seconds per full cycle

def interpolate_waypoints(wp_start, wp_end, t_phase):
    """Linear interpolation between waypoints over phase duration."""
    if t_phase >= PHASE_DURATION:
        return wp_end
    alpha = t_phase / PHASE_DURATION
    return (1 - alpha) * wp_start + alpha * wp_end

def reference_trajectory(t: float) -> np.ndarray:
    """
    Pick-and-place trajectory:
    1. Home → Pick approach → Grasp
    2. Grasp → Place approach → Release
    3. Release → Pick approach → Grasp  
    4. Grasp → Place approach → Release
    5. Release → Home
    Then loop
    """
    t_cycle = t % TOTAL_CYCLE  # Position in current cycle
    phase = int(t_cycle / PHASE_DURATION) % 10
    t_phase = t_cycle % PHASE_DURATION
    
    # State machine through phases
    if phase == 0:  # Home to pick approach
        q_ref = interpolate_waypoints(WAYPOINTS['home'], WAYPOINTS['pick_approach'], t_phase)
    elif phase == 1:  # Pick approach to grasp
        q_ref = interpolate_waypoints(WAYPOINTS['pick_approach'], WAYPOINTS['pick_grasp'], t_phase)
    elif phase == 2:  # Grasp hold (gripper closes, arm static)
        q_ref = WAYPOINTS['pick_grasp']
    elif phase == 3:  # Grasp to place approach
        q_ref = interpolate_waypoints(WAYPOINTS['pick_grasp'], WAYPOINTS['place_approach'], t_phase)
    elif phase == 4:  # Place approach to release
        q_ref = interpolate_waypoints(WAYPOINTS['place_approach'], WAYPOINTS['place_release'], t_phase)
    elif phase == 5:  # Release hold (gripper opens, arm static)
        q_ref = WAYPOINTS['place_release']
    elif phase == 6:  # Release back to pick approach
        q_ref = interpolate_waypoints(WAYPOINTS['place_release'], WAYPOINTS['pick_approach'], t_phase)
    elif phase == 7:  # Pick approach to grasp
        q_ref = interpolate_waypoints(WAYPOINTS['pick_approach'], WAYPOINTS['pick_grasp'], t_phase)
    elif phase == 8:  # Grasp hold
        q_ref = WAYPOINTS['pick_grasp']
    else:  # phase == 9: back to home
        q_ref = interpolate_waypoints(WAYPOINTS['pick_grasp'], WAYPOINTS['home'], t_phase)
    
    # Velocity reference: finite difference (2-step)
    if phase in [2, 5, 8]:  # Holding phases
        dq_ref = np.zeros(N_ARM)
    else:
        dt_fd = 0.01  # Finite difference step
        q_ref_next = reference_trajectory(t + dt_fd)[:N_ARM]
        dq_ref = (q_ref_next - q_ref) / dt_fd
    
    return np.concatenate([q_ref, dq_ref])


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics (from MuJoCo)
# ─────────────────────────────────────────────────────────────────────────────
def get_dynamics(model, data, q, qdot):
    """
    Compute M^-1, C, G at state (q, qdot).
    Returns:
      M_inv: [6,6] inverse inertia
      C:     [6]   Coriolis vector
      G:     [6]   gravity vector
    """
    nv = model.nv  # Full system DOFs (14 for xarm + base)
    
    # Set state
    data.qpos[:N_ARM] = q
    data.qvel[:N_ARM] = qdot
    mujoco.mj_forward(model, data)
    
    # Inertia matrix for arm - extract from full system
    # mj_mulM works on full nv, so we need to work in that space
    M_full = np.zeros((nv, nv))
    for i in range(N_ARM):
        e_i = np.zeros(nv)
        e_i[i] = 1.0
        res = np.zeros(nv)
        mujoco.mj_mulM(model, data, res, e_i)
        M_full[:, i] = res
    
    # Extract arm submatrix (first N_ARM rows and columns)
    M = M_full[:N_ARM, :N_ARM]
    M_inv = np.linalg.inv(M)
    
    # Gravity (zero velocity, zero torque)
    data.qvel[:N_ARM] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_step1(model, data)
    G = data.qacc[:N_ARM].copy()
    
    # Coriolis + gravity (with velocity, zero torque)
    data.qvel[:N_ARM] = qdot
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    mujoco.mj_step1(model, data)
    C_plus_G = data.qacc[:N_ARM].copy()
    
    C = C_plus_G - G
    
    return M_inv, C, G


# ─────────────────────────────────────────────────────────────────────────────
# MPC QP Builder (single-step)
# ─────────────────────────────────────────────────────────────────────────────
def build_mpc_qp(model, data, q, qdot, q_ref):
    """
    Build single-step MPC QP.
    
    Formulation:
      minimize  (q_next - q_ref)^T Q (q_next - q_ref) + tau^T R tau
      s.t.      |tau| <= u_max
    
    where q_next = q + dt*qdot + dt^2 * M_inv @ (tau - C - G)
    
    After substitution:
      q_next - q_ref = (q + dt*qdot - q_ref) + dt^2 * M_inv @ (tau - C - G)
                     = const_offset + dt^2 * M_inv @ tau - dt^2 * M_inv @ (C+G)
                     = const_offset_total + dt^2 * M_inv @ tau
    
    Cost in tau:
      0.5 * tau^T P tau + q_vec^T * tau  (OSQP form)
    
    where:
      P = 2 * (dt^4 * M_inv^T Q M_inv + R)
      q_vec = 2 * dt^2 * M_inv^T Q * offset
    """
    
    M_inv, C, G = get_dynamics(model, data, q, qdot)
    
    # Prediction: q_next = q + dt*qdot + dt^2 * M_inv @ (tau - C - G)
    # Error: e(tau) = q_next - q_ref
    #              = (q + dt*qdot - q_ref) + dt^2 * M_inv @ tau - dt^2 * M_inv @ (C+G)
    
    offset = q + DT_SIM * qdot - q_ref - (DT_SIM**2) * M_inv @ (C + G)
    A_dyn = (DT_SIM**2) * M_inv
    
    # Cost weighting
    Q_weight = np.diag([Q_POS] * N_ARM)
    R_weight = R_TORQUE * np.eye(N_ARM)
    
    # QP matrices for OSQP: minimize 0.5 * tau^T P tau + q^T * tau
    P = A_dyn.T @ Q_weight @ A_dyn + R_weight
    P = 2.0 * P  # Scale for OSQP
    q_vec = 2.0 * A_dyn.T @ Q_weight @ offset
    
    # Symmetrize P
    P = 0.5 * (P + P.T)
    
    # Box constraints (sparse)
    A_box = sparse.eye(N_ARM)
    l_box = U_MIN
    u_box = U_MAX
    
    return sparse.csc_matrix(P), q_vec, A_box, l_box, u_box


# ─────────────────────────────────────────────────────────────────────────────
# OSQP Solver
# ─────────────────────────────────────────────────────────────────────────────
_solve_count = 0

def solve_mpc(model, data, q, qdot, q_ref):
    """Solve single-step MPC, return tau [6] or zeros on failure."""
    global _solve_count
    
    try:
        P, q_vec, A, l, u = build_mpc_qp(model, data, q, qdot, q_ref)
        
        # Create and solve
        prob = osqp.OSQP()
        prob.setup(
            P=P,
            q=q_vec,
            A=A,
            l=l,
            u=u,
            verbose=False,
            eps_abs=1e-2,
            eps_rel=1e-2,
            max_iter=500,
            polish=False,
        )
        
        result = prob.solve()
        
        if result.info.status not in ('solved', 'solved_inaccurate'):
            _solve_count += 1
            if _solve_count % 100 == 0:
                print(f"[MPC] Solver: {result.info.status}")
            return np.zeros(N_ARM)
        
        tau = result.x
        _solve_count += 1
        if _solve_count % 500 == 0:
            print(f"[MPC] ✓ {result.info.iter} iters, ||tau||={np.linalg.norm(tau):.4f}")
        
        return tau
        
    except Exception as e:
        _solve_count += 1
        if _solve_count % 100 == 0:
            print(f"[MPC] Exception: {e}")
        return np.zeros(N_ARM)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def init_logger():
    f = open(LOG_PATH, 'w', newline='')
    w = csv.writer(f)
    w.writerow(
        ['t_sim'] +
        [f'q{i+1}' for i in range(N_ARM)] +
        [f'q{i+1}_ref' for i in range(N_ARM)] +
        [f'tau{i+1}' for i in range(N_ARM)] +
        ['tracking_error']
    )
    return f, w

def log_row(f, w, t, q, q_ref, tau):
    error = np.linalg.norm(q - q_ref)
    w.writerow(
        [f'{t:.4f}'] +
        [f'{v:.6f}' for v in q] +
        [f'{v:.6f}' for v in q_ref] +
        [f'{v:.6f}' for v in tau] +
        [f'{error:.6f}']
    )
    if int(t * 100) % 5 == 0:
        f.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM] = 0.0
    data.qvel[:N_ARM] = 0.0
    
    # Initialize red_block position using proper body/joint lookup
    # Find red_block body and its free joint
    try:
        red_block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'red_block')
        red_block_jnt_id = model.body_jntadr[red_block_body_id]
        red_block_qpos_adr = model.jnt_qposadr[red_block_jnt_id]
        
        # Set position [x, y, z] and quaternion [qw, qx, qy, qz]
        data.qpos[red_block_qpos_adr:red_block_qpos_adr+3] = np.array([0.2, 0.0, 0.52])
        data.qpos[red_block_qpos_adr+3:red_block_qpos_adr+7] = np.array([1.0, 0.0, 0.0, 0.0])
        print(f"[Init] Red block at qpos[{red_block_qpos_adr}:{red_block_qpos_adr+7}]")
    except Exception as e:
        print(f"[Init] Could not set red_block position: {e}")
    
    mujoco.mj_forward(model, data)
    
    print(f"\n┌─ xArm MPC ──────────────────────────")
    print(f"│ Model: nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"│ Horizon: 1 step ({DT_SIM*1000:.1f}ms)")
    print(f"│ Rate: {N_HZ_MPC} Hz")
    print(f"│ Q_pos={Q_POS:.0f}  Q_vel={Q_VEL:.1f}  R={R_TORQUE:.3f}")
    print(f"└─────────────────────────────────────\n")
    
    log_f, log_w = init_logger()
    
    u_current = np.zeros(N_ARM)
    t_sim = 0.0
    step = 0
    times = []
    
    print("Starting viewer... (close to stop)\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Solve MPC
            if step % MPC_EVERY == 0:
                q = data.qpos[:N_ARM].copy()
                qdot = data.qvel[:N_ARM].copy()
                x_ref = reference_trajectory(t_sim)
                
                t0 = time.perf_counter()
                u_current = solve_mpc(model, data, q, qdot, x_ref[:N_ARM])
                times.append(time.perf_counter() - t0)
                
                log_row(log_f, log_w, t_sim, q, x_ref[:N_ARM], u_current)
            
            # Apply control
            data.ctrl[:N_ARM] = u_current
            data.ctrl[N_ARM:] = 0.0
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            t_sim += DT_SIM
            step += 1
    
    log_f.close()
    
    if times:
        times = np.array(times) * 1000
        print(f"\n── Solve Times ──────────")
        print(f" Mean: {times.mean():.2f} ms")
        print(f" Max:  {times.max():.2f} ms")
        print(f" Calls: {len(times)}")
        print(f" Budget: {MPC_EVERY * DT_SIM * 1000:.1f} ms")
    print(f" Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
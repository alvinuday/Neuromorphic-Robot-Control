"""
Franka Panda 7-DOF - Multi-Step MPC Controller with Pinocchio Integration
===========================================================================
Receding-horizon optimization over arm position-actuator commands.
Uses OSQP (horizon=20) + Pinocchio for kinematics support.

Run
---
    .venv/bin/python mjpc/arm_mpc.py
    .venv/bin/python mjpc/arm_mpc.py --headless --sim-time 20
    .venv/bin/python mjpc/arm_mpc.py --headless --mode openloop --openloop-speed 2.0 --sim-dt 0.005 --log-every 10
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
import osqp
from scipy import sparse
import csv
import pathlib
from pathlib import Path

try:
    from .franka_motion_planning import FrankaMotionPlanning  # Pinocchio-integrated planner
except ImportError:
    from franka_motion_planning import FrankaMotionPlanning

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
SCENE_XML_PATH = str(ROOT_DIR / "franka_panda/mjx_single_cube.xml")
PIN_MODEL_PATH = str(ROOT_DIR / "franka_panda/mjx_panda.xml")

N_ARM = 7
N_GRIPPER = 2
N_TOTAL = N_ARM + N_GRIPPER

HORIZON = 20
DT_SIM = 0.002
N_HZ_MPC = 50
MPC_EVERY = int(1.0 / (N_HZ_MPC * DT_SIM))
DT_CTRL = MPC_EVERY * DT_SIM

# Position-command MPC weights
Q_TRACK = 220.0
R_CMD = 0.8
R_DELTA = 35.0

# First-order closed-loop approximation for position actuators.
SERVO_ALPHA = 0.28

# Logger
LOG_PATH = pathlib.Path("arm_mpc_log.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Motion Planning - Franka-Specific with Pinocchio
# ─────────────────────────────────────────────────────────────────────────────

MOTION_PLANNER = FrankaMotionPlanning(
    urdf_path=PIN_MODEL_PATH,
    move_time=3.0,
    hold_time=1.0,
    lift_height=0.10,
)

def reference_trajectory_smooth(t_sim):
    """
    Get reference trajectory using Franka motion planner with Pinocchio.
    Handles 7-DOF arm + 2-DOF gripper.
    
    Args:
        t_sim: simulation time (seconds)
    
    Returns:
        (q_ref, dq_ref): reference joint angles and velocities [9]
    """
    q_ref, dq_ref = MOTION_PLANNER.get_reference(t_sim)
    return q_ref, dq_ref

class PositionMPCController:
    """OSQP MPC over joint position actuator commands (7 arm joints)."""

    def __init__(self, model):
        self.model = model
        self.n = N_ARM
        self.h = HORIZON
        self.nx = self.n * self.h

        self.u_min = model.actuator_ctrlrange[:N_ARM, 0].astype(float)
        self.u_max = model.actuator_ctrlrange[:N_ARM, 1].astype(float)

        a = 1.0 - SERVO_ALPHA
        b = SERVO_ALPHA

        eye_n = np.eye(self.n)
        s = np.zeros((self.nx, self.nx))
        c_map = np.zeros((self.nx, self.n))
        for k in range(self.h):
            row = slice(k * self.n, (k + 1) * self.n)
            c_map[row, :] = (a ** (k + 1)) * eye_n
            for j in range(k + 1):
                col = slice(j * self.n, (j + 1) * self.n)
                s[row, col] = (b * (a ** (k - j))) * eye_n

        self.s = s
        self.c_map = c_map

        q_big = np.eye(self.nx) * Q_TRACK
        r_big = np.eye(self.nx) * R_CMD

        e = np.zeros((self.nx, self.nx))
        for k in range(self.h):
            row = slice(k * self.n, (k + 1) * self.n)
            e[row, row] = eye_n
            if k > 0:
                prev = slice((k - 1) * self.n, k * self.n)
                e[row, prev] = -eye_n
        w_big = np.eye(self.nx) * R_DELTA

        p = 2.0 * (s.T @ q_big @ s + r_big + e.T @ w_big @ e)
        p = 0.5 * (p + p.T)

        self.q_big = q_big
        self.r_big = r_big
        self.e = e
        self.w_big = w_big
        self.p = sparse.csc_matrix(p)
        self.a = sparse.eye(self.nx, format="csc")
        self.l = np.tile(self.u_min, self.h)
        self.u = np.tile(self.u_max, self.h)

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.p,
            q=np.zeros(self.nx),
            A=self.a,
            l=self.l,
            u=self.u,
            verbose=False,
            eps_abs=5e-3,
            eps_rel=5e-3,
            max_iter=300,
            polish=False,
            warm_start=True,
        )

    def solve(self, q_arm, q_ref_arm, u_prev):
        r = np.tile(q_ref_arm, self.h)
        c = self.c_map @ q_arm - r
        d = np.zeros(self.nx)
        d[:self.n] = -u_prev

        q_vec = 2.0 * (
            self.s.T @ self.q_big @ c
            - self.r_big @ r
            + self.e.T @ self.w_big @ d
        )

        self.prob.update(q=q_vec, l=self.l, u=self.u)
        result = self.prob.solve()
        if result.info.status not in ("solved", "solved_inaccurate"):
            return np.clip(q_ref_arm, self.u_min, self.u_max)

        u_cmd = np.array(result.x[:self.n], dtype=float)
        return np.clip(u_cmd, self.u_min, self.u_max)


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
        ['gripper_cmd', 'tracking_error', 'ee_x', 'ee_y', 'ee_z']
    )
    return f, w

def log_row(f, w, t, q_arm, q_ref_arm, tau, gripper_ref, ee_pos):
    error = np.linalg.norm(q_arm - q_ref_arm)
    w.writerow(
        [f'{t:.4f}'] +
        [f'{v:.6f}' for v in q_arm] +
        [f'{v:.6f}' for v in q_ref_arm] +
        [f'{v:.6f}' for v in tau] +
        [f'{gripper_ref:.4f}', f'{error:.6f}'] +
        [f'{x:.4f}' for x in (ee_pos if ee_pos is not None else [0, 0, 0])]
    )
    if int(t * 100) % 5 == 0:
        f.flush()


def _load_keyframe_qpos(model, key_name):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        return None
    return model.key_qpos.reshape(model.nkey, model.nq)[key_id].copy()


def _configure_motion_targets(model, data):
    home_full = _load_keyframe_qpos(model, "home")
    pickup_full = _load_keyframe_qpos(model, "pickup")
    pickup1_full = _load_keyframe_qpos(model, "pickup1")

    home = home_full[:N_ARM].copy() if home_full is not None else MOTION_PLANNER.home.copy()
    pickup = pickup_full[:N_ARM].copy() if pickup_full is not None else MOTION_PLANNER.grasp_target.copy()

    if pickup1_full is not None:
        lift = pickup1_full[:N_ARM].copy()
    else:
        lift = pickup.copy()
        lift[3] = np.clip(lift[3] + 0.20, model.jnt_range[3, 0], model.jnt_range[3, 1])

    place = lift.copy()
    place[0] = np.clip(place[0] - 0.35, model.jnt_range[0, 0], model.jnt_range[0, 1])
    place[1] = np.clip(place[1] - 0.20, model.jnt_range[1, 0], model.jnt_range[1, 1])

    # Use Pinocchio IK to adapt waypoints to the actual box pose in mjx_single_cube.xml.
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if (
        box_body_id >= 0
        and hand_body_id >= 0
        and pickup_full is not None
        and MOTION_PLANNER.pin_model is not None
    ):
        qpos_saved = data.qpos.copy()
        qvel_saved = data.qvel.copy()
        ctrl_saved = data.ctrl.copy()

        # Calibrate hand-to-box offsets from provided keyframes.
        data.qpos[:] = pickup_full
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        grasp_offset = data.xpos[hand_body_id].copy() - data.xpos[box_body_id].copy()

        lift_offset = grasp_offset + np.array([0.0, 0.0, 0.10])
        if pickup1_full is not None:
            data.qpos[:] = pickup1_full
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            lift_offset = data.xpos[hand_body_id].copy() - data.xpos[box_body_id].copy()

        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        data.ctrl[:] = ctrl_saved
        mujoco.mj_forward(model, data)

        box_pos = data.xpos[box_body_id].copy()
        grasp_pos = box_pos + grasp_offset
        lift_pos = box_pos + lift_offset
        place_box_pos = box_pos + np.array([-0.18, -0.16, 0.0])
        place_pos = place_box_pos + lift_offset

        q_grasp, ok_grasp = MOTION_PLANNER.solve_ik_position(grasp_pos, pickup)
        q_lift, ok_lift = MOTION_PLANNER.solve_ik_position(lift_pos, q_grasp)
        q_place, ok_place = MOTION_PLANNER.solve_ik_position(place_pos, q_lift)

        if ok_grasp and ok_lift and ok_place:
            pickup = q_grasp
            lift = q_lift
            place = q_place
            print("[MPC] Using calibrated Pinocchio IK waypoints from scene box pose")
        else:
            print("[MPC] IK waypoint solve incomplete; using keyframe-derived targets")

    MOTION_PLANNER.home = home
    MOTION_PLANNER.grasp_target = pickup
    MOTION_PLANNER.lift_target = lift
    MOTION_PLANNER.place_target = place

    return home


def _preplan_openloop_trajectory(sim_time, dt_sim):
    """Precompute the full reference sequence once for fast open-loop execution."""
    steps = int(sim_time / dt_sim)
    q_plan = np.zeros((steps, N_TOTAL), dtype=float)

    for i in range(steps):
        t_sim = i * dt_sim
        q_ref, _ = reference_trajectory_smooth(t_sim)
        q_plan[i, :] = q_ref

    # Sample FK points for diagnostics so we can validate the preplanned path shape.
    if MOTION_PLANNER.pin_model is not None and steps > 0:
        sample_count = min(12, steps)
        for idx in np.linspace(0, steps - 1, sample_count, dtype=int):
            MOTION_PLANNER.forward_kinematics_pinocchio(q_plan[idx, :N_ARM])

    return q_plan


def run_simulation(
    use_viewer=True,
    sim_time=20.0,
    mode="mpc",
    sim_dt=None,
    openloop_speed=1.0,
    log_every=1,
):
    if mode not in ("mpc", "openloop"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'mpc' or 'openloop'.")
    if openloop_speed <= 0.0:
        raise ValueError("openloop_speed must be > 0")
    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    if sim_dt is not None:
        model.opt.timestep = float(sim_dt)

    data = mujoco.MjData(model)
    dt_sim = float(model.opt.timestep)
    mpc_every = max(1, int(round(1.0 / (N_HZ_MPC * dt_sim))))
    dt_ctrl = mpc_every * dt_sim

    mpc = PositionMPCController(model) if mode == "mpc" else None

    mujoco.mj_resetData(model, data)
    home = _configure_motion_targets(model, data)

    q_openloop = None
    if mode == "openloop":
        q_openloop = _preplan_openloop_trajectory(sim_time, dt_sim)
        print(f"[OpenLoop] Preplanned {len(q_openloop)} trajectory steps")

    data.qpos[0:N_ARM] = home
    data.qpos[N_ARM:N_TOTAL] = [0.04, 0.04]
    data.qvel[:] = 0.0
    data.ctrl[:N_ARM] = home
    mujoco.mj_forward(model, data)

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if gripper_act_id < 0:
        gripper_act_id = 7

    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")

    print("\nFranka Panda Controller")
    print(f"Scene: {Path(SCENE_XML_PATH).name}")
    print(f"Pinocchio model: {Path(PIN_MODEL_PATH).name}")
    print(f"Mode: {mode}")
    print(f"MPC period: {dt_ctrl*1000:.1f} ms | sim dt: {dt_sim*1000:.1f} ms")
    if mode == "openloop":
        print(f"Open-loop speed: {openloop_speed:.2f}x")
    if log_every > 1:
        print(f"Log decimation: every {log_every} steps")

    log_f, log_w = init_logger()
    u_current = home.copy()
    t_sim = 0.0
    step = 0
    solve_times = []
    box_start = None
    wall_t0 = time.perf_counter()

    if box_body_id >= 0:
        box_start = data.xpos[box_body_id].copy()

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                q_now = data.qpos.copy()
                q_ref, _ = reference_trajectory_smooth(t_sim)

                if mode == "mpc":
                    if step % mpc_every == 0:
                        t0 = time.perf_counter()
                        u_current = mpc.solve(q_now[:N_ARM], q_ref[:N_ARM], u_current)
                        solve_times.append(time.perf_counter() - t0)
                else:
                    plan_idx = min(int(step * openloop_speed), len(q_openloop) - 1)
                    if plan_idx < len(q_openloop):
                        u_current = q_openloop[plan_idx, :N_ARM].copy()
                        q_ref = q_openloop[plan_idx, :].copy()
                    else:
                        u_current = q_openloop[-1, :N_ARM].copy()
                        q_ref = q_openloop[-1, :].copy()

                data.ctrl[:N_ARM] = u_current
                data.ctrl[gripper_act_id] = float(np.clip(q_ref[N_ARM], 0.0, 0.04))

                ee_pos = data.xpos[ee_body_id].copy() if ee_body_id >= 0 else np.zeros(3)
                q_now = data.qpos.copy()
                if step % log_every == 0:
                    log_row(log_f, log_w, t_sim, q_now[:N_ARM], q_ref[:N_ARM], u_current, q_ref[N_ARM], ee_pos)

                mujoco.mj_step(model, data)
                viewer.sync()
                t_sim += dt_sim
                step += 1
    else:
        steps = int(sim_time / dt_sim)
        for _ in range(steps):
            q_now = data.qpos.copy()
            q_ref, _ = reference_trajectory_smooth(t_sim)

            if mode == "mpc":
                if step % mpc_every == 0:
                    t0 = time.perf_counter()
                    u_current = mpc.solve(q_now[:N_ARM], q_ref[:N_ARM], u_current)
                    solve_times.append(time.perf_counter() - t0)
            else:
                plan_idx = min(int(step * openloop_speed), len(q_openloop) - 1)
                u_current = q_openloop[plan_idx, :N_ARM].copy()
                q_ref = q_openloop[plan_idx, :].copy()

            data.ctrl[:N_ARM] = u_current
            data.ctrl[gripper_act_id] = float(np.clip(q_ref[N_ARM], 0.0, 0.04))

            ee_pos = data.xpos[ee_body_id].copy() if ee_body_id >= 0 else np.zeros(3)
            q_now = data.qpos.copy()
            if step % log_every == 0:
                log_row(log_f, log_w, t_sim, q_now[:N_ARM], q_ref[:N_ARM], u_current, q_ref[N_ARM], ee_pos)

            mujoco.mj_step(model, data)
            t_sim += dt_sim
            step += 1

    log_f.close()

    wall_dt = time.perf_counter() - wall_t0
    if wall_dt > 0.0:
        print(f"Wall time: {wall_dt:.2f} s | sim speed: {sim_time / wall_dt:.1f}x")

    if mode == "mpc" and solve_times:
        ms = np.array(solve_times) * 1000.0
        print("\nSolve Times")
        print(f"Mean: {ms.mean():.2f} ms")
        print(f"Max:  {ms.max():.2f} ms")
        print(f"Budget: {dt_ctrl*1000:.1f} ms")
        print(f"Calls: {len(ms)}")

    if box_body_id >= 0 and box_start is not None:
        box_end = data.xpos[box_body_id].copy()
        disp = np.linalg.norm(box_end - box_start)
        print(f"Box displacement: {disp:.4f} m")
        print(f"Box start: [{box_start[0]:.3f}, {box_start[1]:.3f}, {box_start[2]:.3f}]")
        print(f"Box end:   [{box_end[0]:.3f}, {box_end[1]:.3f}, {box_end[2]:.3f}]")

    print(f"Log: {LOG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Franka OSQP MPC controller")
    parser.add_argument("--headless", action="store_true", help="run without viewer")
    parser.add_argument("--sim-time", type=float, default=20.0, help="headless simulation duration (seconds)")
    parser.add_argument("--mode", choices=["mpc", "openloop"], default="mpc", help="control mode")
    parser.add_argument("--sim-dt", type=float, default=None, help="override MuJoCo timestep (seconds)")
    parser.add_argument("--openloop-speed", type=float, default=1.0, help="trajectory playback speed for open-loop mode")
    parser.add_argument("--log-every", type=int, default=1, help="log one row every N simulation steps")
    args = parser.parse_args()

    run_simulation(
        use_viewer=not args.headless,
        sim_time=args.sim_time,
        mode=args.mode,
        sim_dt=args.sim_dt,
        openloop_speed=args.openloop_speed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
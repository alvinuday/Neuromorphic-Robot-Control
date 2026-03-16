"""
Franka Panda 7-DOF - Multi-Step MPC Controller with Pinocchio Integration
===========================================================================
Receding-horizon optimization over arm position-actuator commands.
Uses OSQP (horizon=20) + Pinocchio for kinematics support.

Run
---
    .venv/bin/python mjpc/arm_mpc.py
    .venv/bin/python mjpc/arm_mpc.py --headless --sim-time 20
    .venv/bin/python mjpc/arm_mpc.py --headless --mode openloop --preset balanced
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

# End-of-place target relative to the current box spawn pose.
PLACE_BOX_OFFSET = np.array([-0.1, -0.25, 0.0], dtype=float)
# Force grasp at box center height (box-frame z = 0) to avoid corner pinches.
GRASP_REL_Z_TARGET = 0.0
# Open-loop retreat shaping after release to avoid clipping the box on the way out.
POST_RELEASE_LIFT = np.array([0.0, 0.0, 0.10], dtype=float)
POST_RELEASE_BACKOFF = np.array([0.10, 0.00, 0.02], dtype=float)

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


def _get_grasp_anchor_pose(
    model,
    data,
    gripper_site_id=None,
    hand_body_id=None,
    left_pad_geom_id=None,
    right_pad_geom_id=None,
):
    """Return grasp-anchor world pose, preferring finger-pad midpoint over nominal site/body."""
    if left_pad_geom_id is None:
        left_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    if right_pad_geom_id is None:
        right_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")

    if left_pad_geom_id >= 0 and right_pad_geom_id >= 0:
        pos_l = data.geom_xpos[left_pad_geom_id].copy()
        pos_r = data.geom_xpos[right_pad_geom_id].copy()
        pos = 0.5 * (pos_l + pos_r)

        if gripper_site_id is None:
            gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        if gripper_site_id >= 0:
            rot = data.site_xmat[gripper_site_id].reshape(3, 3).copy()
        else:
            rot = data.geom_xmat[left_pad_geom_id].reshape(3, 3).copy()

        return pos, rot, "geom:pad_midpoint"

    if gripper_site_id is None:
        gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    if gripper_site_id >= 0:
        pos = data.site_xpos[gripper_site_id].copy()
        rot = data.site_xmat[gripper_site_id].reshape(3, 3).copy()
        return pos, rot, "site:gripper"

    if hand_body_id is None:
        hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_body_id >= 0:
        pos = data.xpos[hand_body_id].copy()
        rot = data.xmat[hand_body_id].reshape(3, 3).copy()
        return pos, rot, "body:hand"

    return None, None, None


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
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    right_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
    if (
        box_body_id >= 0
        and (gripper_site_id >= 0 or hand_body_id >= 0)
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
        hand_p_pick, hand_R_pick, anchor_name = _get_grasp_anchor_pose(
            model, data, gripper_site_id, hand_body_id, left_pad_geom_id, right_pad_geom_id
        )
        box_p_pick = data.xpos[box_body_id].copy()
        box_R_pick = data.xmat[box_body_id].reshape(3, 3).copy()

        grasp_rel_pos = box_R_pick.T @ (hand_p_pick - box_p_pick)
        grasp_rel_rot = box_R_pick.T @ hand_R_pick

        lift_rel_pos = grasp_rel_pos.copy()
        lift_rel_rot = grasp_rel_rot.copy()
        if pickup1_full is not None:
            data.qpos[:] = pickup1_full
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            hand_p_lift, hand_R_lift, _ = _get_grasp_anchor_pose(
                model, data, gripper_site_id, hand_body_id, left_pad_geom_id, right_pad_geom_id
            )
            box_p_lift = data.xpos[box_body_id].copy()
            box_R_lift = data.xmat[box_body_id].reshape(3, 3).copy()

            lift_rel_pos = box_R_lift.T @ (hand_p_lift - box_p_lift)
            lift_rel_rot = box_R_lift.T @ hand_R_lift

        # Keep calibrated vertical standoff/orientation but force centered pinch in box XY.
        # Old keyframes can encode side-biased offsets that cause edge-pushing contacts.
        grasp_rel_pos[:2] = 0.0
        grasp_rel_pos[2] = GRASP_REL_Z_TARGET
        lift_rel_pos[:2] = 0.0

        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        data.ctrl[:] = ctrl_saved
        mujoco.mj_forward(model, data)

        box_pos = data.xpos[box_body_id].copy()
        box_rot = data.xmat[box_body_id].reshape(3, 3).copy()
        grasp_pos = box_pos + box_rot @ grasp_rel_pos
        lift_pos = box_pos + box_rot @ lift_rel_pos
        place_box_pos = box_pos + PLACE_BOX_OFFSET
        place_pos = place_box_pos + box_rot @ lift_rel_pos

        q_grasp, ok_grasp = MOTION_PLANNER.solve_ik_position(grasp_pos, pickup)
        q_lift, ok_lift = MOTION_PLANNER.solve_ik_position(lift_pos, q_grasp)
        q_place, ok_place = MOTION_PLANNER.solve_ik_position(place_pos, q_lift)

        if ok_grasp and ok_lift and ok_place:
            pickup = q_grasp
            lift = q_lift
            place = q_place
            print(f"[MPC] Using calibrated IK waypoints from scene box pose ({anchor_name})")
            print(f"[MPC] Centered grasp rel pos: [{grasp_rel_pos[0]:.4f}, {grasp_rel_pos[1]:.4f}, {grasp_rel_pos[2]:.4f}]")
            print(f"[MPC] Centered lift  rel pos: [{lift_rel_pos[0]:.4f}, {lift_rel_pos[1]:.4f}, {lift_rel_pos[2]:.4f}]")
        else:
            print("[MPC] IK waypoint solve incomplete; using keyframe-derived targets")

    MOTION_PLANNER.home = home
    MOTION_PLANNER.grasp_target = pickup
    MOTION_PLANNER.lift_target = lift
    MOTION_PLANNER.place_target = place
    MOTION_PLANNER.grasp_rel_pos = grasp_rel_pos if 'grasp_rel_pos' in locals() else np.array([0.0, 0.0, 0.12])
    MOTION_PLANNER.grasp_rel_rot = grasp_rel_rot if 'grasp_rel_rot' in locals() else np.eye(3)
    MOTION_PLANNER.lift_rel_pos = lift_rel_pos if 'lift_rel_pos' in locals() else np.array([0.0, 0.0, 0.22])
    MOTION_PLANNER.lift_rel_rot = lift_rel_rot if 'lift_rel_rot' in locals() else np.eye(3)

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


def _quintic_segment(q0, q1, t, duration):
    """Evaluate joint-space quintic with zero velocity/acceleration endpoints."""
    if duration <= 1e-9:
        return q1.copy(), np.zeros_like(q0)

    tau = np.clip(t / duration, 0.0, 1.0)
    alpha = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    d_alpha = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration

    q = q0 + alpha * (q1 - q0)
    dq = d_alpha * (q1 - q0)
    return q, dq


def _build_cartesian_waypoints(box_pos, place_pos):
    """Create an explicit pick/place Cartesian path with a deliberate Z-first approach."""
    z_hi = 0.20
    z_mid = 0.08
    z_touch = 0.010

    return [
        (box_pos + np.array([0.0, 0.0, z_hi]), 0.04, 1.2, "pre_grasp"),
        (box_pos + np.array([0.0, 0.0, z_touch]), 0.04, 0.9, "grasp_descend"),
        (box_pos + np.array([0.0, 0.0, z_touch]), 0.001, 0.5, "close_gripper"),
        (box_pos + np.array([0.0, 0.0, z_hi]), 0.001, 1.0, "lift"),
        (place_pos + np.array([0.0, 0.0, z_hi]), 0.001, 1.6, "transfer"),
        (place_pos + np.array([0.0, 0.0, z_mid]), 0.001, 0.8, "pre_place"),
        (place_pos + np.array([0.0, 0.0, z_touch]), 0.001, 0.8, "place_descend"),
        (place_pos + np.array([0.0, 0.0, z_touch]), 0.04, 0.5, "open_gripper"),
        (place_pos + np.array([0.0, 0.0, z_hi]), 0.04, 0.8, "retreat"),
    ]


def _solve_pose_waypoints(home_q, pose_waypoints):
    """Solve position+orientation waypoint sequence with robust task-phase fallbacks."""
    q_waypoints = [home_q.copy()]
    grip_waypoints = [0.04]
    durations = []

    q_prev = home_q.copy()
    ik_success = 0
    for target_pos, target_rot, grip_cmd, duration, phase_name in pose_waypoints:
        q_next, ok = MOTION_PLANNER.solve_ik_pose(target_pos, target_rot, q_prev)
        if not ok:
            if phase_name in ("grasp_descend", "close_gripper"):
                q_next = MOTION_PLANNER.grasp_target.copy()
            elif phase_name == "lift":
                q_next = MOTION_PLANNER.lift_target.copy()
            elif phase_name in ("transfer", "pre_place", "place_descend", "open_gripper", "retreat"):
                q_next = MOTION_PLANNER.place_target.copy()
            elif phase_name == "pre_grasp":
                q_next = 0.5 * (home_q + MOTION_PLANNER.grasp_target)
            else:
                q_next = q_prev.copy()
        else:
            ik_success += 1

        q_waypoints.append(q_next.copy())
        grip_waypoints.append(float(grip_cmd))
        durations.append(float(duration))
        q_prev = q_next.copy()

    q_waypoints.append(home_q.copy())
    grip_waypoints.append(0.04)
    durations.append(1.2)
    print(f"[Planner] Pose IK solved {ik_success}/{len(pose_waypoints)} waypoints")
    return q_waypoints, grip_waypoints, durations


def _solve_joint_waypoints(home_q, cart_waypoints):
    """Solve IK for waypoint sequence by continuity (closest from previous)."""
    q_waypoints = [home_q.copy()]
    grip_waypoints = [0.04]
    durations = []

    q_prev = home_q.copy()
    ik_success = 0
    for target_pos, grip_cmd, duration, phase_name in cart_waypoints:
        q_next, ok = MOTION_PLANNER.solve_ik_position(target_pos, q_prev)
        if not ok:
            # Robust fallback: keep explicit phase structure while using calibrated joint anchors.
            if phase_name in ("grasp_descend", "close_gripper"):
                q_next = MOTION_PLANNER.grasp_target.copy()
            elif phase_name == "lift":
                q_next = MOTION_PLANNER.lift_target.copy()
            elif phase_name in ("transfer", "pre_place", "place_descend", "open_gripper", "retreat"):
                q_next = MOTION_PLANNER.place_target.copy()
            elif phase_name == "pre_grasp":
                q_next = 0.5 * (home_q + MOTION_PLANNER.grasp_target)
            else:
                q_next = q_prev.copy()
        else:
            ik_success += 1
        q_waypoints.append(q_next.copy())
        grip_waypoints.append(float(grip_cmd))
        durations.append(float(duration))
        q_prev = q_next.copy()

    q_waypoints.append(home_q.copy())
    grip_waypoints.append(0.04)
    durations.append(1.2)

    print(f"[Planner] IK solved {ik_success}/{len(cart_waypoints)} Cartesian waypoints")
    return q_waypoints, grip_waypoints, durations


def _preplan_reference_trajectory(sim_time, dt_sim, data, home):
    """Plan a full-cycle trajectory offline and sample it into q/dq references."""
    box_body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "box")
    if box_body_id < 0:
        q_plan = _preplan_openloop_trajectory(sim_time, dt_sim)
        dq_plan = np.zeros_like(q_plan)
        return q_plan, dq_plan

    box_pos = data.xpos[box_body_id].copy()
    box_rot = data.xmat[box_body_id].reshape(3, 3).copy()
    place_box_pos = box_pos + PLACE_BOX_OFFSET
    place_box_rot = box_rot.copy()

    # Build grasp/lift/place poses from calibrated hand-to-box relative transforms.
    grasp_rel_pos = np.array(getattr(MOTION_PLANNER, "grasp_rel_pos", np.array([0.0, 0.0, 0.12])), dtype=float)
    grasp_rel_rot = np.array(getattr(MOTION_PLANNER, "grasp_rel_rot", np.eye(3)), dtype=float)
    lift_rel_pos = np.array(getattr(MOTION_PLANNER, "lift_rel_pos", np.array([0.0, 0.0, 0.22])), dtype=float)
    lift_rel_rot = np.array(getattr(MOTION_PLANNER, "lift_rel_rot", grasp_rel_rot), dtype=float)

    grasp_pos = box_pos + box_rot @ grasp_rel_pos
    grasp_rot = box_rot @ grasp_rel_rot
    lift_pos = box_pos + box_rot @ lift_rel_pos
    lift_rot = box_rot @ lift_rel_rot
    place_grasp_pos = place_box_pos + place_box_rot @ grasp_rel_pos
    place_grasp_rot = place_box_rot @ grasp_rel_rot
    place_lift_pos = place_box_pos + place_box_rot @ lift_rel_pos
    place_lift_rot = place_box_rot @ lift_rel_rot
    post_release_lift_pos = place_grasp_pos + POST_RELEASE_LIFT
    retreat_clear_pos = place_lift_pos + POST_RELEASE_BACKOFF

    z_up = np.array([0.0, 0.0, 0.12])
    z_pre_touch = np.array([0.0, 0.0, 0.04])
    pose_waypoints = [
        (grasp_pos + z_up, grasp_rot, 0.04, 1.4, "pre_grasp"),
        (grasp_pos + z_pre_touch, grasp_rot, 0.04, 0.8, "pre_touch"),
        (grasp_pos, grasp_rot, 0.04, 1.0, "grasp_descend"),
        (grasp_pos, grasp_rot, 0.001, 1.6, "close_gripper"),
        (grasp_pos, grasp_rot, 0.001, 0.8, "grasp_settle"),
        (lift_pos, lift_rot, 0.001, 1.3, "lift"),
        (place_lift_pos, place_lift_rot, 0.001, 1.8, "transfer"),
        (place_grasp_pos + z_up, place_grasp_rot, 0.001, 1.0, "pre_place"),
        (place_grasp_pos + z_pre_touch, place_grasp_rot, 0.001, 0.8, "pre_place_touch"),
        (place_grasp_pos, place_grasp_rot, 0.001, 1.0, "place_descend"),
        (place_grasp_pos, place_grasp_rot, 0.04, 0.7, "open_gripper"),
        (post_release_lift_pos, place_grasp_rot, 0.04, 0.9, "post_release_lift"),
        (retreat_clear_pos, place_lift_rot, 0.04, 1.0, "retreat_clear"),
    ]
    q_waypoints, grip_waypoints, durations = _solve_pose_waypoints(home, pose_waypoints)

    cycle_time = np.sum(durations)
    cycle_steps = max(1, int(round(cycle_time / dt_sim)))
    q_cycle = np.zeros((cycle_steps, N_TOTAL), dtype=float)
    dq_cycle = np.zeros((cycle_steps, N_TOTAL), dtype=float)

    t_cursor = 0.0
    for seg_idx, duration in enumerate(durations):
        q0 = q_waypoints[seg_idx]
        q1 = q_waypoints[seg_idx + 1]
        g0 = grip_waypoints[seg_idx]
        g1 = grip_waypoints[seg_idx + 1]

        seg_steps = max(1, int(round(duration / dt_sim)))
        for i in range(seg_steps):
            t_local = min(i * dt_sim, duration)
            q_arm, dq_arm = _quintic_segment(q0, q1, t_local, duration)
            g, dg = _quintic_segment(np.array([g0]), np.array([g1]), t_local, duration)

            idx = min(int(round(t_cursor / dt_sim)), cycle_steps - 1)
            q_cycle[idx, :N_ARM] = q_arm
            dq_cycle[idx, :N_ARM] = dq_arm
            q_cycle[idx, N_ARM:N_TOTAL] = [g[0], g[0]]
            dq_cycle[idx, N_ARM:N_TOTAL] = [dg[0], dg[0]]

            t_cursor += dt_sim

    # Fill any gap due to rounding with final home state.
    q_cycle[-1, :N_ARM] = q_waypoints[-1]
    q_cycle[-1, N_ARM:N_TOTAL] = [grip_waypoints[-1], grip_waypoints[-1]]

    # One-shot execution by default: hold final state after the planned cycle.
    steps = int(sim_time / dt_sim)
    q_plan = np.zeros((steps, N_TOTAL), dtype=float)
    dq_plan = np.zeros((steps, N_TOTAL), dtype=float)
    for i in range(steps):
        idx = min(i, cycle_steps - 1)
        q_plan[i, :] = q_cycle[idx, :]
        dq_plan[i, :] = dq_cycle[idx, :]

    return q_plan, dq_plan


def run_simulation(
    use_viewer=True,
    sim_time=20.0,
    mode="mpc",
    sim_dt=None,
    openloop_speed=1.0,
    log_every=1,
    grasp_assist=False,
    record_gif_path=None,
    gif_fps=20,
):
    if mode not in ("mpc", "openloop"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'mpc' or 'openloop'.")
    if openloop_speed <= 0.0:
        raise ValueError("openloop_speed must be > 0")
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    if gif_fps <= 0:
        raise ValueError("gif_fps must be > 0")

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

    q_plan, dq_plan = _preplan_reference_trajectory(sim_time, dt_sim, data, home)
    print(f"[Planner] Preplanned {len(q_plan)} steps with explicit Cartesian phases")

    data.qpos[0:N_ARM] = home
    data.qpos[N_ARM:N_TOTAL] = [0.04, 0.04]
    data.qvel[:] = 0.0
    data.ctrl[:N_ARM] = home
    mujoco.mj_forward(model, data)

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if gripper_act_id < 0:
        gripper_act_id = 7

    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
    right_pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_qpos_adr = None
    if box_body_id >= 0:
        box_jnt_id = model.body_jntadr[box_body_id]
        if box_jnt_id >= 0 and model.jnt_type[box_jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            box_qpos_adr = model.jnt_qposadr[box_jnt_id]

    print("\nFranka Panda Controller")
    print(f"Scene: {Path(SCENE_XML_PATH).name}")
    print(f"Pinocchio model: {Path(PIN_MODEL_PATH).name}")
    print(f"Mode: {mode}")
    print(f"MPC period: {dt_ctrl*1000:.1f} ms | sim dt: {dt_sim*1000:.1f} ms")
    _, _, anchor_name = _get_grasp_anchor_pose(
        model, data, gripper_site_id, ee_body_id, left_pad_geom_id, right_pad_geom_id
    )
    if anchor_name is not None:
        print(f"Grasp anchor: {anchor_name}")
    if mode == "openloop":
        print(f"Open-loop speed: {openloop_speed:.2f}x")
        print(f"Grasp assist: {'on' if grasp_assist else 'off'}")
    if record_gif_path:
        print(f"Recording GIF: {record_gif_path} @ {gif_fps} fps")
    if log_every > 1:
        print(f"Log decimation: every {log_every} steps")

    log_f, log_w = init_logger()
    u_current = home.copy()
    t_sim = 0.0
    step = 0
    solve_times = []
    box_start = None
    box_max_z = -1e9
    wall_t0 = time.perf_counter()
    assist_active = False
    carry_rel_pos = None
    carry_rel_rot = None
    gif_frames = []
    gif_frame_stride = max(1, int(round(1.0 / (gif_fps * dt_sim)))) if record_gif_path else None
    renderer = None
    camera = None
    if record_gif_path:
        import imageio.v2 as imageio

        off_w = int(getattr(model.vis.global_, "offwidth", 640))
        off_h = int(getattr(model.vis.global_, "offheight", 480))
        req_w, req_h = 960, 720
        render_w = max(120, min(req_w, off_w))
        render_h = max(120, min(req_h, off_h))

        try:
            renderer = mujoco.Renderer(model, height=render_h, width=render_w)
        except ValueError:
            # Final fallback for environments with unexpectedly tiny framebuffers.
            safe_w = max(64, min(render_w, off_w, 320))
            safe_h = max(64, min(render_h, off_h, 240))
            renderer = mujoco.Renderer(model, height=safe_h, width=safe_w)
            render_w, render_h = safe_w, safe_h

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth = 142.0
        camera.elevation = -15.0
        camera.distance = 1.25
        camera.lookat[:] = np.array([0.48, -0.03, 0.14], dtype=float)

        print(f"GIF render size: {render_w}x{render_h} (offscreen max {off_w}x{off_h})")

        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0

    if box_body_id >= 0:
        box_start = data.xpos[box_body_id].copy()

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                q_now = data.qpos.copy()
                q_ref = q_plan[min(step, len(q_plan) - 1), :].copy()
                dq_ref = dq_plan[min(step, len(dq_plan) - 1), :].copy()

                if mode == "mpc":
                    if step % mpc_every == 0:
                        t0 = time.perf_counter()
                        u_current = mpc.solve(q_now[:N_ARM], q_ref[:N_ARM], u_current)
                        solve_times.append(time.perf_counter() - t0)
                else:
                    plan_idx = min(int(step * openloop_speed), len(q_plan) - 1)
                    u_current = q_plan[plan_idx, :N_ARM].copy()
                    q_ref = q_plan[plan_idx, :].copy()
                    dq_ref = dq_plan[plan_idx, :].copy()

                data.ctrl[:N_ARM] = u_current
                data.ctrl[gripper_act_id] = float(np.clip(q_ref[N_ARM], 0.0, 0.04))

                if grasp_assist and mode == "openloop" and box_qpos_adr is not None and box_body_id >= 0:
                    grip_cmd = float(np.clip(q_ref[N_ARM], 0.0, 0.04))
                    hand_p, hand_R, _ = _get_grasp_anchor_pose(
                        model, data, gripper_site_id, ee_body_id, left_pad_geom_id, right_pad_geom_id
                    )
                    if hand_p is not None and hand_R is not None:
                        box_p = data.xpos[box_body_id].copy()
                        box_R = data.xmat[box_body_id].reshape(3, 3).copy()

                        if (not assist_active) and grip_cmd <= 0.006:
                            carry_rel_pos = hand_R.T @ (box_p - hand_p)
                            carry_rel_rot = hand_R.T @ box_R
                            assist_active = True
                        if assist_active and grip_cmd >= 0.020:
                            assist_active = False

                        if assist_active and carry_rel_pos is not None and carry_rel_rot is not None:
                            new_box_p = hand_p + hand_R @ carry_rel_pos
                            new_box_R = hand_R @ carry_rel_rot
                            quat = np.zeros(4, dtype=float)
                            mujoco.mju_mat2Quat(quat, new_box_R.reshape(-1))
                            data.qpos[box_qpos_adr:box_qpos_adr + 3] = new_box_p
                            data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = quat
                            data.qvel[model.jnt_dofadr[model.body_jntadr[box_body_id]]:model.jnt_dofadr[model.body_jntadr[box_body_id]] + 6] = 0.0
                            mujoco.mj_forward(model, data)

                ee_pos, _, _ = _get_grasp_anchor_pose(
                    model, data, gripper_site_id, ee_body_id, left_pad_geom_id, right_pad_geom_id
                )
                if ee_pos is None:
                    ee_pos = np.zeros(3)
                q_now = data.qpos.copy()
                if step % log_every == 0:
                    log_row(log_f, log_w, t_sim, q_now[:N_ARM], q_ref[:N_ARM], u_current, q_ref[N_ARM], ee_pos)

                mujoco.mj_step(model, data)
                if box_body_id >= 0:
                    box_max_z = max(box_max_z, float(data.xpos[box_body_id][2]))
                if renderer is not None and (step % gif_frame_stride == 0):
                    renderer.update_scene(data, camera=camera)
                    gif_frames.append(renderer.render().copy())
                viewer.sync()
                t_sim += dt_sim
                step += 1
    else:
        steps = int(sim_time / dt_sim)
        for _ in range(steps):
            q_now = data.qpos.copy()
            q_ref = q_plan[min(step, len(q_plan) - 1), :].copy()
            dq_ref = dq_plan[min(step, len(dq_plan) - 1), :].copy()

            if mode == "mpc":
                if step % mpc_every == 0:
                    t0 = time.perf_counter()
                    u_current = mpc.solve(q_now[:N_ARM], q_ref[:N_ARM], u_current)
                    solve_times.append(time.perf_counter() - t0)
            else:
                plan_idx = min(int(step * openloop_speed), len(q_plan) - 1)
                u_current = q_plan[plan_idx, :N_ARM].copy()
                q_ref = q_plan[plan_idx, :].copy()
                dq_ref = dq_plan[plan_idx, :].copy()

            data.ctrl[:N_ARM] = u_current
            data.ctrl[gripper_act_id] = float(np.clip(q_ref[N_ARM], 0.0, 0.04))

            if grasp_assist and mode == "openloop" and box_qpos_adr is not None and box_body_id >= 0:
                grip_cmd = float(np.clip(q_ref[N_ARM], 0.0, 0.04))
                hand_p, hand_R, _ = _get_grasp_anchor_pose(
                    model, data, gripper_site_id, ee_body_id, left_pad_geom_id, right_pad_geom_id
                )
                if hand_p is not None and hand_R is not None:
                    box_p = data.xpos[box_body_id].copy()
                    box_R = data.xmat[box_body_id].reshape(3, 3).copy()

                    if (not assist_active) and grip_cmd <= 0.006:
                        carry_rel_pos = hand_R.T @ (box_p - hand_p)
                        carry_rel_rot = hand_R.T @ box_R
                        assist_active = True
                    if assist_active and grip_cmd >= 0.020:
                        assist_active = False

                    if assist_active and carry_rel_pos is not None and carry_rel_rot is not None:
                        new_box_p = hand_p + hand_R @ carry_rel_pos
                        new_box_R = hand_R @ carry_rel_rot
                        quat = np.zeros(4, dtype=float)
                        mujoco.mju_mat2Quat(quat, new_box_R.reshape(-1))
                        data.qpos[box_qpos_adr:box_qpos_adr + 3] = new_box_p
                        data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = quat
                        data.qvel[model.jnt_dofadr[model.body_jntadr[box_body_id]]:model.jnt_dofadr[model.body_jntadr[box_body_id]] + 6] = 0.0
                        mujoco.mj_forward(model, data)

            ee_pos, _, _ = _get_grasp_anchor_pose(
                model, data, gripper_site_id, ee_body_id, left_pad_geom_id, right_pad_geom_id
            )
            if ee_pos is None:
                ee_pos = np.zeros(3)
            q_now = data.qpos.copy()
            if step % log_every == 0:
                log_row(log_f, log_w, t_sim, q_now[:N_ARM], q_ref[:N_ARM], u_current, q_ref[N_ARM], ee_pos)

            mujoco.mj_step(model, data)
            if box_body_id >= 0:
                box_max_z = max(box_max_z, float(data.xpos[box_body_id][2]))
            if renderer is not None and (step % gif_frame_stride == 0):
                renderer.update_scene(data, camera=camera)
                gif_frames.append(renderer.render().copy())
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
        print(f"Box max z: {box_max_z:.4f} m")
        print(f"Box start: [{box_start[0]:.3f}, {box_start[1]:.3f}, {box_start[2]:.3f}]")
        print(f"Box end:   [{box_end[0]:.3f}, {box_end[1]:.3f}, {box_end[2]:.3f}]")

    if record_gif_path and gif_frames:
        gif_out = Path(record_gif_path)
        gif_out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(gif_out), gif_frames, fps=gif_fps)
        print(f"GIF saved: {gif_out}")

    print(f"Log: {LOG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    presets = {
        "fast": {"sim_dt": 0.01, "openloop_speed": 2.5, "log_every": 30},
        "balanced": {"sim_dt": 0.005, "openloop_speed": 1.5, "log_every": 10},
        "accurate": {"sim_dt": 0.002, "openloop_speed": 1.0, "log_every": 1},
    }

    parser = argparse.ArgumentParser(description="Franka OSQP MPC controller")
    parser.add_argument("--headless", action="store_true", help="run without viewer")
    parser.add_argument("--sim-time", type=float, default=20.0, help="headless simulation duration (seconds)")
    parser.add_argument("--mode", choices=["mpc", "openloop"], default="mpc", help="control mode")
    parser.add_argument("--preset", choices=["fast", "balanced", "accurate"], default=None, help="runtime speed/fidelity preset")
    parser.add_argument("--sim-dt", type=float, default=None, help="override MuJoCo timestep (seconds)")
    parser.add_argument("--openloop-speed", type=float, default=None, help="trajectory playback speed for open-loop mode")
    parser.add_argument("--log-every", type=int, default=None, help="log one row every N simulation steps")
    parser.add_argument("--grasp-assist", action="store_true", help="deterministic carry assist in open-loop while gripper is closed")
    parser.add_argument("--record-gif", type=str, default=None, help="optional output GIF path for whole-arm visualization")
    parser.add_argument("--gif-fps", type=int, default=20, help="GIF frame rate when --record-gif is set")
    args = parser.parse_args()

    if args.preset is not None:
        p = presets[args.preset]
        if args.sim_dt is None:
            args.sim_dt = p["sim_dt"]
        if args.openloop_speed is None:
            args.openloop_speed = p["openloop_speed"]
        if args.log_every is None:
            args.log_every = p["log_every"]

    if args.openloop_speed is None:
        args.openloop_speed = 1.0
    if args.log_every is None:
        args.log_every = 1

    run_simulation(
        use_viewer=not args.headless,
        sim_time=args.sim_time,
        mode=args.mode,
        sim_dt=args.sim_dt,
        openloop_speed=args.openloop_speed,
        log_every=args.log_every,
        grasp_assist=args.grasp_assist,
        record_gif_path=args.record_gif,
        gif_fps=args.gif_fps,
    )


if __name__ == "__main__":
    main()
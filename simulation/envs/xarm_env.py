"""
xArm 6-DOF MuJoCo environment implementation.

Canonical environment interface matching the dataset dimensionality.
Used for simulation replay, control testing, and benchmarking.
Matches lerobot/utokyo_xarm_pick_and_place (6-DOF robot, parallel gripper).

Reference: tech spec §5 (MuJoCo Environment)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not installed. Install with: pip install mujoco")
    raise

logger = logging.getLogger(__name__)


class XArmEnv:
    """
    xArm 6-DOF MuJoCo environment for lerobot/utokyo_xarm_pick_and_place dataset.
    
    State: [q1-q6, gripper_left, gripper_right] (8-DOF: 6 arm + 2 gripper fingers)
    Action: [v1-v6, gripper_cmd] (7-D: 6 arm velocities + 1 gripper command)
    
    Sensors:
    - RGB camera (84×84 pixels, matches dataset)
    - Simulated event camera (via event_camera.py)
    - LiDAR rangefinders (32 rays)
    - Joint proprioception (positions and velocities)
    """
    
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    GRIPPER_JOINT_NAMES = ["gripper_left", "gripper_right"]
    ALL_JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES
    
    # Joint limits (rad for arm joints, m for gripper)
    JOINT_LIMITS = np.array([
        [-6.283, 6.283],      # joint1 (base yaw)
        [-3.665, 3.665],      # joint2 (shoulder pitch)
        [-6.109, 6.109],      # joint3 (shoulder roll)
        [-4.555, 4.555],      # joint4 (elbow pitch)
        [-6.109, 6.109],      # joint5 (wrist pitch)
        [-6.283, 6.283],      # joint6 (wrist roll)
        [0.0, 0.05],          # gripper_left (m)
        [0.0, 0.05],          # gripper_right (m)
    ])
    
    # Torque/force limits (Nm for arm joints, N for gripper)
    TORQUE_LIMITS = np.array([20.0, 15.0, 15.0, 10.0, 8.0, 6.0, 3.0, 3.0])
    
    # Velocity limits (rad/s for arm, m/s for gripper)
    VELOCITY_LIMITS = np.array([3.0, 2.5, 2.5, 2.0, 1.5, 2.0, 1.0, 1.0])
    
    def __init__(self, xml_path: Optional[Path] = None, render_size: int = 84):
        """
        Initialize xArm 6-DOF environment.
        
        Args:
            xml_path: Path to xarm_6dof.xml (default: simulation/models/xarm_6dof.xml)
            render_size: RGB camera resolution (default: 84 to match dataset)
        """
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "models" / "xarm_6dof.xml"
        
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MJCF model not found: {xml_path}")
        
        logger.info(f"Loading MuJoCo model from {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        self.n_joints = 8  # 6 arm + 2 gripper
        self.dt = self.model.opt.timestep
        self.render_size = render_size
        
        # Joint address lookup for efficient state access
        self._q_addrs = [self.model.joint(n).qposadr[0] for n in self.ALL_JOINT_NAMES]
        self._qd_addrs = [self.model.joint(n).dofadr[0] for n in self.ALL_JOINT_NAMES]
        
        # Site IDs for body tracking
        self._ee_site_id = self.model.site("ee_site").id
        self._obj_site_id = self.model.site("object_site").id
        
        # Renderers for different resolutions
        self.renderer_rgb = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.renderer_hi = mujoco.Renderer(self.model, height=480, width=480)
        
        # Step counter for tracking
        self._step_count = 0
        
        # Forward to initialize data
        mujoco.mj_forward(self.model, self.data)
        logger.info(f"✓ MuJoCo environment ready (dt={self.dt:.3f}s, n_joints={self.n_joints})")
    
    # ────────────────────────────────────────────────────────────────────────
    # State Access
    # ────────────────────────────────────────────────────────────────────────
    
    def get_joint_pos(self) -> np.ndarray:
        """Return joint positions [4] in rad/m."""
        return np.array([self.data.qpos[a] for a in self._q_addrs])
    
    def get_joint_vel(self) -> np.ndarray:
        """Return joint velocities [4] in rad/s or m/s."""
        return np.array([self.data.qvel[a] for a in self._qd_addrs])
    
    def get_state(self) -> np.ndarray:
        """Return full state [q, qd] (8-dim)."""
        return np.concatenate([self.get_joint_pos(), self.get_joint_vel()])
    
    def get_ee_pos(self) -> np.ndarray:
        """Return end-effector position [3] in world frame (m)."""
        return self.data.site_xpos[self._ee_site_id].copy()
    
    def get_object_pos(self) -> np.ndarray:
        """Return object center position [3] in world frame (m)."""
        return self.data.site_xpos[self._obj_site_id].copy()
    
    # ────────────────────────────────────────────────────────────────────────
    # Control & Simulation
    # ────────────────────────────────────────────────────────────────────────
    
    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Apply joint torques and advance simulation by one timestep.
        
        Args:
            action: [8] joint torques [Nm×6, N×2 for gripper]
        
        Returns:
            dict with sensor observations
        """
        assert action.shape == (8,), f"Expected action shape (8,), got {action.shape}"
        
        # Clip to torque limits
        action_clipped = np.clip(action, -self.TORQUE_LIMITS, self.TORQUE_LIMITS)
        
        # Apply control
        self.data.ctrl[:8] = action_clipped
        
        # Simulation step
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        
        return self._get_obs()
    
    def step_position(self, q_target: np.ndarray) -> Dict[str, Any]:
        """
        Position servo: set joint positions directly.
        
        Used for dataset replay validation (bypasses control).
        
        Args:
            q_target: [8] target joint positions
        
        Returns:
            observation dict
        """
        for i, addr in enumerate(self._q_addrs):
            self.data.qpos[addr] = q_target[i]
        
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        
        return self._get_obs()
    
    # ────────────────────────────────────────────────────────────────────────
    # Observation
    # ────────────────────────────────────────────────────────────────────────
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        return {
            "joint_pos": self.get_joint_pos(),      # [4]
            "joint_vel": self.get_joint_vel(),      # [4]
            "ee_pos": self.get_ee_pos(),            # [3]
            "object_pos": self.get_object_pos(),    # [3]
        }
    
    def render_rgb(self, camera: str = "camera_rgb", size: Optional[int] = None) -> np.ndarray:
        """
        Render RGB image from camera.
        
        Args:
            camera: Camera name in MJCF
            size: Image size (default: render_size)
        
        Returns:
            [H, W, 3] uint8 RGB array
        """
        if size is None:
            size = self.render_size
        
        # Use appropriate renderer based on size
        if size == self.render_size:
            renderer = self.renderer_rgb
        elif size == 480:
            renderer = self.renderer_hi
        else:
            # Create a new renderer for custom sizes
            renderer = mujoco.Renderer(self.model, height=size, width=size)
        
        renderer.update_scene(self.data, camera=camera)
        rgb = renderer.render()
        
        return rgb
    
    def get_lidar_readings(self) -> np.ndarray:
        """
        Get LiDAR rangefinder readings.
        
        Returns:
            [32] distances in meters (-1 = no hit within cutoff)
        """
        # Count rangefinders
        n_rangefinders = sum(
            1 for i in range(self.model.nsensor)
            if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_RANGEFINDER
        )
        
        readings = np.array([self.data.sensordata[i] for i in range(n_rangefinders)])
        return readings
    
    # ────────────────────────────────────────────────────────────────────────
    # Reset & Utilities
    # ────────────────────────────────────────────────────────────────────────
    
    def reset(self, q_init: Optional[np.ndarray] = None, object_pos: Optional[np.ndarray] = None):
        """
        Reset environment to initial state.
        
        Args:
            q_init: Initial joint positions [8] (6 arm + 2 gripper), default: home
            object_pos: Initial object position [3] (default: on table)
        """
        mujoco.mj_resetData(self.model, self.data)
        
        # Set arm and gripper pose
        if q_init is not None:
            assert q_init.shape == (8,), f"Expected q_init shape (8,), got {q_init.shape}"
            for i, addr in enumerate(self._q_addrs):
                self.data.qpos[addr] = q_init[i]
        else:
            # Home position: arm at rest, gripper open
            for i, addr in enumerate(self._q_addrs):
                self.data.qpos[addr] = 0.0
        
        # Set object position (on table surface by default)
        if object_pos is not None:
            # Object has freejoint, so qpos indices are [7, 8, 9] (pos) + [10-13] (quat)
            # Find object body
            obj_body_id = self.model.body("object").id
            # Generic approach: look for free joint
            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    # This is likely the object joint
                    qpos_adr = self.model.jnt_qposadr[i]
                    self.data.qpos[qpos_adr:qpos_adr+3] = object_pos
                    break
        else:
            # Default: object on table
            self.data.qpos[7:10] = [0.5, 0.0, 0.37]
        
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
    
    def check_success(self, height_threshold: float = 0.1) -> bool:
        """
        Check if task succeeded (object lifted above table).
        
        Success condition: object height > table_surface + threshold
        Table surface is at z=0.32m.
        
        Args:
            height_threshold: Minimum height above table (m)
        
        Returns:
            bool indicating success
        """
        obj_z = self.get_object_pos()[2]
        table_z = 0.32
        return obj_z > (table_z + height_threshold)
    
    def close(self):
        """Clean up renderer resources."""
        if hasattr(self, "renderer_rgb"):
            self.renderer_rgb.close()
        if hasattr(self, "renderer_hi"):
            self.renderer_hi.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

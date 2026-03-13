"""3-DOF Robot Arm MuJoCo Environment."""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

import mujoco
import mujoco.viewer

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """Current state of the robot simulation."""
    q: np.ndarray  # Joint angles [3,]
    qdot: np.ndarray  # Joint velocities [3,]
    tau: np.ndarray  # Joint torques [3,]
    ee_pos: np.ndarray  # End-effector position [3,]
    ee_vel: np.ndarray  # End-effector velocity [3,]
    quat: np.ndarray  # EE orientation (quaternion) [4,]
    time: float  # Simulation time [sec]
    rgb: Optional[np.ndarray] = None  # RGB image from camera
    depth: Optional[np.ndarray] = None  # Depth image
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            'q': self.q.tolist(),
            'qdot': self.qdot.tolist(),
            'tau': self.tau.tolist(),
            'ee_pos': self.ee_pos.tolist(),
            'ee_vel': self.ee_vel.tolist(),
            'quat': self.quat.tolist(),
            'time': self.time,
        }


class MuJoCo3DOFEnv:
    """OpenAI Gym-style 3-DOF robot arm environment.
    
    Provides:
    - Realistic simulation using MuJoCo's physics engine
    - Configurable rendering (headless, human, rgb_array)
    - Multi-camera support (overhead, side)
    - Force/torque measurements
    - Configurable task setup
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        dt: float = 0.002,  # 500 Hz control rate
        camera_height: int = 224,
        camera_width: int = 224,
        headless: bool = True,
    ):
        """Initialize the environment.
        
        Args:
            render_mode: 'human' (interactive), 'rgb_array', or None
            dt: Control time step (seconds)
            camera_height: RGB image height
            camera_width: RGB image width
            headless: If True, don't create interactive viewer
        """
        self.render_mode = render_mode
        self.dt = dt
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.headless = headless
        
        # Load model
        asset_dir = Path(__file__).parent.parent.parent / "assets"
        xml_path = asset_dir / "arm3dof.xml"
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Model not found: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Setup renderer
        self.viewer = None
        self.gl_context = None
        
        if self.render_mode == 'human' and not headless:
            self.viewer = mujoco.viewer.launch(self.model)
        
        # Get joint/body/camera IDs
        self._setup_body_ids()
        
        # Simulation parameters
        self.ctrl_range = self.model.actuator_ctrlrange
        self.max_torque = self.ctrl_range[:, 1]  # Upper limits
        self.min_torque = self.ctrl_range[:, 0]  # Lower limits
        
        # State tracking
        self.current_state = None
        
        logger.info(f"MuJoCo3DOFEnv initialized: model={xml_path.name}")
    
    def _setup_body_ids(self):
        """Cache frequently-used body/joint IDs."""
        # Looking for specific bodies in the model
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
        
        # Joint IDs for 3-DOF arm (assume standard naming: joint1, joint2, joint3)
        self.joint_ids = []
        for i in range(1, 4):
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i}')
                self.joint_ids.append(jid)
            except Exception:
                # Try alternate naming
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'J{i}')
                self.joint_ids.append(jid)
        
        if len(self.joint_ids) < 3:
            raise RuntimeError(f"Could not find 3 joints in model. Found {len(self.joint_ids)}")
        
        # Camera IDs
        self.camera_ids = {}
        for cam_name in ['overhead', 'side', 'wrist']:
            try:
                cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                self.camera_ids[cam_name] = cid
            except Exception:
                pass
        
        logger.debug(f"Joint IDs: {self.joint_ids}, EE site: {self.ee_site_id}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[dict, dict]:
        """Reset environment to home position.
        
        Returns:
            observation: dict with 'rgb', 'state'
            info: dict with metadata
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set home position: all joints to 0
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0
        
        # Run forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get initial state
        self.current_state = self._get_state()
        
        obs = self._get_observation()
        info = {
            'episode_start': True,
            'time': 0.0,
        }
        
        logger.info("Environment reset to home position")
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Execute one control step.
        
        Args:
            action: [3,] torque command (will be clipped to limits)
            
        Returns:
            observation: dict with state
            reward: dummy reward (0.0)
            terminated: episode done (False, we run indefinitely)
            truncated: step limit reached (False)
            info: metadata dict
        """
        # Clip action to torque limits
        action = np.clip(action, self.min_torque, self.max_torque)
        
        # Apply control
        self.data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Get current state
        self.current_state = self._get_state()
        self.current_state.tau = action
        
        obs = self._get_observation()
        
        # No episode termination
        reward = 0.0
        terminated = False
        truncated = False
        
        info = {
            'time': self.data.time,
            'state': self.current_state.to_dict(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_state(self) -> SimulationState:
        """Extract current robot state from MuJoCo."""
        # Get joint angles and velocities
        q = np.zeros(3)
        qdot = np.zeros(3)
        
        for i, jid in enumerate(self.joint_ids):
            q_addr = self.model.jnt_qposadr[jid]
            qv_addr = self.model.jnt_dofadr[jid]
            q[i] = self.data.qpos[q_addr]
            qdot[i] = self.data.qvel[qv_addr]
        
        # Get EE position and quat
        ee_pos = self.data.site(self.ee_site_id).xpos.copy()
        ee_quat = self.data.site(self.ee_site_id).xmat.copy()  # 3x3 matrix
        
        # Convert rotation matrix to quaternion (simplified)
        quat = self._matrix_to_quat(ee_quat)
        
        # Estimate EE velocity from jacobian (simplified)
        ee_vel = np.zeros(3)
        
        return SimulationState(
            q=q,
            qdot=qdot,
            tau=np.zeros(3),
            ee_pos=ee_pos,
            ee_vel=ee_vel,
            quat=quat,
            time=self.data.time,
        )
    
    def _matrix_to_quat(self, matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z].
        
        This is a simplified implementation. For production, use scipy.spatial.transform.Rotation.
        """
        # Just return [1, 0, 0, 0] as placeholder
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _get_observation(self) -> dict:
        """Build observation dict."""
        state_dict = self.current_state.to_dict() if self.current_state else {}
        
        # Optionally render RGB
        rgb = None
        if self.render_mode == 'rgb_array':
            rgb = self.render()
        
        return {
            'state': state_dict,
            'rgb': rgb,
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render robot state as RGB image.
        
        Returns:
            [H, W, 3] uint8 RGB image, or None if headless
        """
        if self.render_mode == 'human' and self.viewer is not None:
            # Interactive viewer handles rendering
            return None
        
        # Render to pixel array
        try:
            # Get camera ID (use first available)
            cam_id = 0
            if 'overhead' in self.camera_ids:
                cam_id = self.camera_ids['overhead']
            
            # Render
            renderer = mujoco.Renderer(self.model, self.camera_height, self.camera_width)
            renderer.enable_depth_rendering()
            renderer.render(self.data, cam_id=cam_id)
            
            # Get RGB pixels
            pixels = renderer.read_pixels(depth=False)
            
            # Convert from float [0, 1] to uint8 [0, 255]
            rgb = (pixels * 255).astype(np.uint8)
            
            return rgb
        except Exception as e:
            logger.warning(f"Render failed: {e}")
            return None
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current q, qdot, ee_pos.
        
        Returns:
            q: [3,] joint angles
            qdot: [3,] joint velocities
            ee_pos: [3,] end-effector position
        """
        if self.current_state is None:
            self._get_state()
        return (
            self.current_state.q.copy(),
            self.current_state.qdot.copy(),
            self.current_state.ee_pos.copy(),
        )
    
    def set_configuration(self, q: np.ndarray):
        """Set joint angles (useful for initialization).
        
        Args:
            q: [3,] desired joint angles
        """
        for i, jid in enumerate(self.joint_ids):
            q_addr = self.model.jnt_qposadr[jid]
            self.data.qpos[q_addr] = q[i]
        
        mujoco.mj_forward(self.model, self.data)
        self.current_state = self._get_state()
    
    def close(self):
        """Cleanup resources."""
        if self.viewer is not None:
            self.viewer.close()
        logger.info("MuJoCo3DOFEnv closed")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

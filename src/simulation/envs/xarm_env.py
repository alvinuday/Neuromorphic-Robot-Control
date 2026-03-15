"""xArm 6-DOF MuJoCo environment."""
import os
import numpy as np
import mujoco
from src.core.base_env import BaseEnv


_MJCF_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'assets/xarm_6dof.xml')


class XArmEnv(BaseEnv):
    """
    MuJoCo simulation environment for xArm 6-DOF robot.
    
    Observation: {'q': [6], 'qdot': [6], 'rgb': [84, 84, 3]}
    Action: [8] torques (6 arm + 2 gripper)
    """
    
    def __init__(self, render_mode='human'):
        """
        Initialize environment.
        
        Args:
            render_mode: 'human' (display), 'offscreen', or None
        """
        self.render_mode = render_mode
        assert render_mode in ('human', 'offscreen', None)
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(_MJCF_PATH)
        self.data = mujoco.MjData(self.model)

        # Resolve arm/gripper joint addresses explicitly.
        # The scene contains a free block joint before arm joints, so array prefixes
        # do not correspond to the robot arm.
        self.arm_joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
        ]
        self.gripper_joint_names = ['gripper_left', 'gripper_right']
        self.control_joint_names = self.arm_joint_names + self.gripper_joint_names

        self.arm_qpos_idx = np.array([self._joint_qpos_adr(n) for n in self.arm_joint_names], dtype=np.int32)
        self.arm_qvel_idx = np.array([self._joint_qvel_adr(n) for n in self.arm_joint_names], dtype=np.int32)
        self.ctrl_qfrc_idx = np.array([self._joint_qvel_adr(n) for n in self.control_joint_names], dtype=np.int32)
        self.gripper_qpos_idx = np.array([self._joint_qpos_adr(n) for n in self.gripper_joint_names], dtype=np.int32)
        self.available_cameras = {"cam_front", "cam_side", "cam_far", "cam_top"}
        
        # Renderer
        self.renderer = None
        self.camera_name = 'cam_side'
        if render_mode in ('human', 'offscreen'):
            # Keep a high-resolution render target for clear GIFs and previews.
            self.renderer = mujoco.Renderer(self.model, width=640, height=480)
        
        # Action limits
        self.act_range = np.array(
            [20., 15., 15., 10., 8., 6., 5., 5.],  # tau_max
            dtype=np.float32
        )
        
        self.step_count = 0
        self.max_steps = 1000

    def _joint_id(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint not found in model: {joint_name}")
        return jid

    def _joint_qpos_adr(self, joint_name: str) -> int:
        jid = self._joint_id(joint_name)
        return int(self.model.jnt_qposadr[jid])

    def _joint_qvel_adr(self, joint_name: str) -> int:
        jid = self._joint_id(joint_name)
        return int(self.model.jnt_dofadr[jid])
    
    def reset(self):
        """Reset to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        # Start from a slightly bent pose so links are visible from the default camera.
        self.data.qpos[self.arm_qpos_idx] = np.array([0.2, -0.7, 0.5, 0.8, -0.4, 0.2], dtype=np.float32)
        self.data.qpos[self.gripper_qpos_idx] = 0.045  # Gripper open (slide joints)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    def step(self, action):
        """Apply action and step the environment."""
        # Clamp action to limits
        action = np.clip(action, -self.act_range, self.act_range)
        
        # Apply torques only to arm+gripper DOFs (not object free-joint DOFs).
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.ctrl_qfrc_idx] = action
        
        # Step simulation (5 substeps)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        reward = 0.0  # No reward shaping in base env
        info = {}
        
        return obs, reward, done, info
    
    def _get_obs(self):
        """Generate observation dict."""
        # Joint angles (first 6 DOF, arm only)
        q = self.data.qpos[self.arm_qpos_idx].copy().astype(np.float32)
        
        # Joint velocities
        qdot = self.data.qvel[self.arm_qvel_idx].copy().astype(np.float32)
        
        # RGB image render
        rgb = self._render_rgb()
        
        return {
            'q': q,
            'qdot': qdot,
            'rgb': rgb,
        }

    def set_camera(self, camera_name: str):
        """Set active named camera for rendering."""
        if not camera_name:
            return
        self.camera_name = camera_name if camera_name in self.available_cameras else 'cam_far'

    def set_arm_state(self, q_arm: np.ndarray, qdot_arm: np.ndarray | None = None):
        """Directly set 6-DOF arm joint state for deterministic pose replay."""
        q_arm = np.asarray(q_arm, dtype=np.float32).reshape(-1)
        if q_arm.size < 6:
            raise ValueError(f"Expected at least 6 arm joints, got {q_arm.size}")
        self.data.qpos[self.arm_qpos_idx] = q_arm[:6]
        if qdot_arm is None:
            self.data.qvel[self.arm_qvel_idx] = 0.0
        else:
            qdot_arm = np.asarray(qdot_arm, dtype=np.float32).reshape(-1)
            self.data.qvel[self.arm_qvel_idx] = qdot_arm[:6]
        mujoco.mj_forward(self.model, self.data)
    
    def _render_rgb(self, output_size=(84, 84), enhance=True):
        """Render RGB image with configurable size and optional contrast enhancement."""
        if self.renderer is None:
            # Return dummy image if no renderer
            return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        try:
            self.renderer.update_scene(self.data, camera=self.camera_name)
        except Exception:
            # Fallback for older MuJoCo bindings that do not accept camera name.
            self.renderer.update_scene(self.data)

        rgb = self.renderer.render()

        # Robust fallback for occasional invalid camera orientation that yields near-black frames.
        if rgb.mean() < 1.5 and self.camera_name != 'cam_far':
            try:
                self.renderer.update_scene(self.data, camera='cam_far')
                rgb = self.renderer.render()
            except Exception:
                pass

        if enhance:
            # Improve dark scene visibility by stretching the dynamic range.
            low = np.percentile(rgb, 2)
            high = np.percentile(rgb, 98)
            if high > low:
                rgb = np.clip((rgb - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        
        # Convert from [H, W, 3] BGR to RGB
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb = rgb[..., ::-1]  # BGR to RGB
        
        # Resize to requested shape if needed
        target_w, target_h = output_size
        if rgb.shape[0] != target_h or rgb.shape[1] != target_w:
            from src.utils.image_utils import resize_image
            rgb = resize_image(rgb, (target_w, target_h))
        
        return rgb.astype(np.uint8)

    def render_frame(self, width=640, height=480):
        """Render a high-quality frame for web visualization/GIF export."""
        return self._render_rgb(output_size=(width, height), enhance=True)
    
    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            del self.renderer
        self.renderer = None

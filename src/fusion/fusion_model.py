"""
Multimodal Sensor Fusion Module (Deferred to Phase 9-10).

Currently: Pass-through numpy-based preprocessor for sensor data normalization.

Future (Phase 9-10): Implement sklearn-based or torch-based fusion encoders after:
1. ✓ Verify system works end-to-end with VLA (Gate 5-6, B1-B5 benchmarks)
2. ✓ Validate that feature vectors improve VLA + MPC performance
3. Determine whether to:
   a. Modify VLA to accept feature vectors (requires retraining)
   b. Use sklearn dimensionality reduction (PCA, etc.)
   c. Use lightweight torch-based encoders (after validation)

Reference: tech spec §8 (Multimodal Sensor Fusion), Phase 9-10 roadmap
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SensorFusionProcessor:
    """
    Numpy-based sensor data preprocessor.
    
    Normalizes all sensor modalities to consistent ranges for passing to MPC/VLA.
    
    Currently: Pass-through. Future fusion encoding deferred to Phase 9-10.
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        logger.info("SensorFusionProcessor initialized (deferred fusion encoder to Phase 9-10)")
    
    def preprocess_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """
        Normalize RGB to [0, 1].
        
        Args:
            rgb: [H, W, 3] uint8 RGB image
        
        Returns:
            [H, W, 3] float32 normalized
        """
        if rgb.dtype == np.uint8:
            return rgb.astype(np.float32) / 255.0
        return rgb.astype(np.float32)
    
    def preprocess_events(self, event_voxel: np.ndarray) -> np.ndarray:
        """
        Normalize event voxel grid to [-1, 1].
        
        Args:
            event_voxel: [T, H, W] int8 event voxel
        
        Returns:
            [T, H, W] float32 normalized
        """
        return event_voxel.astype(np.float32) / 127.0
    
    def preprocess_lidar(self, lidar_features: np.ndarray) -> np.ndarray:
        """
        LiDAR features already normalized [0, 1]; ensure dtype.
        
        Args:
            lidar_features: [32] normalized features
        
        Returns:
            [32] float32
        """
        return lidar_features.astype(np.float32)
    
    def preprocess_proprio(self, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        """
        Normalize proprioceptive state.
        
        Args:
            joint_pos: [8] joint positions in rad/m
            joint_vel: [8] joint velocities in rad/s or m/s
        
        Returns:
            [16] float32 normalized joint state
        """
        # Normalize positions to [-1, 1] (assuming ±π radians)
        pos_norm = joint_pos / np.pi
        # Normalize velocities to [-1, 1] (max ~3 rad/s)
        vel_norm = joint_vel / 3.0
        
        return np.concatenate([pos_norm, vel_norm]).astype(np.float32)
    
    def fuse_all_modalities(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess all modalities from raw observation.
        
        Args:
            obs: observation dict from XArmEnv._get_obs()
        
        Returns:
            dict with preprocessed modalities
        """
        return {
            "rgb": self.preprocess_rgb(obs["rgb"]),
            "event_voxel": self.preprocess_events(obs["event_voxel"]),
            "lidar_features": self.preprocess_lidar(obs["lidar_features"]),
            "proprio": self.preprocess_proprio(obs["joint_pos"], obs["joint_vel"]),
            
            # Passthrough tracking data
            "ee_pos": obs["ee_pos"],
            "object_pos": obs["object_pos"],
        }
    
    def extract_vla_input(self, obs: Dict[str, np.ndarray]) -> tuple:
        """
        Extract minimal inputs for SmolVLA (RGB + state).
        
        SmolVLA accepts: [RGB image, language instruction, robot state]
        
        Args:
            obs: preprocessed observation dict
        
        Returns:
            (rgb, state) tuple for VLA input
        """
        rgb = obs["rgb"]  # [H, W, 3] float32
        
        # VLA state input: joint positions (6 arm + gripper)
        # Matches lerobot/utokyo_xarm_pick_and_place state format
        state = obs["proprio"][:7]  # Take first 7 components (6 arm + gripper)
        
        return rgb, state


# Placeholder for future neural fusion encoders (Phase 9-10)
class FusionEncoderPlaceholder:
    """Placeholder for future sklearn/torch-based fusion encoder."""
    
    def __init__(self):
        raise NotImplementedError(
            "Fusion encoders deferred to Phase 9-10. "
            "Current approach: Pass preprocessed sensor data directly to MPC/VLA. "
            "See AGENT_STATE.md for deferral plan."
        )


def create_fusion_processor() -> SensorFusionProcessor:
    """Factory function to create sensor fusion preprocessor."""
    return SensorFusionProcessor()

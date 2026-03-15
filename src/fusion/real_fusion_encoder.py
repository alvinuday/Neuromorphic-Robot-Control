"""Multimodal sensor fusion encoder for xArm control."""
import numpy as np
import time
from typing import Dict, Optional
from src.simulation.cameras.event_camera_simple import EventCameraSimulator, LiDARSimulator


class RealFusionEncoder:
    """
    Multimodal sensor fusion with 5 modes.
    
    Modes:
        M0: RGB only (128-dim)
        M1: RGB + Events (224-dim)
        M2: RGB + LiDAR (192-dim)
        M3: RGB + Proprioception (160-dim)
        M4: Full fusion RGB+Events+LiDAR+Proprio (320-dim)
    
    All feature extraction uses numpy only — no pretrained neural networks.
    """
    
    def __init__(self, 
                 use_rgb: bool = True,
                 use_events: bool = False,
                 use_lidar: bool = False,
                 use_proprio: bool = False):
        self.use_rgb = use_rgb
        self.use_events = use_events
        self.use_lidar = use_lidar
        self.use_proprio = use_proprio
        
        # Initialize simulators
        self._event_cam = EventCameraSimulator()
        self._lidar_sim = LiDARSimulator()
        
        # Compute feature dimension
        dim = 0
        if use_rgb:
            dim += 128  # spatial(64) + stats(32) + gradient(32)
        if use_events:
            dim += 96   # event features
        if use_lidar:
            dim += 64   # lidar features
        if use_proprio:
            dim += 32   # proprioception features
        
        self.feature_dim = dim
    
    @classmethod
    def mode_rgb_only(cls) -> 'RealFusionEncoder':
        """M0: RGB only (128-dim)."""
        return cls(use_rgb=True, use_events=False, use_lidar=False, use_proprio=False)
    
    @classmethod
    def mode_rgb_events(cls) -> 'RealFusionEncoder':
        """M1: RGB + Events (224-dim)."""
        return cls(use_rgb=True, use_events=True, use_lidar=False, use_proprio=False)
    
    @classmethod
    def mode_rgb_lidar(cls) -> 'RealFusionEncoder':
        """M2: RGB + LiDAR (192-dim)."""
        return cls(use_rgb=True, use_events=False, use_lidar=True, use_proprio=False)
    
    @classmethod
    def mode_rgb_proprio(cls) -> 'RealFusionEncoder':
        """M3: RGB + Proprioception (160-dim)."""
        return cls(use_rgb=True, use_events=False, use_lidar=False, use_proprio=True)
    
    @classmethod
    def mode_full(cls) -> 'RealFusionEncoder':
        """M4: Full fusion (320-dim)."""
        return cls(use_rgb=True, use_events=True, use_lidar=True, use_proprio=True)
    
    def encode(self, observation: Dict) -> np.ndarray:
        """
        Encode multimodal observation into feature vector.
        
        Args:
            observation: dict with keys:
                - 'rgb': [84, 84, 3] uint8 image
                - 'prev_rgb': [84, 84, 3] uint8 (for events, optional)
                - 'state': [7] proprioception state
        
        Returns:
            feat: [feature_dim] float32 feature vector
        """
        rgb = observation['rgb'].astype(np.uint8)
        parts = []
        
        if self.use_rgb:
            parts.append(self._extract_rgb(rgb))
        
        if self.use_events:
            # Process current frame; EventCameraSimulator maintains internal buffer
            events = self._event_cam.process_frame(rgb)
            parts.append(self._extract_events(events))
        
        if self.use_lidar:
            parts.append(self._extract_lidar(rgb))
        
        if self.use_proprio:
            state = observation.get('state', np.zeros(7))
            parts.append(self._extract_proprio(state))
        
        feat = np.concatenate(parts).astype(np.float32)
        assert feat.shape == (self.feature_dim,), \
            f"Dim mismatch: {feat.shape} vs expected {(self.feature_dim,)}"
        
        return feat
    
    def encode_with_timing(self, observation: Dict) -> tuple:
        """Encode with timing information."""
        t0 = time.perf_counter()
        feat = self.encode(observation)
        ms = (time.perf_counter() - t0) * 1000.
        return feat, ms
    
    # ── Feature extractors ───────────────────────────────────────────────────
    
    def _extract_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """128-dim: spatial(64) + channel_stats(32) + gradient(32)."""
        img = rgb.astype(np.float32) / 255.
        H, W = img.shape[:2]
        
        # 1. 8×8 spatial grid → 64-dim
        bh, bw = max(1, H // 8), max(1, W // 8)
        spatial = img[:bh*8, :bw*8].reshape(8, bh, 8, bw, 3).mean(axis=(1, 3, 4))
        spatial = spatial.flatten()  # [64]
        
        # 2. Per-channel statistics → 32-dim
        stats = []
        for c in range(3):
            ch = img[:, :, c]
            stats += [
                float(ch.mean()),
                float(ch.std()),
                float(np.percentile(ch, 25)),
                float(np.percentile(ch, 75)),
                float(np.median(ch)),
                float(ch.min()),
                float(ch.max()),
                float(np.var(ch)),
                float((ch > 0.5).mean()),
                float((ch < 0.1).mean()),
            ]
        stats_array = np.array(stats[:30], dtype=np.float32)
        pad_amount = max(0, 32 - len(stats_array))
        if pad_amount > 0:
            stats_array = np.pad(stats_array, (0, pad_amount))
        
        # 3. Gradient features → 32-dim
        gray = img.mean(axis=2)
        gx = np.abs(np.diff(gray, axis=1)).mean(axis=1)  # [H]
        gy = np.abs(np.diff(gray, axis=0)).mean(axis=1)  # [H-1]
        gx16 = np.interp(np.linspace(0, len(gx)-1, 16), np.arange(len(gx)), gx)
        gy16 = np.interp(np.linspace(0, len(gy)-1, 16), np.arange(len(gy)), gy)
        grad = np.concatenate([gx16, gy16]).astype(np.float32)  # [32]
        
        return np.concatenate([spatial, stats_array, grad])  # [128]
    
    def _extract_events(self, events: np.ndarray) -> np.ndarray:
        """96-dim event features from [T, H, W] voxel grid."""
        if events is None or events.size == 0:
            return np.zeros(96, dtype=np.float32)
        
        # Events shape: [temporal_bins, 84, 84]
        feat = []
        
        # Total event count across all time bins
        total = float(events.sum())
        feat.append(total / max(1.0, 255.0 * 84 * 84))  # normalized
        
        # Temporal statistics
        if events.ndim >= 3:
            for t in range(min(events.shape[0], 10)):
                feat.append(float(events[t].sum()))
        
        # Spatial distribution (8x8 grid on sum across time)
        events_spatial = events.sum(axis=0) if events.ndim >= 3 else events  # [H, W]
        H, W = events_spatial.shape[:2]
        bh, bw = max(1, H // 8), max(1, W // 8)
        
        spatial_grid = events_spatial[:bh*8, :bw*8].reshape(8, bh, 8, bw).sum(axis=(1, 3))
        spatial_features = spatial_grid.flatten() / max(1.0, spatial_grid.sum())
        feat.extend(spatial_features.tolist())
        
        # Ensure we have exactly 96 features
        result = np.array(feat[:96], dtype=np.float32)
        pad_amount = max(0, 96 - len(result))
        if pad_amount > 0:
            result = np.pad(result, (0, pad_amount))
        
        return result.astype(np.float32)
    
    def _extract_lidar(self, rgb: np.ndarray) -> np.ndarray:
        """64-dim LiDAR features from simulated depth."""
        # Simulate depth map from RGB intensity
        depth = rgb.mean(axis=2).astype(np.float32) / 255.0
        
        # Divide into 8x8 grid, compute statistics
        H, W = depth.shape
        bh, bw = max(1, H // 8), max(1, W // 8)
        
        lidar_grid = depth[:bh*8, :bw*8].reshape(8, bh, 8, bw)
        features = []
        
        for i in range(8):
            for j in range(8):
                patch = lidar_grid[i, :, j, :]
                features.extend([
                    float(patch.mean()),
                    float(patch.std()),
                ])
        
        # Ensure we have exactly 64 features
        result = np.array(features[:64], dtype=np.float32)
        pad_amount = max(0, 64 - len(result))
        if pad_amount > 0:
            result = np.pad(result, (0, pad_amount))
        
        return result.astype(np.float32)
    
    def _extract_proprio(self, state: np.ndarray) -> np.ndarray:
        """32-dim: joint angles(6) + velocities(6) + gripper(2) + pad(18)."""
        q = np.asarray(state, dtype=np.float32)
        
        # Normalize joint angles to [-1, 1]
        q6 = q[:6] / np.pi if len(q) >= 6 else np.zeros(6)
        
        # Velocities (not available in state, use zeros)
        vel = np.zeros(6, dtype=np.float32)
        
        # Gripper position
        grip = np.array([float(q[6])] if len(q) > 6 else [0.], dtype=np.float32)
        
        # Concatenate and pad
        raw = np.concatenate([q6, vel, grip]).astype(np.float32)  # [13]
        pad_amount = max(0, 32 - len(raw))
        if pad_amount > 0:
            result = np.pad(raw, (0, pad_amount)).astype(np.float32)  # [32]
        else:
            result = raw[:32].astype(np.float32)
        
        return result

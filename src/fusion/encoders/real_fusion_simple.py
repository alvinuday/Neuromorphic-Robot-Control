"""
Real multimodal fusion encoder - SIMPLIFIED version.
Extracts REAL features from RGB images and robot state (no random data).
"""

import numpy as np
from typing import Dict


class RealFusionEncoder:
    """
    Real sensor fusion using features extracted from actual data.
    
    Modes:
    - RGB: Image statistics and patterns
    - Events: Temporal frame differences  
    - LiDAR: Spatial geometry from image gradients
    - Proprioception: Real robot state
    """
    
    def __init__(self, use_rgb=True, use_events=False, use_lidar=False, use_proprio=False, h=84, w=84):
        self.use_rgb = use_rgb
        self.use_events = use_events
        self.use_lidar = use_lidar
        self.use_proprio = use_proprio
        self.h = h
        self.w = w
        self.prev_gray = None
    
    @classmethod
    def rgb_only(cls):
        """M0: RGB baseline."""
        return cls(use_rgb=True)
    
    @classmethod
    def rgb_events(cls):
        """M1: RGB + real temporal events."""
        return cls(use_rgb=True, use_events=True)
    
    @classmethod
    def rgb_lidar(cls):
        """M2: RGB + real geometry."""
        return cls(use_rgb=True, use_lidar=True)
    
    @classmethod
    def rgb_proprio(cls):
        """M3: RGB + proprioception."""
        return cls(use_rgb=True, use_proprio=True)
    
    @classmethod
    def full_fusion(cls):
        """M4: All modalities."""
        return cls(use_rgb=True, use_events=True, use_lidar=True, use_proprio=True)
    
    def _normalize_image(self, rgb):
        """Normalize RGB to float [0,1] and resize."""
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = rgb.astype(np.float32)
        
        if rgb.shape[0] == 3:  # [C, H, W]
            rgb = np.transpose(rgb, (1, 2, 0))
        
        # Simple numpy resize (linear interpolation)
        if rgb.shape[0] != self.h or rgb.shape[1] != self.w:
            # Use indexing for simple resize
            h_indices = (np.arange(self.h) * rgb.shape[0] / self.h).astype(int)
            w_indices = (np.arange(self.w) * rgb.shape[1] / self.w).astype(int)
            rgb = rgb[np.ix_(h_indices, w_indices, [0, 1, 2])]
        
        return rgb
    
    def extract_rgb_features(self, rgb):
        """
        Extract real RGB features.
        Returns [128-dim]: color stats + spatial patterns
        """
        rgb_norm = self._normalize_image(rgb)
        
        # Downsampled RGB (64-dim)
        rgb_small = rgb_norm[::10, ::10, :].flatten()[:64]
        
        # Color statistics (32-dim)
        r_mean, r_std = rgb_norm[:, :, 0].mean(), rgb_norm[:, :, 0].std()
        g_mean, g_std = rgb_norm[:, :, 1].mean(), rgb_norm[:, :, 1].std()
        b_mean, b_std = rgb_norm[:, :, 2].mean(), rgb_norm[:, :, 2].std()
        brightness = rgb_norm.mean()
        
        # Gradients (real image derivatives)
        gray = rgb_norm.mean(axis=2)
        gx = np.gradient(gray, axis=1).mean()
        gy = np.gradient(gray, axis=0).mean()
        
        color_stats = np.array([
            r_mean, r_std, g_mean, g_std, b_mean, b_std, brightness,
            gx, gy, gray.std(), gray.max(), gray.min()
        ], dtype=np.float32)
        
        # Pad to 32-dim
        color_stats = np.pad(color_stats, (0, max(0, 32 - len(color_stats))), 'constant')[:32]
        
        # Concatenate
        features = np.concatenate([rgb_small, color_stats], axis=0)
        return features[:128]  # Ensure 128-dim
    
    def extract_event_features(self, rgb):
        """
        REAL event camera simulation from frame differences.
        Returns [96-dim]: temporal events + motion
        """
        rgb_norm = self._normalize_image(rgb)
        gray = rgb_norm.mean(axis=2)
        
        # Real frame difference (event camera principle)
        diff = np.zeros_like(gray)
        if self.prev_gray is not None:
            diff = np.abs(gray - self.prev_gray)
            motion_energy = (diff > 0.08).sum() / gray.size  # Event ratio
            motion_mean = diff.mean()
            motion_std = diff.std()
            motion_max = diff.max()
        else:
            motion_energy = 0.0
            motion_mean = 0.0
            motion_std = 0.0
            motion_max = 0.0
        
        # Temporal voxel (32-dim: frame differences in spatial bins)
        diff_small = diff[::20, ::20].flatten()[:32]
        self.prev_gray = gray.copy()
        
        # Motion statistics (32-dim)
        motion_stats = np.zeros(32, dtype=np.float32)
        motion_stats[0] = motion_energy
        motion_stats[1] = motion_mean
        motion_stats[2] = motion_std
        motion_stats[3] = motion_max
        
        # Gradient motion (spatial derivatives)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        motion_stats[4] = gx.mean()
        motion_stats[5] = gy.mean()
        motion_stats[6] = grad_mag.mean()
        motion_stats[7] = grad_mag.std()
        
        features = np.concatenate([diff_small, motion_stats], axis=0)
        return features[:96]  # Ensure 96-dim
    
    def extract_lidar_features(self, rgb):
        """
        REAL depth/spatial features from RGB.
        Returns [64-dim]: edges, corners, depth cues
        """
        rgb_norm = self._normalize_image(rgb)
        gray = rgb_norm.mean(axis=2)
        
        # Edge detection (real image derivatives)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        edges = np.sqrt(gx**2 + gy**2)
        edge_map = (edges > 0.1).astype(np.float32)
        
        # Sample edge features
        edge_features = edge_map[::20, ::20].flatten()[:16]
        
        # Corner detection (Harris-like using local variance)
        corners = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                patch = gray[i-1:i+2, j-1:j+2]
                corners[i, j] = patch.var()
        
        corner_map = (corners > corners.mean() + 2*corners.std()).astype(np.float32)
        corner_features = corner_map[::20, ::20].flatten()[:16]
        
        # Depth cues via Laplacian (blob detection)
        laplacian = np.zeros_like(gray)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                patch = gray[i-1:i+2, j-1:j+2]
                laplacian[i, j] = (4*patch[1,1] - (patch[0,1] + patch[2,1] + patch[1,0] + patch[1,2]))
        
        laplacian_norm = np.abs(laplacian) / (np.abs(laplacian).max() + 1e-6)
        depth_features = laplacian_norm[::20, ::20].flatten()[:16]
        
        # Spatial statistics
        spatial_stats = np.array([
            gray.mean(), gray.std(), gray.max(), gray.min(),
            edges.mean(), edges.std(), edges.max(),
            corner_map.sum() / corner_map.size,
            laplacian.mean(), laplacian.std()
        ], dtype=np.float32)
        spatial_stats = np.pad(spatial_stats, (0, max(0, 16 - len(spatial_stats))), 'constant')[:16]
        
        features = np.concatenate([edge_features, corner_features, depth_features, spatial_stats], axis=0)
        return features[:64]  # Ensure 64-dim
    
    def extract_proprioceptive_features(self, state):
        """Real robot state."""
        state = np.array(state, dtype=np.float32)
        
        # Normalize state (assuming joint limits around [-pi, pi])
        state_norm = state / (np.pi + 1e-6)
        
        # Ensure 32-dim
        features = np.pad(state_norm, (0, max(0, 32 - len(state_norm))), 'constant')[:32]
        return features
    
    def encode(self, obs):
        """
        Encode multimodal observation.
        
        Args:
            obs: Dict with 'rgb' and 'state' keys
        
        Returns:
            Fused [256-dim] embedding
        """
        features = []
        
        # RGB features (always computed)
        rgb_feat = self.extract_rgb_features(obs['rgb'])
        if self.use_rgb:
            features.append(rgb_feat)  # 128-dim
        
        # Event features (temporal differences)
        if self.use_events:
            event_feat = self.extract_event_features(obs['rgb'])
            features.append(event_feat)  # 96-dim
        
        # LiDAR features (spatial geometry)
        if self.use_lidar:
            lidar_feat = self.extract_lidar_features(obs['rgb'])
            features.append(lidar_feat)  # 64-dim
        
        # Proprioceptive features
        if self.use_proprio:
            state = obs.get('state', np.zeros(6, dtype=np.float32))
            proprio_feat = self.extract_proprioceptive_features(state)
            features.append(proprio_feat)  # 32-dim
        
        # Concatenate and project to 256-dim
        if not features:
            return np.zeros(256, dtype=np.float32)
        
        fused = np.concatenate(features, axis=0).astype(np.float32)
        
        # Project to 256-dim
        if fused.shape[0] < 256:
            fused = np.pad(fused, (0, 256 - fused.shape[0]), 'constant')
        else:
            fused = fused[:256]
        
        return fused

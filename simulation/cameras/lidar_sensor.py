"""
LiDAR sensor processing for xArm 4-DOF environment.

Converts MuJoCo rangefinder data to structured point cloud and features.
Used for obstacle avoidance, environment sensing, and sensor fusion.

Reference: tech spec §6 (Sensor Fusion - LiDAR)
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LiDARProcessor:
    """
    Process LiDAR rangefinder data from MuJoCo.
    
    The xArm 4-DOF MJCF has 32 rangefinders arranged in a dome pattern:
    - 8 horizontal rays (0-45-90-135-180-225-270-315 degrees)
    - 4 vertical layers (0°, 15°, 30°, 45° elevation)
    
    Output: 32-dim feature vector with normalized distances and statistics.
    """
    
    def __init__(
        self,
        n_rangefinders: int = 32,
        max_range: float = 2.0,
        min_range: float = 0.01,
    ):
        """
        Initialize LiDAR processor.
        
        Args:
            n_rangefinders: Number of physical rangefinders (typically 32)
            max_range: Maximum sensing range in meters
            min_range: Minimum sensing range (to filter reflections)
        """
        self.n_rangefinders = n_rangefinders
        self.max_range = max_range
        self.min_range = min_range
        
        logger.info(
            f"LiDAR processor initialized: {n_rangefinders} rays, "
            f"range=[{min_range}, {max_range}]m"
        )
    
    # ────────────────────────────────────────────────────────────────────────
    # Raw Processing
    # ────────────────────────────────────────────────────────────────────────
    
    def readings_to_features(self, readings: np.ndarray) -> np.ndarray:
        """
        Convert raw rangefinder readings to normalized feature vector.
        
        Processing:
        1. Clip to [min_range, max_range]
        2. Normalize to [0, 1]
        3. Compute statistics (min, max, mean per quadrant)
        
        Args:
            readings: [32] raw rangefinder distances (meters, -1 = no hit)
        
        Returns:
            [32] normalized feature vector [0, 1]
        """
        assert readings.shape == (self.n_rangefinders,), \
            f"Expected shape ({self.n_rangefinders},), got {readings.shape}"
        
        # Handle no-hits (-1)
        valid = readings >= 0
        clipped = readings.copy()
        clipped[~valid] = self.max_range  # Treat no-hit as far away
        
        # Clip to range
        clipped = np.clip(clipped, self.min_range, self.max_range)
        
        # Normalize to [0, 1]
        normalized = (clipped - self.min_range) / (self.max_range - self.min_range)
        
        return normalized
    
    def readings_to_pointcloud(self, readings: np.ndarray) -> np.ndarray:
        """
        Convert rangefinder readings to 3D point cloud (approximate).
        
        Assumes rangefinders are arranged as a dome:
        - 8 horizontal directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        - 4 elevation angles: 0°, 15°, 30°, 45°
        
        Args:
            readings: [32] rangefinder distances
        
        Returns:
            [32, 3] 3D points in camera/base frame
        """
        assert readings.shape == (self.n_rangefinders,), \
            f"Expected shape ({self.n_rangefinders},), got {readings.shape}"
        
        points = np.zeros((self.n_rangefinders, 3))
        
        # Ray directions (approximate dome arrangement)
        azimuths = np.array([0, 45, 90, 135, 180, 225, 270, 315]) * np.pi / 180
        elevations = np.array([0, 15, 30, 45]) * np.pi / 180
        
        idx = 0
        for elev in elevations:
            for azim in azimuths:
                if idx >= self.n_rangefinders:
                    break
                
                r = readings[idx]
                if r < 0:  # No hit
                    r = self.max_range
                
                # Spherical to Cartesian
                x = r * np.cos(elev) * np.cos(azim)
                y = r * np.cos(elev) * np.sin(azim)
                z = r * np.sin(elev)
                
                points[idx] = [x, y, z]
                idx += 1
        
        return points
    
    # ────────────────────────────────────────────────────────────────────────
    # Statistics & Analysis
    # ────────────────────────────────────────────────────────────────────────
    
    def compute_statistics(self, readings: np.ndarray) -> dict:
        """
        Compute LiDAR statistics for diagnostics.
        
        Args:
            readings: [32] rangefinder distances
        
        Returns:
            dict with min_dist, max_dist, mean_dist, num_hits
        """
        valid = readings >= 0
        n_hits = np.sum(valid)
        
        if n_hits > 0:
            valid_readings = readings[valid]
            stats = {
                "min_dist": float(np.min(valid_readings)),
                "max_dist": float(np.max(valid_readings)),
                "mean_dist": float(np.mean(valid_readings)),
                "num_hits": int(n_hits),
                "hit_rate": float(n_hits / self.n_rangefinders),
            }
        else:
            stats = {
                "min_dist": float(self.max_range),
                "max_dist": float(self.max_range),
                "mean_dist": float(self.max_range),
                "num_hits": 0,
                "hit_rate": 0.0,
            }
        
        return stats
    
    def get_nearest_obstacle(self, readings: np.ndarray) -> Tuple[float, int]:
        """
        Find closest obstacle distance and ray index.
        
        Args:
            readings: [32] rangefinder distances
        
        Returns:
            (distance, ray_index) tuple
        """
        valid = readings >= 0
        if not np.any(valid):
            return self.max_range, -1
        
        valid_readings = readings.copy()
        valid_readings[~valid] = self.max_range
        
        min_idx = np.argmin(valid_readings)
        min_dist = valid_readings[min_idx]
        
        return float(min_dist), int(min_idx)


class LiDAREnvironmentMap:
    """
    Aggregate multiple LiDAR scans into occupancy grid or distance map.
    
    Used for spatial awareness and collision checking.
    """
    
    def __init__(
        self,
        grid_size: float = 1.0,
        grid_resolution: int = 64,
        max_range: float = 2.0,
    ):
        """
        Initialize environment map.
        
        Args:
            grid_size: Physical size of map (m)
            grid_resolution: Grid resolution (cells per side)
            max_range: LiDAR max range
        """
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.cell_size = grid_size / grid_resolution
        self.max_range = max_range
        
        # Occupancy grid: 0=free, 1=occupied
        self.occupancy = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        
        logger.info(
            f"LiDAR environment map initialized: {grid_resolution}×{grid_resolution}, "
            f"cell_size={self.cell_size:.4f}m"
        )
    
    def update(self, readings: np.ndarray, pose: Optional[np.ndarray] = None):
        """
        Update map with new LiDAR scan.
        
        Args:
            readings: [32] rangefinder distances
            pose: [x, y, theta] sensor pose (default: origin)
        """
        if pose is None:
            pose = np.array([0.0, 0.0, 0.0])
        
        processor = LiDARProcessor(n_rangefinders=len(readings))
        points = processor.readings_to_pointcloud(readings)
        
        # Transform points to global frame
        x_s, y_s, theta = pose
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        for point in points:
            # Rotate and translate
            x_local = point[0]
            y_local = point[1]
            
            x_global = x_s + cos_t * x_local - sin_t * y_local
            y_global = y_s + sin_t * x_local + cos_t * y_local
            
            # Convert to grid coordinates
            grid_x = int((x_global + self.grid_size / 2) / self.cell_size)
            grid_y = int((y_global + self.grid_size / 2) / self.cell_size)
            
            # Mark occupied if within bounds
            if 0 <= grid_x < self.grid_resolution and 0 <= grid_y < self.grid_resolution:
                self.occupancy[grid_y, grid_x] = 1.0
    
    def reset(self):
        """Clear occupancy grid."""
        self.occupancy.fill(0.0)

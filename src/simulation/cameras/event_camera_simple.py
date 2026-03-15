"""
Event camera simulator for Phase 13 ablation study.

Simple frame-difference based event generation.
"""

import numpy as np
from typing import List, Tuple


class EventCameraSimulator:
    """Simulates event camera by computing frame differences."""
    
    def __init__(self, threshold: float = 0.15, temporal_bins: int = 5, h: int = 84, w: int = 84):
        """
        Initialize event camera.
        
        Args:
            threshold: Per-pixel contrast threshold for event generation [0-1]
            temporal_bins: Number of temporal bins for voxel grid
            h: Height of output voxel grid
            w: Width of output voxel grid
        """
        self.threshold = threshold
        self.temporal_bins = temporal_bins
        self.h = h
        self.w = w
        self.frame_buffer = []  # Keep track of recent frames
        self.max_buffer = temporal_bins
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame and return event voxel grid based on frame differences.
        
        Args:
            frame: [H, W, 3] uint8 RGB frame
        
        Returns:
            [temporal_bins, H, W] float32 event voxel grid
        """
        # Normalize frame
        if frame.dtype == np.uint8:
            frame_norm = frame.astype(np.float32) / 255.0
        else:
            frame_norm = frame.astype(np.float32)
        
        # Simple resize using numpy (no scipy dependency)
        if frame_norm.shape[0] != self.h or frame_norm.shape[1] != self.w:
            # Use simple interpolation via indexing
            h_indices = (np.arange(self.h) * frame_norm.shape[0] / self.h).astype(int)
            w_indices = (np.arange(self.w) * frame_norm.shape[1] / self.w).astype(int)
            frame_norm = frame_norm[np.ix_(h_indices, w_indices, [0, 1, 2])]
        
        self.frame_buffer.append(frame_norm)
        if len(self.frame_buffer) > self.max_buffer + 1:
            self.frame_buffer.pop(0)
        
        # Compute event voxel grid from frame differences
        events = np.zeros((self.temporal_bins, self.h, self.w), dtype=np.float32)
        
        for t in range(1, min(len(self.frame_buffer), self.temporal_bins + 1)):
            prev_frame = self.frame_buffer[-t-1]
            curr_frame = self.frame_buffer[-t]
            
            # Compute per-channel differences
            diff = np.abs(curr_frame - prev_frame).mean(axis=2)
            
            # Threshold to binary events, then scale by magnitude
            event_mask = diff > self.threshold
            event_strength = diff * event_mask
            
            # Store in appropriate bin
            bin_idx = min(t - 1, self.temporal_bins - 1)
            events[bin_idx] = np.maximum(events[bin_idx], event_strength)
        
        return events
    
    def frames_to_events(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Convert sequence of RGB frames to event voxel grid.
        
        Args:
            frames: List of [H, W, 3] uint8 RGB frames
        
        Returns:
            [temporal_bins, H, W] float32 event voxel grid
        """
        if len(frames) == 0:
            return np.zeros((self.temporal_bins, self.h, self.w), dtype=np.float32)
        
        # Process last frame to get full event voxel
        self.frame_buffer.clear()
        self.frame_buffer.extend(frames[-min(len(frames), self.max_buffer + 1):])
        
        return self.process_frame(frames[-1])
    
    def reset(self):
        """Reset internal state."""
        self.frame_buffer = []


class LiDARSimulator:
    """Simulates LiDAR by aggregating sensor data."""
    
    def __init__(self, num_rays: int = 32):
        """
        Initialize LiDAR simulator.
        
        Args:
            num_rays: Number of rangefinder rays
        """
        self.num_rays = num_rays
    
    def query_ranges(self, ranges: np.ndarray) -> np.ndarray:
        """
        Process rangefinder measurements.
        
        Args:
            ranges: [num_rays] distance measurements
        
        Returns:
            [35] feature vector (32 ranges + 3 stats)
        """
        ranges = np.clip(ranges, 0, 1.0)
        
        # Add simple statistics
        features = np.concatenate([
            ranges,
            np.array([ranges.mean(), ranges.std(), ranges.max()], dtype=np.float32)
        ])
        
        return features[:35].astype(np.float32)

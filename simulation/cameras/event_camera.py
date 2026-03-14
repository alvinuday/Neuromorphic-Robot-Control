"""
Event camera simulation via DVS model (log-intensity differencing).

Simulates asynchronous event-based vision sensor suitable for
high-speed, dynamic scenes and temporal encoding in neural networks.

Reference: v2e framework + tech spec §6 (Sensor Fusion)
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Single event: (x, y, t, polarity)."""
    x: int
    y: int
    t: float
    polarity: int  # +1 or -1


class EventCameraSimulator:
    """
    Simulates DVS (Dynamic Vision Sensor) via log-intensity differencing.
    
    Model:
    - Pixel log-contrast: log(I_new) - log(I_old)
    - Threshold crossing: |log-contrast| > threshold triggers event
    - Polarity: +1 (increase), -1 (decrease)
    
    Used for high-speed control and temporal encoding.
    """
    
    def __init__(
        self,
        height: int = 84,
        width: int = 84,
        threshold: float = 0.3,
        tau_decay: float = 0.1,
    ):
        """
        Initialize event camera simulator.
        
        Args:
            height: Image height (pixels)
            width: Image width (pixels)
            threshold: Log-intensity threshold for triggering event
            tau_decay: Exponential decay constant for pixel memories (seconds)
        """
        self.height = height
        self.width = width
        self.threshold = threshold
        self.tau_decay = tau_decay
        
        # Pixel memory: stores log-intensity of last frame
        self.log_intensity_mem = np.zeros((height, width), dtype=np.float32)
        
        # Event buffer
        self.events: List[Event] = []
        
        # Timestamp tracking
        self.last_frame_t = 0.0
        
        logger.info(
            f"EventCameraSimulator initialized: {height}×{width}, "
            f"threshold={threshold:.3f}, tau={tau_decay:.3f}s"
        )
    
    def on_frame(self, rgb: np.ndarray, t: float) -> List[Event]:
        """
        Process new RGB frame, generate events.
        
        Args:
            rgb: [H, W, 3] uint8 RGB image
            t: Timestamp in seconds
        
        Returns:
            List of events triggered since last frame
        """
        assert rgb.shape == (self.height, self.width, 3), \
            f"Expected ({self.height}, {self.width}, 3), got {rgb.shape}"
        assert rgb.dtype == np.uint8
        
        # Convert to grayscale
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        
        # Clamp to avoid log(0)
        gray = np.clip(gray, 1.0, 255.0)
        
        # Compute log intensity
        log_I = np.log(gray)
        
        # Log-contrast with memory decay
        dt = t - self.last_frame_t if self.last_frame_t > 0 else 0.0
        decay_factor = np.exp(-dt / self.tau_decay) if dt > 0 else 1.0
        
        log_contrast = log_I - (self.log_intensity_mem * decay_factor)
        
        # Detect events: threshold crossing
        events_pos = log_contrast > self.threshold
        events_neg = log_contrast < -self.threshold
        
        # Generate event list
        self.events = []
        
        # Positive events (brightness increase)
        y_pos, x_pos = np.where(events_pos)
        for x, y in zip(x_pos, y_pos):
            self.events.append(Event(x=int(x), y=int(y), t=t, polarity=1))
        
        # Negative events (brightness decrease)
        y_neg, x_neg = np.where(events_neg)
        for x, y in zip(x_neg, y_neg):
            self.events.append(Event(x=int(x), y=int(y), t=t, polarity=-1))
        
        # Update memory
        self.log_intensity_mem = log_I.copy()
        self.last_frame_t = t
        
        return self.events
    
    def to_voxel_grid(
        self,
        events: Optional[List[Event]] = None,
        time_bins: int = 5,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert events to voxel grid: [time_bins, H, W].
        
        Voxel (t, y, x) is +1 if positive event, -1 if negative event,
        0 if no event. Time bins are equally spaced.
        
        Args:
            events: Event list (default: use last frame's events)
            time_bins: Number of temporal bins
            normalize: Normalize to [-1, 1]
        
        Returns:
            [time_bins, H, W] voxel grid (int8 or float32)
        """
        if events is None:
            events = self.events
        
        if len(events) == 0:
            return np.zeros((time_bins, self.height, self.width), dtype=np.int8)
        
        # Time range
        t_min = min(e.t for e in events)
        t_max = max(e.t for e in events)
        t_range = t_max - t_min if t_max > t_min else 1e-6
        
        # Initialize voxel grid
        voxel = np.zeros((time_bins, self.height, self.width), dtype=np.float32)
        
        # Fill voxel grid
        for event in events:
            # Time bin
            t_norm = (event.t - t_min) / t_range
            t_idx = int(t_norm * time_bins)
            t_idx = min(t_idx, time_bins - 1)
            
            # Accumulate polarity
            voxel[t_idx, event.y, event.x] += event.polarity
        
        # Clip to [-1, 1]
        if normalize:
            voxel = np.clip(voxel, -1, 1)
        
        return voxel.astype(np.int8) if normalize else voxel
    
    def get_event_rate(self) -> float:
        """
        Get instantaneous event rate (events/frame).
        
        Used for diagnostics: high rate → too sensitive, low rate → threshold too high.
        """
        return float(len(self.events))
    
    def reset(self):
        """Reset simulator state."""
        self.log_intensity_mem.fill(0.0)
        self.events.clear()
        self.last_frame_t = 0.0


class EventFrameProcessor:
    """
    Batch processor for event voxel grids.
    
    Converts stream of RGB frames into event voxels suitable for
    temporal encoding in neural networks.
    """
    
    def __init__(
        self,
        height: int = 84,
        width: int = 84,
        time_bins: int = 5,
        threshold: float = 0.3,
    ):
        self.height = height
        self.width = width
        self.time_bins = time_bins
        
        self.simulator = EventCameraSimulator(
            height=height,
            width=width,
            threshold=threshold,
        )
    
    def process_frame(self, rgb: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Process single frame to event voxel.
        
        Args:
            rgb: [H, W, 3] uint8 image
            t: Timestamp (default: 0.0)
        
        Returns:
            [time_bins, H, W] voxel grid
        """
        events = self.simulator.on_frame(rgb, t)
        voxel = self.simulator.to_voxel_grid(events, time_bins=self.time_bins)
        return voxel
    
    def process_sequence(
        self,
        rgb_sequence: np.ndarray,
        fps: float = 30.0,
    ) -> np.ndarray:
        """
        Process sequence of frames to event voxels.
        
        Args:
            rgb_sequence: [N, H, W, 3] uint8 images
            fps: Frame rate for timestamp calculation
        
        Returns:
            [N, time_bins, H, W] voxel grids
        """
        dt = 1.0 / fps
        voxels = []
        
        for i, rgb in enumerate(rgb_sequence):
            t = i * dt
            voxel = self.process_frame(rgb, t)
            voxels.append(voxel)
        
        return np.array(voxels)

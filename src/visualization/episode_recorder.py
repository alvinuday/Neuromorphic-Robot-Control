"""
Episode recorder for saving MuJoCo frames to GIF.

Records RGB frames from simulation and exports as animated GIF.
Supports frame resizing via PIL.
"""

import numpy as np
from pathlib import Path
from contextlib import contextmanager


class EpisodeRecorder:
    """Record MuJoCo RGB frames to GIF using imageio."""

    def __init__(self, fps: int = 10, resize: tuple = (640, 480)):
        """
        Initialize recorder.
        
        Args:
            fps: Frames per second for output GIF
            resize: (width, height) tuple for output frames
        """
        self.fps    = fps
        self.resize = resize
        self._frames = []

    def start(self):
        """Begin recording frames."""
        self._frames = []

    def add_frame(self, frame: np.ndarray):
        """
        Add [H, W, 3] uint8 frame.
        
        Args:
            frame: RGB frame as uint8 array
        """
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # Resize if needed
        if frame.shape[:2] != (self.resize[1], self.resize[0]):
            from PIL import Image
            pil = Image.fromarray(frame).resize(self.resize)
            frame = np.array(pil)
        self._frames.append(frame)

    def save(self, path: str) -> str:
        """
        Save recorded frames to GIF.
        
        Args:
            path: Output GIF file path
            
        Returns:
            Path to saved GIF
        """
        import imageio.v3 as iio
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Handle empty frame list
        if not self._frames:
            print(f"Warning: No frames to save to {path}")
            return path
        
        iio.imwrite(path, self._frames, loop=0, fps=self.fps)
        size_kb = Path(path).stat().st_size / 1024
        print(f"GIF saved: {path}  ({len(self._frames)} frames, {size_kb:.1f} KB)")
        return path

    @contextmanager
    def recording(self, path: str):
        """
        Context manager for recording.
        
        Usage:
            with recorder.recording("output.gif"):
                for frame in frames:
                    recorder.add_frame(frame)
        """
        self.start()
        try:
            yield self
        finally:
            self.save(path)

"""Base abstract class for environments."""
from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    """Abstract base for all simulation environments in the system."""

    @abstractmethod
    def reset(self) -> dict:
        """Reset to initial state. Returns observation dict."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        """Apply action. Returns (obs, reward, done, info)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

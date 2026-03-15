"""Base abstract class for controllers."""
from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """Abstract base for all controllers in the system."""

    @abstractmethod
    def step(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute control output. MUST be synchronous. MUST return ndarray."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset controller state."""
        pass

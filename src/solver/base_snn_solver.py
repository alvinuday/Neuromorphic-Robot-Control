
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class SNNIteration:
    t: int
    x: np.ndarray
    v: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    cost: float = 0.0
    max_viol: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

@dataclass
class SNNResult:
    status: str
    z_star: np.ndarray
    history: List[SNNIteration]
    solve_time_ms: float
    objective: float
    eq_norm: float
    ineq_viol: float
    passed: bool
    hardware_stats: Dict[str, Any] = field(default_factory=dict)

class BaseSNNSolver(ABC):
    """Abstract base class for SNN-based QP solvers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Solver name identifier."""
        pass

    @abstractmethod
    def solve(self, P: np.ndarray, q: np.ndarray, 
              A: np.ndarray, l: np.ndarray, u: np.ndarray,
              x0: Optional[np.ndarray] = None,
              **kwargs) -> SNNResult:
        """
        Solve the QP: min 1/2 x'Px + q'x s.t. l <= Ax <= u.
        
        Args:
            P: Cost Hessian (L x L)
            q: Linear cost (L)
            A: Constraint matrix (M x L)
            l: Lower bounds (M)
            u: Upper bounds (M)
            x0: Initial guess (L)
            **kwargs: Hyperparameters (alpha, beta, etc.)
        """
        pass

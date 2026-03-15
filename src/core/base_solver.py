"""Base abstract class for QP solvers."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict


class BaseQPSolver(ABC):
    """
    Abstract base for QP solvers. Standard OSQP-style interface.

    Solves:
        minimize    0.5 * x^T P x + q^T x
        subject to  l <= A x <= u

    Equality constraints are encoded as l[i] == u[i].
    """

    @abstractmethod
    def solve(
        self,
        P: np.ndarray,   # [n, n] positive semi-definite cost matrix
        q: np.ndarray,   # [n]    linear cost vector
        A: np.ndarray,   # [m, n] constraint matrix
        l: np.ndarray,   # [m]    lower bounds  (-inf for one-sided)
        u: np.ndarray,   # [m]    upper bounds  (+inf for one-sided)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Returns:
            x:    [n] optimal primal solution
            info: dict with keys:
                    solve_time_ms     (float)
                    obj_val           (float)  — 0.5 x^T P x + q^T x
                    constraint_viol   (float)  — max(0, Ax-u, l-Ax).max()
                    status            (str)    — "optimal"|"max_iter"|"error"
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name, e.g. 'StuartLandauLagrange' or 'OSQP'."""
        pass

"""xArm 6-DOF Model Predictive Controller."""
import numpy as np
from typing import Optional, Dict, Tuple
from src.core.base_controller import BaseController
from src.core.base_solver import BaseQPSolver
from src.dynamics.xarm_dynamics import XArmDynamics


class XArmMPCController(BaseController):
    """
    Single-step torque MPC for xArm 6-DOF.

    Formulation (single-step lookahead, linearized dynamics):
    
        minimize   (q_next - q_ref)^T Q (q_next - q_ref) + tau^T R tau
        subject to  q_next  = q + dt * (qdot + dt * M^-1(tau - C - G))
                   |tau_i| <= tau_max_i
    
    After substitution, this becomes a standard QP in tau.
    
    Args:
        solver:       BaseQPSolver instance (SL or OSQP)
        robot_config: dict loaded from config/robots/xarm_6dof.yaml
        dt:           control timestep (seconds)
        Q:            state cost matrix [6,6]  (default: identity)
        R:            input cost matrix [6,6]  (default: 0.01 * identity)
    """

    def __init__(
        self,
        solver: BaseQPSolver,
        robot_config: dict,
        dt: float = 0.01,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        self.solver = solver
        self.dt     = dt
        self.dynamics = XArmDynamics(robot_config)
        rc = robot_config['robot']

        self.tau_max = np.array(rc['torque_limits']['tau_max'][:6])
        self.tau_min = -self.tau_max

        n = 6
        self.Q = Q if Q is not None else np.eye(n)
        self.R = R if R is not None else 0.01 * np.eye(n)

        # For QP inspector
        self._last_qp: Dict = {}

    def reset(self) -> None:
        """Reset controller state."""
        self._last_qp = {}

    def step(self, state: Tuple, reference: np.ndarray) -> np.ndarray:
        """
        Args:
            state:     (q [6], qdot [6])
            reference: [N, 6] or [6] reference joint angles (only first row used)
        
        Returns:
            tau: [8] torques (6 arm + 2 gripper = 0)
        """
        q, qdot = state
        q_ref = reference[0] if reference.ndim == 2 else reference

        # Build QP
        P, qv, A, l, u = self._build_qp(q, qdot, q_ref)

        # Solve
        tau_arm, info = self.solver.solve(P, qv, A, l, u)

        # Cache for QP inspector
        self._last_qp = {'P': P, 'q': qv, 'A': A, 'l': l, 'u': u,
                         'solution': tau_arm, 'info': info}

        # Append zero gripper torques
        tau_full = np.append(tau_arm, [0., 0.])
        return tau_full

    def _build_qp(self, q, qdot, q_ref):
        """Build QP matrices for torque optimization."""
        M  = self.dynamics.inertia_matrix(q)     # [6,6]
        C  = self.dynamics.coriolis_vector(q, qdot)  # [6]
        G  = self.dynamics.gravity_vector(q)         # [6]
        M_inv = np.linalg.solve(M, np.eye(6))

        dt = self.dt
        # q_next = q + dt*qdot + dt^2 * M_inv @ (tau - C - G)
        # error(tau) = q_next - q_ref = const + dt^2 * M_inv @ tau - dt^2 * M_inv @ (C+G)
        #
        # cost = error^T Q error + tau^T R tau
        #      = tau^T [A_d^T Q A_d + R] tau + 2 b^T Q A_d tau  + const
        # where A_d = dt^2 * M_inv,  b = q + dt*qdot - q_ref - dt^2 * M_inv@(C+G)

        A_d   = (dt**2) * M_inv                       # [6,6]
        b     = q + dt * qdot - q_ref - A_d @ (C + G) # [6]

        P_qp  = A_d.T @ self.Q @ A_d + self.R          # [6,6] symmetric PSD
        q_qp  = A_d.T @ self.Q @ b                      # [6]

        # Symmetrize P (numerical safety)
        P_qp  = 0.5 * (P_qp + P_qp.T)

        # Box constraints on tau
        A_box = np.eye(6)
        l_box = self.tau_min
        u_box = self.tau_max

        return P_qp, q_qp, A_box, l_box, u_box

    def get_last_qp_matrices(self) -> Dict:
        """Return QP matrices for QP inspector webapp."""
        return self._last_qp

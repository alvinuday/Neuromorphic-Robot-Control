"""
DualSystemController: Orchestrate synchronous MPC + asynchronous VLA queries.

System 1 (Synchronous, 100-500 Hz): Stuart-Landau MPC control loop
System 2 (Asynchronous, 1-5 Hz): SmolVLA queries via background thread

Critical Invariant: step() is purely synchronous (< 20ms, no await, no I/O).
"""

import logging
import time
from enum import Enum
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


class ControlState(Enum):
    """Control system state machine."""
    INIT = 0              # Initialization (first observation)
    TRACKING = 1         # Following reference trajectory to subgoal
    GOAL_REACHED = 2     # At subgoal, waiting for next VLA query
    ERROR = 3            # Unrecoverable error (logged at WARNING)


class DualSystemController:
    """
    Main control interface for dual-system architecture.

    System 1 (Synchronous, ~100-500 Hz):
      - MPC control loop
      - Guaranteed timing (< 20ms per step)
      - Never waits on network I/O

    System 2 (Asynchronous, 1-5 Hz):
      - SmolVLA queries in background thread
      - Non-blocking HTTP via aiohttp
      - Updates reference trajectory via TrajectoryBuffer

    Thread safety: TrajectoryBuffer uses GIL-based atomicity for numpy ops.
    No explicit locks needed.
    """

    def __init__(
        self,
        mpc_solver,
        smolvla_client,
        trajectory_buffer,
        logger_instance: Optional[logging.Logger] = None,
        mpc_horizon_steps: int = 10,
        control_dt_s: float = 0.01,
        vla_query_interval_s: float = 0.2,
    ):
        """
        Initialize dual-system controller.

        Args:
            mpc_solver: Existing MPC solver with solve(state, ref_traj) → τ
            smolvla_client: SmolVLAClient for async VLA queries
            trajectory_buffer: TrajectoryBuffer for interpolated reference
            logger_instance: Logger instance (default: module logger)
            mpc_horizon_steps: MPC horizon N (default 10)
            control_dt_s: Control timestep (default 10ms → 100 Hz)
            vla_query_interval_s: VLA query interval (default 200ms → 5 Hz)
        """
        self.mpc_solver = mpc_solver
        self.vla_client = smolvla_client
        self.trajectory_buffer = trajectory_buffer
        self.logger = logger_instance or logging.getLogger(__name__)

        # Timing
        self.mpc_horizon = mpc_horizon_steps
        self.dt = control_dt_s
        self.vla_interval = vla_query_interval_s

        # State machine
        self.state = ControlState.INIT
        self._last_state = None

        # Observation tracking
        self.q_current = None
        self.qdot_current = None
        self.rgb_current = None
        self.instruction = None

        # Timing instrumentation
        self.step_count = 0
        self.step_times_ms = []
        self.last_vla_query_time = 0.0

        # Safety
        self.running = True

    def step(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        rgb: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """
        Main synchronous control loop step.

        **CRITICAL:** This function NEVER blocks, NEVER calls await,
        NEVER touches network I/O. Must complete in < 20ms.

        Args:
            q: Current joint angles [3] (rad)
            qdot: Current joint velocities [3] (rad/s)
            rgb: Current RGB frame [H, W, 3] uint8 (batched if needed)
            instruction: Task instruction string ("pick up", "place", etc.)

        Returns:
            tau: Optimal torque command [3] (N·m)

        Raises:
            Nothing — all errors logged at WARNING level, fallback returned
        """
        t0 = time.perf_counter()

        try:
            # 1. Update internal state
            self.q_current = q.copy()
            self.qdot_current = qdot.copy()
            self.rgb_current = rgb.copy() if isinstance(rgb, np.ndarray) else rgb
            self.instruction = instruction

            # 2. Transition from INIT to TRACKING on first step
            if self.state == ControlState.INIT:
                self.state = ControlState.TRACKING
                self.logger.info(
                    f"[DualSystemController] Transitioning to TRACKING on first step"
                )

            # 3. Check goal arrival (updates state machine if needed)
            self._check_goal_arrival(q)

            # 4. Get reference trajectory from buffer (instant, thread-safe)
            N = self.mpc_horizon
            q_ref, qdot_ref = self.trajectory_buffer.get_reference_trajectory(
                q, N=N, dt=self.dt
            )

            # 5. Prepare MPC state
            x_curr = np.concatenate([q, qdot])  # [6]
            x_ref = q_ref[-1]  # Terminal reference (last point of horizon)

            # 6. Run MPC solver
            # Input: current state, reference trajectory (or hold trajectory)
            # Output: optimal torques for next step
            tau = self.mpc_solver.solve(
                x_curr=x_curr,
                x_ref=x_ref,
                q_ref=q_ref,  # Full reference trajectory for cost
                qdot_ref=qdot_ref,  # Velocity reference
            )

            # Ensure tau is [3] numpy array
            if not isinstance(tau, np.ndarray):
                tau = np.array(tau)
            if tau.shape != (3,):
                tau = tau.flatten()[:3]

            # 7. State machine transitions (for logging & diagnostics)
            self._update_state_machine()

            # 8. Measure timing
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            self.step_times_ms.append(elapsed)
            if elapsed > 20:
                self.logger.warning(
                    f"[DualSystemController] Step {self.step_count} took {elapsed:.1f}ms "
                    f"(target < 20ms)"
                )

            self.step_count += 1
            return tau

        except Exception as e:
            # Graceful fallback: return zero torque (hold position)
            self.logger.warning(
                f"[DualSystemController] Error in step: {e}. "
                f"Returning zero torque (hold position)."
            )
            self.state = ControlState.ERROR
            self.step_count += 1
            return np.zeros(3)

    def _check_goal_arrival(self, q_current: np.ndarray):
        """
        Check if current position has reached the subgoal.

        Updates self.state if arrival detected.
        Uses TrajectoryBuffer.check_arrival() for hysteresis logic.

        Args:
            q_current: Current joint angles [3]
        """
        if self.trajectory_buffer.is_goal_reached(q_current):
            if self.state == ControlState.TRACKING:
                self.state = ControlState.GOAL_REACHED
                self.logger.info(
                    f"[DualSystemController] Goal reached at step {self.step_count}. "
                    f"Waiting for next subgoal."
                )

    def _update_state_machine(self):
        """
        Log state transitions for debugging.

        Called after every step.
        """
        if self.state != self._last_state:
            self.logger.info(
                f"[DualSystemController] State transition: "
                f"{self._last_state} → {self.state} at step {self.step_count}"
            )
            self._last_state = self.state

    def get_stats(self) -> dict:
        """
        Return controller statistics.

        Returns:
            dict: Statistics including step count, state, timing metrics
        """
        stats = {
            "step_count": self.step_count,
            "state": self.state.name,
            "step_time_mean_ms": np.mean(self.step_times_ms)
            if self.step_times_ms
            else 0,
            "step_time_max_ms": np.max(self.step_times_ms)
            if self.step_times_ms
            else 0,
            "step_time_p95_ms": np.percentile(self.step_times_ms, 95)
            if len(self.step_times_ms) > 20
            else 0,
        }
        return stats

    def reset(self):
        """
        Reset controller to initial state.

        Clears all counters, timing history, and state machine.
        """
        self.state = ControlState.INIT
        self._last_state = None
        self.step_count = 0
        self.step_times_ms = []
        self.trajectory_buffer.reset()
        self.logger.info("[DualSystemController] Reset to INIT state")

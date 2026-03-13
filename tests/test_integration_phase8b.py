"""
Integration & E2E Tests for Phase 8B (Task 5).

5 major integration tests covering full dual-system operation:
  1. Mock VLA with controlled latency
  2. Timing variance validation (MPC not blocked by VLA)
  3. Graceful fallback on VLA timeout
  4. Full pointing task with multiple subgoals
  5. Stress test: 100+ steps with continuous polling
"""

import time
from unittest.mock import MagicMock
from queue import Queue

import numpy as np
import pytest

from src.integration import (
    DualSystemController,
    VLAQueryThread,
    ControlState,
)
from src.smolvla_client import SmolVLAClient, SmolVLAResponse, TrajectoryBuffer


# ============================================================================
# Fixtures: Mock Components
# ============================================================================


@pytest.fixture
def mock_mpc_solver():
    """Mock MPC solver that returns reasonable torques."""
    solver = MagicMock()

    def mock_solve(x_curr, x_ref, q_ref=None, qdot_ref=None):
        # Simple proportional control
        q_error = x_curr[:3] - x_ref
        return -q_error * 2  # PD-like feedback

    solver.solve = mock_solve
    return solver


@pytest.fixture
def mock_vla_client_with_latency():
    """Mock VLA client with controlled latencies."""
    client = MagicMock()
    query_delay_ms = 100  # Simulated latency

    async def mock_query(rgb, instruction, current_joints=None):
        import asyncio

        await asyncio.sleep(query_delay_ms / 1000)
        # Return random goal in workspace
        q_goal = np.array([0.1, 0.3, -0.2], dtype=np.float32)
        return SmolVLAResponse(
            action_chunk=np.tile(q_goal, (1, 7 // 3 + 1))[:, :7].astype(np.float32),
            subgoal_xyz=np.array([0.2, 0.3, 0.1]),
            latency_ms=query_delay_ms,
            timestamp=time.time(),
        )

    client.query_action = mock_query
    return client


# ============================================================================
# Test 1: Mock VLA with Controlled Latency
# ============================================================================


def test_mock_vla_returns_valid_response(mock_vla_client_with_latency):
    """Mock VLA client returns valid responses."""
    import asyncio

    async def run_query():
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        response = await mock_vla_client_with_latency.query_action(
            rgb=rgb, instruction="test", current_joints=None
        )
        return response

    response = asyncio.run(run_query())
    assert response is not None
    assert response.action_chunk.shape[0] >= 1
    assert response.subgoal_xyz.shape == (3,)


# ============================================================================
# Test 2: MPC Timing Not Affected by VLA Latency
# ============================================================================


def test_mpc_timing_unaffected_by_vla(
    mock_mpc_solver, mock_vla_client_with_latency
):
    """Main MPC loop timing is not affected by VLA latency (non-blocking)."""
    trajectory_buffer = TrajectoryBuffer()
    controller = DualSystemController(
        mpc_solver=mock_mpc_solver,
        smolvla_client=mock_vla_client_with_latency,
        trajectory_buffer=trajectory_buffer,
    )

    # Set initial subgoal
    trajectory_buffer.update_subgoal(np.array([0.1, 0.3, -0.2]))

    # Measure MPC step timing
    q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qdot = np.zeros(3, dtype=np.float32)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    mpc_timings = []
    for _ in range(10):
        t0 = time.perf_counter()
        tau = controller.step(q, qdot, rgb, "test")
        elapsed = (time.perf_counter() - t0) * 1000
        mpc_timings.append(elapsed)

    mean_timing = np.mean(mpc_timings)
    assert (
        mean_timing < 15
    ), f"MPC timing {mean_timing:.1f}ms too slow (should be < 15ms)"


def test_vla_thread_doesnt_block_mpc(mock_mpc_solver, mock_vla_client_with_latency):
    """Background VLA thread doesn't block MPC loop."""
    trajectory_buffer = TrajectoryBuffer()
    controller = DualSystemController(
        mpc_solver=mock_mpc_solver,
        smolvla_client=mock_vla_client_with_latency,
        trajectory_buffer=trajectory_buffer,
    )

    # Set initial subgoal
    trajectory_buffer.update_subgoal(np.array([0.1, 0.3, -0.2]))

    # Start VLA thread
    def rgb_source():
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    thread = VLAQueryThread(
        smolvla_client=mock_vla_client_with_latency,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.2,
    )
    thread.start(rgb_source, "reach target")

    # Measure MPC timing with VLA polling active
    q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qdot = np.zeros(3, dtype=np.float32)
    rgb = rgb_source()

    mpc_timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        tau = controller.step(q, qdot, rgb, "test")
        elapsed = (time.perf_counter() - t0) * 1000
        mpc_timings.append(elapsed)

    thread.stop()

    mean_timing = np.mean(mpc_timings)
    max_timing = np.max(mpc_timings)

    assert mean_timing < 15, f"MPC mean timing {mean_timing:.1f}ms > 15ms"
    assert max_timing < 30, f"MPC max timing {max_timing:.1f}ms > 30ms"


# ============================================================================
# Test 3: Graceful Fallback on VLA Timeout
# ============================================================================


def test_graceful_fallback_on_vla_timeout(mock_mpc_solver):
    """System continues with last trajectory if VLA times out."""
    trajectory_buffer = TrajectoryBuffer()

    # Mock VLA that times out
    async def timeout_query(rgb, instruction, current_joints=None):
        import asyncio

        await asyncio.sleep(10)  # Never returns
        return None

    mock_vla_timeout = MagicMock()
    mock_vla_timeout.query_action = timeout_query

    controller = DualSystemController(
        mpc_solver=mock_mpc_solver,
        smolvla_client=mock_vla_timeout,
        trajectory_buffer=trajectory_buffer,
    )

    # Set initial subgoal manually (VLA would do this normally)
    trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))

    q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qdot = np.zeros(3, dtype=np.float32)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Start VLA thread (will timeout)
    def rgb_source():
        return rgb

    thread = VLAQueryThread(
        smolvla_client=mock_vla_timeout,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.05,
        query_timeout_s=0.1,  # Short timeout
    )
    thread.start(rgb_source, "reach target")

    # MPC should continue working despite VLA timeout
    success_steps = 0
    for i in range(20):
        try:
            tau = controller.step(q + 0.001 * i, qdot, rgb, "test")
            if tau is not None:
                success_steps += 1
        except Exception as e:
            pytest.fail(f"MPC step failed: {e}")

    thread.stop()

    assert success_steps > 15, f"Too many failures: {success_steps}/20 successful"


# ============================================================================
# Test 4: Full Pointing Task with Multiple Subgoals
# ============================================================================


def test_full_pointing_task_e2e(
    mock_mpc_solver, mock_vla_client_with_latency
):
    """Full E2E test: multiple subgoals, state transitions, reaching."""
    trajectory_buffer = TrajectoryBuffer()
    controller = DualSystemController(
        mpc_solver=mock_mpc_solver,
        smolvla_client=mock_vla_client_with_latency,
        trajectory_buffer=trajectory_buffer,
    )

    q_initial = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    q_target_1 = np.array([0.1, 0.3, -0.2], dtype=np.float32)
    q_target_2 = np.array([-0.1, 0.2, 0.1], dtype=np.float32)

    qdot = np.zeros(3, dtype=np.float32)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Phase 1: Move to first target
    trajectory_buffer.update_subgoal(q_target_1)
    assert controller.state == ControlState.INIT or controller.state == ControlState.TRACKING

    step_count = 0
    for i in range(50):
        # Simulate moving toward target
        q = q_initial + (q_target_1 - q_initial) * min((i + 1) / 50, 1.0)
        tau = controller.step(q, qdot, rgb, "reach target 1")
        step_count += 1

        if controller.state == ControlState.GOAL_REACHED:
            break

    assert (
        step_count < 50 or controller.state in (ControlState.TRACKING, ControlState.GOAL_REACHED)
    )

    # Phase 2: Move to second target
    trajectory_buffer.update_subgoal(q_target_2)

    for i in range(50):
        q = q_target_1 + (q_target_2 - q_target_1) * min((i + 1) / 50, 1.0)
        tau = controller.step(q, qdot, rgb, "reach target 2")

        if controller.state == ControlState.GOAL_REACHED:
            break

    # Should have executed many steps across multiple subgoals
    assert controller.step_count >= 45  # Allow some variance in step count


# ============================================================================
# Test 5: Stress Test - 100+ Steps with Continuous Polling
# ============================================================================


def test_stress_test_continuous_polling(
    mock_mpc_solver, mock_vla_client_with_latency
):
    """Stress test: 100+ MPC steps with concurrent VLA polling."""
    trajectory_buffer = TrajectoryBuffer()
    controller = DualSystemController(
        mpc_solver=mock_mpc_solver,
        smolvla_client=mock_vla_client_with_latency,
        trajectory_buffer=trajectory_buffer,
    )

    # Set initial subgoal
    trajectory_buffer.update_subgoal(np.array([0.1, 0.3, -0.2]))

    # Start VLA polling thread
    def rgb_source():
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    thread = VLAQueryThread(
        smolvla_client=mock_vla_client_with_latency,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.1,
    )
    thread.start(rgb_source, "reach target")

    # Run 150 MPC steps
    q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qdot = np.zeros(3, dtype=np.float32)
    rgb = rgb_source()

    failed_steps = 0
    for i in range(150):
        try:
            q_i = q + np.array([0.001 * i, 0.001 * i, -0.001 * i], dtype=np.float32)
            tau = controller.step(q_i, qdot, rgb, "stress test")
            assert isinstance(tau, np.ndarray)
            assert tau.shape == (3,)
        except Exception as e:
            failed_steps += 1

    thread.stop()

    # Should complete almost all steps successfully
    assert (
        failed_steps < 10
    ), f"Too many failures: {failed_steps}/150 failed steps"

    # Check final statistics
    stats = controller.get_stats()
    assert stats["step_count"] == 150
    assert stats["step_time_mean_ms"] > 0

    thread_stats = thread.get_stats()
    assert thread_stats["query_count"] > 0  # At least one VLA query


# ============================================================================
# Test 6: Trajectory Buffer Consistency Under Concurrent Access
# ============================================================================


def test_trajectory_buffer_thread_safety(mock_vla_client_with_latency):
    """TrajectoryBuffer handles concurrent read/write safely."""
    trajectory_buffer = TrajectoryBuffer()

    # Start VLA thread writing to buffer
    def rgb_source():
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    thread = VLAQueryThread(
        smolvla_client=mock_vla_client_with_latency,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.05,
    )
    thread.start(rgb_source, "test")

    # Continuously read from buffer (simulating MPC loop)
    read_count = 0
    for _ in range(100):
        q_current = np.array([0.1, 0.2, 0.15], dtype=np.float32)
        q_ref, qdot_ref = trajectory_buffer.get_reference_trajectory(
            q_current, N=10, dt=0.01
        )

        assert q_ref.shape == (10, 3)
        assert qdot_ref.shape == (10, 3)
        assert np.all(np.isfinite(q_ref))
        assert np.all(np.isfinite(qdot_ref))
        read_count += 1

        time.sleep(0.001)  # Small delay

    thread.stop()

    assert read_count == 100, "All reads should succeed"


# ============================================================================
# Test 7: State Machine Consistency Over Time
# ============================================================================


def test_state_machine_consistency(mock_mpc_solver):
    """State machine remains consistent through full operation."""
    trajectory_buffer = TrajectoryBuffer()
    controller = DualSystemController(
        mpc_solver=mock_mpc_solver,
        smolvla_client=MagicMock(),
        trajectory_buffer=trajectory_buffer,
    )

    # Track state transitions
    state_history = []

    q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qdot = np.zeros(3, dtype=np.float32)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Set initial subgoal
    trajectory_buffer.update_subgoal(np.array([0.1, 0.2, -0.1]))

    for i in range(50):
        state_history.append(controller.state)
        tau = controller.step(q, qdot, rgb, "test")

    # Check state machine consistency
    # Should transition from INIT to TRACKING
    assert (
        ControlState.INIT in state_history
    ), "Should start in INIT state"
    assert (
        ControlState.TRACKING in state_history
    ), "Should transition to TRACKING"

    # No forbidden transitions (e.g., TRACKING → INIT)
    for i in range(len(state_history) - 1):
        current = state_history[i]
        next_state = state_history[i + 1]

        # INIT can only go to TRACKING or ERROR
        if current == ControlState.INIT:
            assert next_state in (
                ControlState.INIT,
                ControlState.TRACKING,
                ControlState.ERROR,
            )

        # TRACKING can go to GOAL_REACHED or ERROR
        if current == ControlState.TRACKING:
            assert next_state in (
                ControlState.TRACKING,
                ControlState.GOAL_REACHED,
                ControlState.ERROR,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

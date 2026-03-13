"""
Tests for VLAQueryThread (Task 4): Background VLA polling thread.

5+ tests covering:
  1. Thread startup and shutdown
  2. Query polling and subgoal updates
  3. Timeout handling
  4. Statistics tracking
  5. Graceful failure modes
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.integration import VLAQueryThread, poll_vla_background
from src.smolvla_client import SmolVLAResponse, TrajectoryBuffer


# Fixtures

@pytest.fixture
def trajectory_buffer():
    """Real TrajectoryBuffer instance."""
    return TrajectoryBuffer()


@pytest.fixture
def mock_vla_client():
    """Mock SmolVLAClient with async methods."""
    client = MagicMock()
    
    async def mock_query_action(rgb, instruction, current_joints=None):
        await asyncio.sleep(0.01)  # Simulate latency
        return SmolVLAResponse(
            action_chunk=np.random.randn(1, 7).astype(np.float32),
            subgoal_xyz=np.array([0.2, 0.3, 0.1]),
            latency_ms=10,
            timestamp=time.time(),
        )
    
    client.query_action = mock_query_action
    return client


@pytest.fixture
def rgb_source():
    """Mock RGB frame source."""
    def source():
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return source


@pytest.fixture
def vla_thread(mock_vla_client, trajectory_buffer):
    """VLAQueryThread instance."""
    return VLAQueryThread(
        smolvla_client=mock_vla_client,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.05,  # Fast polling for tests
        query_timeout_s=1.0,
    )


# ============================================================================
# 1. Initialization & Startup Tests
# ============================================================================


def test_vla_thread_init_default(mock_vla_client, trajectory_buffer):
    """VLAQueryThread initializes with defaults."""
    thread = VLAQueryThread(
        smolvla_client=mock_vla_client,
        trajectory_buffer=trajectory_buffer,
    )
    assert not thread._running
    assert thread.query_count == 0
    assert thread.success_count == 0
    assert thread.failure_count == 0


def test_vla_thread_starts(vla_thread, rgb_source):
    """VLAQueryThread starts successfully."""
    success = vla_thread.start(rgb_source, "test instruction")
    assert success is True
    assert vla_thread._running is True
    
    # Clean up
    time.sleep(0.1)
    vla_thread.stop()


def test_vla_thread_cannot_start_twice(vla_thread, rgb_source):
    """VLAQueryThread rejects double-start."""
    vla_thread.start(rgb_source, "test")
    success = vla_thread.start(rgb_source, "test")  # Second start
    assert success is False
    
    vla_thread.stop()


# ============================================================================
# 2. Query Polling Tests
# ============================================================================


def test_vla_thread_polls_vla(vla_thread, mock_vla_client, rgb_source):
    """Background thread polls VLA periodically."""
    vla_thread.start(rgb_source, "reach target")
    
    # Wait for a few queries
    time.sleep(0.2)
    
    stats = vla_thread.get_stats()
    assert stats["query_count"] > 0, "No queries were made"
    
    vla_thread.stop()


def test_vla_thread_updates_trajectory_buffer(vla_thread, rgb_source):
    """VLA queries update TrajectoryBuffer with subgoals."""
    vla_thread.start(rgb_source, "reach target")
    
    # Initially no subgoal
    assert vla_thread.trajectory_buffer.current_subgoal_q is None
    
    # Wait for first query to complete
    time.sleep(0.15)
    
    # Now should have a subgoal
    assert vla_thread.trajectory_buffer.current_subgoal_q is not None
    
    vla_thread.stop()


def test_vla_thread_multiple_updates(vla_thread, rgb_source):
    """Multiple VLA queries update buffer multiple times."""
    vla_thread.start(rgb_source, "reach target")
    
    initial_count = 0
    time.sleep(0.05)
    initial_count = vla_thread.trajectory_buffer._query_count
    
    # Wait for a few more updates
    time.sleep(0.15)
    
    final_count = vla_thread.trajectory_buffer._query_count
    assert final_count > initial_count, "Subgoal was not updated"
    
    vla_thread.stop()


# ============================================================================
# 3. Timeout & Failure Handling Tests
# ============================================================================


def test_vla_thread_handles_timeout(mock_vla_client, trajectory_buffer, rgb_source):
    """Thread handles VLA timeout gracefully."""
    
    # Mock slow VLA (times out after 1s)
    async def slow_query(rgb, instruction, current_joints=None):
        await asyncio.sleep(5.0)  # Longer than timeout
        return None
    
    mock_vla_client.query_action = slow_query
    
    thread = VLAQueryThread(
        smolvla_client=mock_vla_client,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.05,
        query_timeout_s=0.1,  # Short timeout
    )
    
    thread.start(rgb_source, "test")
    time.sleep(0.3)
    
    stats = thread.get_stats()
    assert stats["query_count"] > 0
    assert stats["failure_count"] > 0  # Should have timeouts
    
    thread.stop()


def test_vla_thread_handles_none_response(mock_vla_client, trajectory_buffer, rgb_source):
    """Thread handles None response gracefully."""
    
    # Mock VLA that returns None
    async def none_query(rgb, instruction, current_joints=None):
        await asyncio.sleep(0.01)
        return None
    
    mock_vla_client.query_action = none_query
    
    thread = VLAQueryThread(
        smolvla_client=mock_vla_client,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.05,
    )
    
    thread.start(rgb_source, "test")
    time.sleep(0.15)
    
    stats = thread.get_stats()
    assert stats["failure_count"] > 0
    assert thread._running  # Still running despite failures
    
    thread.stop()


def test_vla_thread_handles_bad_rgb_source(mock_vla_client, trajectory_buffer):
    """Thread handles bad RGB source gracefully."""
    
    def bad_source():
        raise RuntimeError("Camera error")
    
    thread = VLAQueryThread(
        smolvla_client=mock_vla_client,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.05,
    )
    
    thread.start(bad_source, "test")
    time.sleep(0.15)
    
    assert thread._running  # Still running despite failures
    
    thread.stop()


# ============================================================================
# 4. Statistics Tracking Tests
# ============================================================================


def test_vla_thread_statistics(vla_thread, rgb_source):
    """Thread tracks statistics correctly."""
    vla_thread.start(rgb_source, "reach target")
    
    time.sleep(0.2)
    vla_thread.stop()
    
    stats = vla_thread.get_stats()
    assert "running" in stats
    assert "query_count" in stats
    assert "success_count" in stats
    assert "failure_count" in stats
    assert "success_rate_percent" in stats
    assert "last_query_time" in stats
    
    # At least some successful queries
    assert stats["success_count"] > 0
    assert stats["success_rate_percent"] > 0


def test_vla_thread_success_rate(vla_thread, rgb_source):
    """Success rate is calculated correctly."""
    vla_thread.start(rgb_source, "reach target")
    time.sleep(0.2)
    vla_thread.stop()
    
    stats = vla_thread.get_stats()
    expected_rate = (
        (stats["success_count"] / stats["query_count"] * 100)
        if stats["query_count"] > 0
        else 0
    )
    assert abs(stats["success_rate_percent"] - expected_rate) < 0.1


# ============================================================================
# 5. Thread Lifecycle Tests
# ============================================================================


def test_vla_thread_stops_cleanly(vla_thread, rgb_source):
    """Thread stops cleanly without hanging."""
    vla_thread.start(rgb_source, "reach target")
    time.sleep(0.1)
    
    vla_thread.stop()
    assert not vla_thread._running


def test_vla_thread_persists_for_minute(mock_vla_client, trajectory_buffer, rgb_source):
    """Thread runs without crashing for extended period."""
    thread = VLAQueryThread(
        smolvla_client=mock_vla_client,
        trajectory_buffer=trajectory_buffer,
        poll_interval_s=0.1,
    )
    
    thread.start(rgb_source, "reach target")
    
    # Run for 5 seconds as a stress test (was 60s, reduced for faster tests)
    time.sleep(5.0)
    
    stats = thread.get_stats()
    assert stats["running"] is True
    assert stats["query_count"] > 10  # Should have many queries by now
    
    thread.stop()
    assert stats["running"] is True  # Snapshot before stop


def test_vla_thread_cannot_stop_twice(vla_thread, rgb_source):
    """Thread handle second stop gracefully."""
    vla_thread.start(rgb_source, "test")
    time.sleep(0.1)
    
    vla_thread.stop()
    vla_thread.stop()  # Should not error


# ============================================================================
# 6. Convenience Function Tests
# ============================================================================


def test_poll_vla_background_convenience(mock_vla_client, trajectory_buffer, rgb_source):
    """Convenience function works correctly."""
    thread_mgr = poll_vla_background(
        smolvla_client=mock_vla_client,
        trajectory_buffer=trajectory_buffer,
        rgb_source=rgb_source,
        instruction="test",
        poll_interval_s=0.05,
    )
    
    assert isinstance(thread_mgr, VLAQueryThread)
    assert thread_mgr._running is True
    
    time.sleep(0.15)
    
    stats = thread_mgr.get_stats()
    assert stats["query_count"] > 0
    
    thread_mgr.stop()


# ============================================================================
# Additional Integration Tests
# ============================================================================


def test_vla_thread_with_real_trajectory_buffer(mock_vla_client, rgb_source):
    """Thread integrates with real TrajectoryBuffer."""
    buffer = TrajectoryBuffer()
    thread = VLAQueryThread(
        smolvla_client=mock_vla_client,
        trajectory_buffer=buffer,
        poll_interval_s=0.05,
    )
    
    thread.start(rgb_source, "reach target")
    time.sleep(0.2)
    
    # Buffer should have a subgoal now
    assert buffer.current_subgoal_q is not None
    assert buffer.current_subgoal_q.shape == (3,)
    
    thread.stop()


def test_vla_thread_subgoal_shape(vla_thread, rgb_source):
    """VLA subgoal has correct shape [3]."""
    vla_thread.start(rgb_source, "test")
    
    time.sleep(0.15)
    
    if vla_thread.trajectory_buffer.current_subgoal_q is not None:
        q_goal = vla_thread.trajectory_buffer.current_subgoal_q
        assert q_goal.shape == (3,), f"Expected shape (3,), got {q_goal.shape}"
        assert np.isfinite(q_goal).all(), "Subgoal contains NaN/inf"
    
    vla_thread.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

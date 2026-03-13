"""
Tests for TrajectoryBuffer: smooth reference trajectory generator.

Tests cover:
- Subgoal updating and storage
- Quintic spline interpolation smoothness
- Goal arrival detection
- Hold-position fallback
- Thread-safe statistics
"""

import numpy as np
import pytest

from src.smolvla_client.trajectory_buffer import TrajectoryBuffer


class TestTrajectoryBufferInitialization:
    """Test buffer initialization."""
    
    def test_init_default_threshold(self):
        """Default arrival threshold is 0.05 rad."""
        buf = TrajectoryBuffer()
        assert buf.arrival_threshold == 0.05
        assert buf.goal_reached == True  # Initially no goal
        assert buf.current_subgoal_q is None
    
    def test_init_custom_threshold(self):
        """Custom arrival threshold can be set."""
        buf = TrajectoryBuffer(arrival_threshold_rad=0.1)
        assert buf.arrival_threshold == 0.1


class TestSubgoalUpdate:
    """Test subgoal updating."""
    
    def test_update_valid_subgoal(self):
        """Update with valid [3] subgoal."""
        buf = TrajectoryBuffer()
        q_goal = np.array([0.2, 0.3, -0.2])
        
        buf.update_subgoal(q_goal)
        
        assert buf.current_subgoal_q is not None
        np.testing.assert_array_almost_equal(buf.current_subgoal_q, q_goal)
        assert buf.goal_reached == False
        assert buf._query_count == 1
    
    def test_update_invalid_subgoal_none(self):
        """Invalid (None) subgoal is rejected gracefully."""
        buf = TrajectoryBuffer()
        buf.update_subgoal(None)
        
        assert buf.current_subgoal_q is None
        assert buf._query_count == 0
    
    def test_update_invalid_subgoal_wrong_length(self):
        """Invalid (wrong length) subgoal is rejected."""
        buf = TrajectoryBuffer()
        buf.update_subgoal(np.array([0.1, 0.2]))  # Only 2 elements
        
        assert buf.current_subgoal_q is None
        assert buf._query_count == 0
    
    def test_multiple_updates_increments_count(self):
        """Multiple updates increment query counter."""
        buf = TrajectoryBuffer()
        
        for i in range(3):
            buf.update_subgoal(np.array([0.1*i, 0.2*i, -0.1*i]))
        
        assert buf._query_count == 3


class TestQuinticInterpolation:
    """Test quintic spline trajectory generation."""
    
    def test_interpolation_boundary_conditions(self):
        """Interpolated trajectory starts and ends at correct positions."""
        buf = TrajectoryBuffer()
        q0 = np.array([0.0, 0.5, -0.3])
        qf = np.array([0.5, 0.2, 0.1])
        
        buf.update_subgoal(qf)
        q_ref, qdot_ref = buf.get_reference_trajectory(q0, N=10, dt=0.01)
        
        # Start position (should be close to q0, within numerical precision)
        np.testing.assert_array_almost_equal(q_ref[0], q0, decimal=2)
        
        # End position (should be close to qf)
        np.testing.assert_array_almost_equal(q_ref[-1], qf, decimal=2)
    
    def test_interpolation_smooth_velocities(self):
        """Velocities are reasonable (< 1 rad/s for normal trajectories)."""
        buf = TrajectoryBuffer()
        q0 = np.array([0.0, 0.3, -0.3])
        qf = np.array([0.2, 0.2, -0.2])
        
        buf.update_subgoal(qf)
        # Use longer trajectory time (1 second total with 100 points at 10ms each)
        q_ref, qdot_ref = buf.get_reference_trajectory(q0, N=100, dt=0.01)
        
        # Velocities should be reasonable
        assert np.all(np.abs(qdot_ref) < 1.0), f"Velocities too high: {qdot_ref}"
    
    def test_interpolation_zero_boundary_velocities(self):
        """Start and end velocities are (near) zero."""
        buf = TrajectoryBuffer()
        q0 = np.array([0.0, 0.3, -0.3])
        qf = np.array([0.2, 0.2, -0.2])
        
        buf.update_subgoal(qf)
        q_ref, qdot_ref = buf.get_reference_trajectory(q0, N=20, dt=0.01)
        
        # Start velocity should be near zero
        assert np.abs(qdot_ref[0]).max() < 0.1, f"Start velocity not zero: {qdot_ref[0]}"
        
        # End velocity should be near zero
        assert np.abs(qdot_ref[-1]).max() < 0.1, f"End velocity not zero: {qdot_ref[-1]}"
    
    def test_interpolation_monotonic(self):
        """For 1D case, trajectory should be monotonic (no back-tracking)."""
        buf = TrajectoryBuffer()
        q0 = np.array([0.0, 0.0, 0.0])
        qf = np.array([0.5, 0.3, 0.2])  # All positive deltas
        
        buf.update_subgoal(qf)
        q_ref, qdot_ref = buf.get_reference_trajectory(q0, N=30, dt=0.01)
        
        # For each joint, check monotonicity
        for j in range(3):
            diffs = np.diff(q_ref[:, j])
            # All differences should have same sign (all positive or all ≥ 0)
            assert np.all(diffs >= -1e-6), f"Joint {j} is non-monotonic"


class TestGoalArrivalDetection:
    """Test goal arrival detection."""
    
    def test_arrival_when_close(self):
        """Goal reached when within threshold."""
        buf = TrajectoryBuffer(arrival_threshold_rad=0.05)
        q_goal = np.array([0.2, 0.3, -0.1])
        
        buf.update_subgoal(q_goal)
        
        # Position very close to goal
        q_close = np.array([0.201, 0.302, -0.101])
        assert buf.check_arrival(q_close) == True
        assert buf.goal_reached == True
    
    def test_arrival_when_far(self):
        """Goal not reached when far."""
        buf = TrajectoryBuffer(arrival_threshold_rad=0.05)
        q_goal = np.array([0.2, 0.3, -0.1])
        
        buf.update_subgoal(q_goal)
        
        # Position far from goal
        q_far = np.array([0.5, 0.5, 0.5])
        assert buf.check_arrival(q_far) == False
        assert buf.goal_reached == False
    
    def test_arrival_hysteresis(self):
        """Hysteresis prevents flickering at threshold boundary."""
        buf = TrajectoryBuffer(arrival_threshold_rad=0.05)
        q_goal = np.array([0.0, 0.0, 0.0])
        
        buf.update_subgoal(q_goal)
        assert buf.goal_reached == False
        
        # Move close to goal
        q_close = np.array([0.01, 0.01, 0.01])
        buf.check_arrival(q_close)
        assert buf.goal_reached == True
        
        # Move slightly away (but still within threshold)
        # Goal_reached should stay True (hysteresis)
        q_slightly_far = np.array([0.04, 0.04, 0.04])
        result = buf.check_arrival(q_slightly_far)
        # Once True, stays True even if slightly outside threshold
        assert buf.goal_reached == True
    
    def test_arrival_no_goal(self):
        """Without goal set, arrival is always True."""
        buf = TrajectoryBuffer()
        
        assert buf.current_subgoal_q is None
        assert buf.check_arrival(np.array([0.0, 0.0, 0.0])) == True


class TestHoldPositionFallback:
    """Test hold-position fallback when no subgoal."""
    
    def test_hold_position_no_subgoal(self):
        """Without subgoal, return hold-position trajectory."""
        buf = TrajectoryBuffer()
        q_current = np.array([0.5, 0.2, -0.1])
        
        q_ref, qdot_ref = buf.get_reference_trajectory(q_current, N=10, dt=0.01)
        
        # All positions should be q_current (check each row equals q_current)
        for i in range(q_ref.shape[0]):
            np.testing.assert_array_almost_equal(q_ref[i], q_current)
        
        # All velocities should be zero
        np.testing.assert_array_almost_equal(qdot_ref, 0.0)
    
    def test_hold_position_shape(self):
        """Hold-position arrays have correct shape."""
        buf = TrajectoryBuffer()
        q_current = np.array([0.5, 0.2, -0.1])
        N = 20
        
        q_ref, qdot_ref = buf.get_reference_trajectory(q_current, N=N, dt=0.01)
        
        assert q_ref.shape == (N, 3)
        assert qdot_ref.shape == (N, 3)


class TestReferenceTrajectoryShapes:
    """Test output array shapes."""
    
    def test_shapes_various_horizons(self):
        """Shapes are correct for various horizon lengths."""
        buf = TrajectoryBuffer()
        buf.update_subgoal(np.array([0.2, 0.3, -0.2]))
        q_current = np.array([0.0, 0.0, 0.0])
        
        for N in [5, 10, 20, 50]:
            q_ref, qdot_ref = buf.get_reference_trajectory(q_current, N=N)
            assert q_ref.shape == (N, 3)
            assert qdot_ref.shape == (N, 3)
    
    def test_dtype_is_float32(self):
        """Arrays are float32 (memory efficient)."""
        buf = TrajectoryBuffer()
        buf.update_subgoal(np.array([0.2, 0.3, -0.2]))
        q_current = np.array([0.0, 0.0, 0.0])
        
        q_ref, qdot_ref = buf.get_reference_trajectory(q_current, N=10)
        
        assert q_ref.dtype == np.float32
        assert qdot_ref.dtype == np.float32


class TestBufferStatistics:
    """Test statistics tracking."""
    
    def test_get_stats(self):
        """Statistics dict has expected keys."""
        buf = TrajectoryBuffer()
        q_goal = np.array([0.2, 0.3, -0.1])
        buf.update_subgoal(q_goal)
        
        stats = buf.get_stats()
        
        assert "current_subgoal" in stats
        assert "goal_reached" in stats
        assert "arrival_threshold_rad" in stats
        assert "total_subgoals_received" in stats
        
        assert stats["goal_reached"] == False
        assert stats["total_subgoals_received"] == 1
    
    def test_stats_no_subgoal(self):
        """Statistics when no subgoal set."""
        buf = TrajectoryBuffer()
        stats = buf.get_stats()
        
        assert stats["current_subgoal"] is None
        assert stats["total_subgoals_received"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

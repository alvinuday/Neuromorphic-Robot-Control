"""Tests for Phase 6: SmolVLA Mock Server."""
import numpy as np
import pytest
from src.smolvla.mock_vla import MockVLAServer


@pytest.fixture
def vla():
    return MockVLAServer()


def _dummy_rgb():
    return np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)


def test_predict_output_shape(vla):
    """Test that predict returns 7-D action."""
    rgb = _dummy_rgb()
    state = np.zeros(6)
    result = vla.predict(rgb, state)
    
    assert 'action' in result
    assert len(result['action']) == 7, f"Expected 7-D action, got {len(result['action'])}"


def test_predict_contains_source_field(vla):
    """CRITICAL: Mock must always have source='MOCK' field."""
    result = vla.predict(_dummy_rgb(), np.zeros(6))
    assert 'source' in result, "Missing 'source' field"
    assert result['source'] == 'MOCK', f"Expected source='MOCK', got {result['source']}"


def test_predict_deterministic(vla):
    """Test that same input gives same action."""
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    rgb = np.ones((84, 84, 3), dtype=np.uint8) * 128
    
    result1 = vla.predict(rgb.copy(), state.copy())
    result2 = vla.predict(rgb.copy(), state.copy())
    
    np.testing.assert_array_almost_equal(
        result1['action'],
        result2['action'],
        err_msg="Same input should give same action (deterministic)"
    )


def test_predict_responds_to_state_changes(vla):
    """Test that different states produce different actions."""
    rgb = _dummy_rgb()
    
    result1 = vla.predict(rgb, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    result2 = vla.predict(rgb, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    
    assert not np.allclose(result1['action'], result2['action']), \
        "Different states should produce different actions"


def test_predict_latency_field(vla):
    """Test that latency is reasonable."""
    result = vla.predict(_dummy_rgb(), np.zeros(6))
    assert 'latency_ms' in result
    assert result['latency_ms'] > 0, "Latency should be positive"
    assert result['latency_ms'] < 1000, "Latency should be reasonable (<1s)"


def test_predict_success_field(vla):
    """Test that success field is present and True."""
    result = vla.predict(_dummy_rgb(), np.zeros(6))
    assert 'success' in result
    assert result['success'] is True, "Mock should always succeed"


def test_predict_output_bounds(vla):
    """Test that action values are bounded."""
    state = np.array([0.5, -0.5, 0.8, -0.2, 0.1, 0.0])
    result = vla.predict(_dummy_rgb(), state)
    action = np.array(result['action'])
    
    # Actions should be in reasonable range for torques/joint targets
    assert np.all(np.isfinite(action)), "Actions should not be inf or NaN"
    assert np.abs(action).max() < 2.0, "Actions should be bounded"


def test_reset_clears_counter(vla):
    """Test that reset clears call counter."""
    vla.predict(_dummy_rgb(), np.zeros(6))
    vla.predict(_dummy_rgb(), np.zeros(6))
    assert vla.call_count == 2
    
    vla.reset()
    assert vla.call_count == 0, "Reset should clear call counter"


def test_call_count_increments(vla):
    """Test that call_count increments with each predict."""
    assert vla.call_count == 0
    
    vla.predict(_dummy_rgb(), np.zeros(6))
    assert vla.call_count == 1
    
    vla.predict(_dummy_rgb(), np.zeros(6))
    assert vla.call_count == 2


def test_missing_instruction_field(vla):
    """Test that instruction parameter is optional."""
    # This should not crash even without explicit instruction
    result = vla.predict(_dummy_rgb(), np.zeros(6))
    assert 'action' in result


def test_action_std_present(vla):
    """Test that action_std is provided."""
    result = vla.predict(_dummy_rgb(), np.zeros(6))
    assert 'action_std' in result
    assert len(result['action_std']) == 7, "Expected 7-D action_std"

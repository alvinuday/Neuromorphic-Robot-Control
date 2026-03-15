"""Tests for Phase 5: Sensor Fusion Encoder."""
import numpy as np
import pytest
from src.fusion.real_fusion_encoder import RealFusionEncoder


def _dummy_rgb(seed=0):
    """Generate dummy RGB image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (84, 84, 3), dtype=np.uint8)


def _obs(seed=0):
    """Generate dummy observation dict."""
    rgb = _dummy_rgb(seed)
    return {
        'rgb': rgb,
        'prev_rgb': _dummy_rgb(seed + 1),
        'state': np.array([0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    }


@pytest.mark.parametrize("mode,factory,expected_dim", [
    ("M0", RealFusionEncoder.mode_rgb_only, 128),
    ("M1", RealFusionEncoder.mode_rgb_events, 224),
    ("M2", RealFusionEncoder.mode_rgb_lidar, 192),
    ("M3", RealFusionEncoder.mode_rgb_proprio, 160),
    ("M4", RealFusionEncoder.mode_full, 320),
])
def test_output_shape(mode, factory, expected_dim):
    """Test that each mode produces correct output shape."""
    enc = factory()
    feat = enc.encode(_obs())
    assert feat.shape == (expected_dim,), \
        f"{mode}: expected shape {(expected_dim,)}, got {feat.shape}"
    assert feat.dtype == np.float32, f"{mode}: expected float32, got {feat.dtype}"
    assert enc.feature_dim == expected_dim, \
        f"{mode}: encoder.feature_dim mismatch"


@pytest.mark.parametrize("factory", [
    RealFusionEncoder.mode_rgb_only,
    RealFusionEncoder.mode_rgb_events,
    RealFusionEncoder.mode_rgb_lidar,
    RealFusionEncoder.mode_rgb_proprio,
    RealFusionEncoder.mode_full,
])
def test_features_not_all_zero(factory):
    """Test that features contain meaningful information."""
    feat = factory().encode(_obs())
    # Features should have some variation (not all zero)
    assert np.abs(feat).sum() > 0.1, \
        "Features are all zero (or nearly zero)"


@pytest.mark.parametrize("factory", [
    RealFusionEncoder.mode_rgb_only,
    RealFusionEncoder.mode_rgb_events,
    RealFusionEncoder.mode_rgb_lidar,
])
def test_different_images_different_features(factory):
    """Test that different images produce different features."""
    enc = factory()
    f1 = enc.encode(_obs(seed=0))
    f2 = enc.encode(_obs(seed=99))
    # Should not be identical (different inputs → different outputs)
    assert not np.allclose(f1, f2, atol=1e-5), \
        "Different inputs should produce different features"


def test_missing_prev_rgb_handled():
    """Test that missing prev_rgb is handled gracefully."""
    enc = RealFusionEncoder.mode_rgb_events()
    obs = {
        'rgb': _dummy_rgb(),
        'state': np.zeros(7)
        # No 'prev_rgb'
    }
    feat = enc.encode(obs)  # Should not crash
    assert feat.shape == (224,), f"Expected shape (224,), got {feat.shape}"


def test_latency_reasonable():
    """Test that fusion encoding runs fast enough."""
    enc = RealFusionEncoder.mode_full()
    feat, ms = enc.encode_with_timing(_obs())
    assert ms < 200, f"Fusion encoding took {ms:.1f}ms — expected <200ms for real-time"
    assert feat.shape == (enc.feature_dim,)


def test_rgb_only_consistency():
    """Test that RGB-only mode is deterministic."""
    enc = RealFusionEncoder.mode_rgb_only()
    obs = _obs(seed=42)
    f1 = enc.encode(obs)
    f2 = enc.encode(obs)
    np.testing.assert_array_equal(f1, f2, 
                                   err_msg="RGB encoding should be deterministic")


def test_proprio_contains_joint_info():
    """Test that proprioception features encode joint state."""
    enc_rgb = RealFusionEncoder.mode_rgb_only()
    enc_prop = RealFusionEncoder.mode_rgb_proprio()
    
    # Observation with non-zero joint angles
    obs_zero = _obs()
    obs_nonzero = _obs()
    obs_nonzero['state'] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    # RGB should be the same (same image)
    f_rgb_zero = enc_rgb.encode(obs_zero)[:128]
    f_rgb_nonzero = enc_rgb.encode(obs_nonzero)[:128]
    
    # Proprioception features should differ
    enc_prop_zero = RealFusionEncoder.mode_rgb_proprio()
    enc_prop_nonzero = RealFusionEncoder.mode_rgb_proprio()
    
    f_prop_zero = enc_prop_zero.encode(obs_zero)
    f_prop_nonzero = enc_prop_nonzero.encode(obs_nonzero)
    
    # First 128 dims (RGB) should match, last 32 dims (proprio) should differ
    assert not np.allclose(f_prop_zero, f_prop_nonzero, atol=1e-5), \
        "Different joint states should produce different proprio features"


def test_all_modes_produce_bounded_features():
    """Test that all feature values are bounded."""
    for factory in [
        RealFusionEncoder.mode_rgb_only,
        RealFusionEncoder.mode_rgb_events,
        RealFusionEncoder.mode_rgb_lidar,
        RealFusionEncoder.mode_rgb_proprio,
        RealFusionEncoder.mode_full,
    ]:
        enc = factory()
        feat = enc.encode(_obs())
        # Features should be bounded (not infinite or NaN)
        assert np.all(np.isfinite(feat)), \
            f"{factory.__name__}: contains infinite or NaN values"
        # Most features should be in reasonable range
        assert np.abs(feat).max() < 1e6, \
            f"{factory.__name__}: features contain very large values"


def test_timing_output_matches_encoding():
    """Test that encode_with_timing produces same features as encode."""
    enc = RealFusionEncoder.mode_full()
    obs = _obs()
    
    feat_plain = enc.encode(obs)
    feat_timed, ms = enc.encode_with_timing(obs)
    
    np.testing.assert_array_equal(feat_plain, feat_timed,
                                   err_msg="encode_with_timing should produce same output as encode")
    assert ms > 0, "Timing should be positive"

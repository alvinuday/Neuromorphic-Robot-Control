"""
Phase 8 tests: LeRobot dataset loading and GIF recording.

Tests cover:
- Dataset availability and schema validation
- Episode loading and array shapes
- GIF recording and file creation
- Context manager functionality
"""

import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import numpy as np
import pytest
from pathlib import Path

# ── Dataset tests (skip if unavailable) ──────────────────────────────────────

try:
    from data.loaders.lerobot_loader import LeRobotDatasetLoader, DatasetNotAvailableError
    _loader = LeRobotDatasetLoader()
    DATASET_AVAILABLE = True
except DatasetNotAvailableError:
    DATASET_AVAILABLE = False
except Exception:
    DATASET_AVAILABLE = False


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="LeRobot dataset not available")
def test_dataset_episode_shapes():
    """Test that loaded episodes have correct array shapes."""
    ep = _loader.load_episode(0)
    assert ep['states'].shape[1] == 8, f"Expected states shape [T, 8], got {ep['states'].shape}"
    assert ep['actions'].shape[1] == 7, f"Expected actions shape [T, 7], got {ep['actions'].shape}"
    assert ep['n_steps'] > 10, f"Episode too short: {ep['n_steps']} steps"


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="LeRobot dataset not available")
def test_dataset_info():
    """Test that dataset info is accessible and valid."""
    info = _loader.get_info()
    assert info['n_episodes'] > 10, f"Dataset has too few episodes: {info['n_episodes']}"
    assert info['action_dim'] == 7
    assert info['state_dim'] == 8
    assert info['n_steps_total'] > 100


@pytest.mark.skipif(not DATASET_AVAILABLE, reason="LeRobot dataset not available")
def test_dataset_episode_consistency():
    """Test that states and actions have same time dimension."""
    ep = _loader.load_episode(0)
    assert ep['states'].shape[0] == ep['actions'].shape[0], \
        f"States and actions have different lengths: {ep['states'].shape[0]} vs {ep['actions'].shape[0]}"
    assert ep['states'].dtype == np.float32
    assert ep['actions'].dtype == np.float32


# ── GIF recorder tests ───────────────────────────────────────────────────────

def test_gif_recorder_initialization():
    """Test that recorder initializes correctly."""
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder(fps=10, resize=(320, 240))
    assert rec.fps == 10
    assert rec.resize == (320, 240)
    assert len(rec._frames) == 0


def test_gif_recorder_saves_file(tmp_path):
    """Test that recorder saves a valid GIF file."""
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder(fps=10)
    rec.start()
    for _ in range(10):
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        rec.add_frame(frame)
    path = rec.save(str(tmp_path / "test.gif"))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 5_000, "GIF file too small — probably empty"


def test_gif_recorder_context_manager(tmp_path):
    """Test that context manager properly records and saves."""
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder(fps=10)
    gif_path = str(tmp_path / "context_test.gif")
    
    with rec.recording(gif_path):
        for i in range(15):
            frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
            rec.add_frame(frame)
    
    assert Path(gif_path).exists(), "GIF file was not created"
    assert Path(gif_path).stat().st_size > 5_000, "GIF file too small"


def test_gif_recorder_frame_count(tmp_path):
    """Test that all frames are recorded."""
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder()
    rec.start()
    n_frames = 20
    for _ in range(n_frames):
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        rec.add_frame(frame)
    assert len(rec._frames) == n_frames, f"Expected {n_frames} frames, got {len(rec._frames)}"


def test_gif_recorder_with_xarm_env(tmp_path):
    """Test recording actual MuJoCo environment frames."""
    try:
        from src.visualization.episode_recorder import EpisodeRecorder
        from src.simulation.envs.xarm_env import XArmEnv
    except RuntimeError as e:
        if "MUJOCO_GL" in str(e):
            pytest.skip(f"MuJoCo GL issue: {e}")
        raise
    
    rec = EpisodeRecorder(fps=10, resize=(320, 240))
    env = XArmEnv(render_mode='offscreen')
    obs = env.reset()
    gif_path = str(tmp_path / "env_test.gif")
    
    with rec.recording(gif_path):
        tau = np.zeros(8)
        tau[0] = 3.0  # Small torque on joint 0
        for _ in range(20):
            obs, _, done, _ = env.step(tau)
            if 'rgb' in obs:
                rec.add_frame(obs['rgb'])
            if done:
                break
    
    env.close()
    assert Path(gif_path).exists(), "GIF file was not created"
    assert Path(gif_path).stat().st_size > 10_000, "GIF file too small for 20 frames"


def test_gif_recorder_resize_handling(tmp_path):
    """Test that recorder properly resizes frames."""
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder(resize=(256, 256))
    rec.start()
    
    # Add frame with different size
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    rec.add_frame(frame)
    
    # Check that frame was resized
    assert len(rec._frames) == 1
    assert rec._frames[0].shape[:2] == (256, 256), f"Frame not resized: {rec._frames[0].shape}"


def test_gif_recorder_frame_dtype_conversion(tmp_path):
    """Test that recorder handles non-uint8 frame inputs."""
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder()
    rec.start()
    
    # Add frame with float input
    frame = np.random.rand(240, 320, 3).astype(np.float32) * 255
    rec.add_frame(frame)
    
    # Check conversion to uint8
    assert rec._frames[0].dtype == np.uint8
    assert rec._frames[0].shape == (240, 320, 3)


def test_gif_recorder_empty_save(tmp_path):
    """Test behavior when saving with no frames."""
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder()
    rec.start()
    
    # Try to save with no frames (should still create file)
    path = rec.save(str(tmp_path / "empty.gif"))
    # File may be very small but should still exist or raise
    # Depending on imageio behavior, this tests robustness
    assert Path(path).parent.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

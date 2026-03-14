"""
Test integration with real SmolVLA server on ngrok.

These tests require the server to be running.
Run with: pytest tests/test_real_smolvla_server.py -v -s

Reference: tech spec §9 (SmolVLA Integration - Real Server)
"""

import pytest
import asyncio
import numpy as np
import logging

from src.smolvla.real_client import RealSmolVLAClient, test_server_connectivity


logger = logging.getLogger(__name__)

# Default ngrok URL (from tech spec)
SMOLVLA_SERVER_URL = "https://symbolistically-unfutile-henriette.ngrok-free.dev"


class TestRealSmolVLAServer:
    """Tests for real SmolVLA server integration."""

    @pytest.fixture(scope="class")
    async def vla_client(self):
        """Initialize real VLA client."""
        return RealSmolVLAClient(server_url=SMOLVLA_SERVER_URL)

    @pytest.mark.asyncio
    async def test_server_health_check(self):
        """Test that server is accessible."""
        client = RealSmolVLAClient(server_url=SMOLVLA_SERVER_URL)
        is_healthy = await client.health_check()

        logger.info(f"Server health check: {'✓ OK' if is_healthy else '✗ FAILED'}")
        assert is_healthy, "SmolVLA server not responding at /health"

    @pytest.mark.asyncio
    async def test_server_inference(self):
        """Test real inference with dummy RGB image."""
        client = RealSmolVLAClient(server_url=SMOLVLA_SERVER_URL)

        # Create dummy RGB image (84x84)
        dummy_rgb = np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8)

        # Query server
        action = await client.predict(dummy_rgb)

        logger.info(f"Inference result: action shape={action.shape}, "
                   f"mean={np.mean(action):.4f}, std={np.std(action):.4f}")

        # Verify output
        assert isinstance(action, np.ndarray), f"Expected ndarray, got {type(action)}"
        assert action.dtype == np.float32, f"Expected float32, got {action.dtype}"
        assert len(action.shape) == 1, f"Expected 1D array, got shape {action.shape}"
        assert action.shape[0] > 0, f"Expected non-empty action, got {action.shape}"

    @pytest.mark.asyncio
    async def test_inference_with_state(self):
        """Test inference with optional state conditioning."""
        client = RealSmolVLAClient(server_url=SMOLVLA_SERVER_URL)

        dummy_rgb = np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8)
        dummy_state = np.zeros(8)  # 6-DOF arm + 2-DOF gripper

        action = await client.predict(
            rgb_image=dummy_rgb,
            state=dummy_state,
            instruction="move to the left"
        )

        logger.info(f"Inference with state: action={action}")
        assert isinstance(action, np.ndarray)

    @pytest.mark.asyncio
    async def test_multiple_sequential_queries(self):
        """Test multiple sequential queries (simulating control loop)."""
        client = RealSmolVLAClient(server_url=SMOLVLA_SERVER_URL)

        n_queries = 3
        actions = []

        for i in range(n_queries):
            dummy_rgb = np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8)
            action = await client.predict(dummy_rgb)
            actions.append(action)
            logger.info(f"Query {i+1}/{n_queries}: action shape={action.shape}")

        stats = client.get_stats()
        logger.info(f"Query statistics:\n{stats}")

        assert stats["success_count"] == n_queries, \
            f"Expected {n_queries} successes, got {stats['success_count']}"
        assert stats["mean_latency_ms"] > 0, "Expected latency to be measured"

    @pytest.mark.asyncio
    async def test_client_statistics(self):
        """Test client statistics tracking."""
        client = RealSmolVLAClient(server_url=SMOLVLA_SERVER_URL)

        # Make a few queries
        for _ in range(2):
            dummy_rgb = np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8)
            await client.predict(dummy_rgb)

        stats = client.get_stats()

        logger.info(f"Client stats:\n"
                   f"  Calls: {stats['call_count']}\n"
                   f"  Success: {stats['success_count']}/{stats['call_count']}\n"
                   f"  Mean latency: {stats['mean_latency_ms']:.1f} ms\n"
                   f"  Max latency: {stats['max_latency_ms']:.1f} ms")

        assert stats["call_count"] == 2
        assert stats["success_count"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["mean_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_server_connectivity_function(self):
        """Test the convenience health check function."""
        is_healthy, stats = await test_server_connectivity(SMOLVLA_SERVER_URL)

        logger.info(f"Server connectivity: {'✓ Healthy' if is_healthy else '✗ Failed'}")
        logger.info(f"Stats: {stats}")

        assert is_healthy, "Server connectivity test failed"


# Run with: python -m pytest tests/test_real_smolvla_server.py -v -s
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

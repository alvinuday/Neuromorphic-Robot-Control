"""
Real Integration Tests for Phase 8B with Actual SmolVLA Server

These tests connect to a REAL SmolVLA server running on Colab via ngrok tunnel.
Unlike test_integration_phase8b.py (which uses mocks), these test the actual
end-to-end system with real inference.

SETUP:
1. Start the SmolVLA server in Colab notebook (vla/smolvla_server.ipynb)
2. Set environment variable: export SMOLVLA_SERVER_URL="https://xxxx-ngrok-free.dev"
3. Run: pytest tests/test_integration_real_smolvla.py -v -s

The -s flag shows all logging output.
"""

import pytest
import asyncio
import numpy as np
import logging
import os
import time
from typing import Optional
from unittest.mock import patch, AsyncMock
import pytest_asyncio

from src.integration.smolvla_server_client import SmolVLAServerConfig, RealSmolVLAClient
from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
from src.integration.dual_system_controller import DualSystemController, ControlState
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


logger = logging.getLogger(__name__)


@pytest.fixture
def server_url() -> Optional[str]:
    """Get SmolVLA server URL from environment."""
    url = os.getenv("SMOLVLA_SERVER_URL", None)
    if url:
        logger.info(f"Using SmolVLA server at: {url}")
    return url


@pytest.fixture
def server_config(server_url) -> Optional[SmolVLAServerConfig]:
    """Create server config if URL is available."""
    if server_url is None:
        pytest.skip("SMOLVLA_SERVER_URL not set")
    return SmolVLAServerConfig(server_url=server_url, timeout_s=2.0)


@pytest_asyncio.fixture
async def real_client(server_config):
    """Create and start real client."""
    client = RealSmolVLAClient(server_config)
    await client.start()
    yield client
    await client.stop()


@pytest.fixture
def trajectory_buffer():
    """Create trajectory buffer."""
    return TrajectoryBuffer(arrival_threshold_rad=0.05)


@pytest.fixture
def mock_mpc_solver():
    """Mock MPC solver."""
    class MockSolver:
        def solve(self, q, qdot, q_ref, qdot_ref):
            return np.random.uniform(-2, 2, size=3)
    return MockSolver()


@pytest.fixture
def controller(mock_mpc_solver, real_client, trajectory_buffer):
    """Create dual system controller."""
    return DualSystemController(
        mpc_solver=mock_mpc_solver,
        smolvla_client=real_client,
        trajectory_buffer=trajectory_buffer,
        mpc_horizon_steps=10,
        control_dt_s=0.01,
        vla_query_interval_s=0.2
    )


# ============================================================================
# GATE 4B: Real SmolVLA Server Integration
# ============================================================================

class TestSmolVLAServerHealth:
    """Test basic server connectivity."""
    
    @pytest.mark.asyncio
    async def test_real_server_health_endpoint(self, real_client):
        """Test that server /health endpoint responds."""
        healthy = await real_client.health_check()
        assert healthy, "SmolVLA server health check failed"
        logger.info("✓ Server health check passed")
    
    @pytest.mark.asyncio
    async def test_real_server_config_valid(self, server_config):
        """Test that server config is valid."""
        assert server_config.is_valid(), "Server config invalid"
        assert "http" in server_config.server_url, "Server URL must have protocol"
        logger.info(f"✓ Server config valid: {server_config.server_url}")
    
    @pytest.mark.asyncio
    async def test_real_server_timeout_handling(self):
        """Test timeout handling with unreasonable timeout."""
        config = SmolVLAServerConfig(
            server_url="https://httpbin.org/delay/10",
            timeout_s=0.1  # Very short timeout
        )
        client = RealSmolVLAClient(config)
        await client.start()
        
        # This should timeout
        result = await client.query_action(np.zeros((224, 224, 3), dtype=np.uint8))
        
        await client.stop()
        
        # Should timeout gracefully
        assert result is None, "Should timeout gracefully"
        assert client.timeout_count > 0, "Should count timeouts"
        logger.info("✓ Timeout handling works")


class TestSmolVLAServerInference:
    """Test actual inference on real server."""
    
    @pytest.mark.asyncio
    async def test_real_inference_single_image(self, real_client):
        """Test single image inference."""
        rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        result = await real_client.query_action(rgb, instruction="reach forward")
        
        assert result is not None, "Inference failed"
        assert "action" in result, "Missing action field"
        assert "latency_ms" in result, "Missing latency_ms field"
        assert len(result["action"]) == 4, "Wrong action dimension"
        assert result["latency_ms"] > 0, "Invalid latency"
        
        logger.info(f"✓ Inference succeeded in {result['latency_ms']:.1f}ms")
        logger.info(f"  Action: {[round(a, 4) for a in result['action']]}")
    
    @pytest.mark.asyncio
    async def test_real_inference_multiple_images(self, real_client):
        """Test multiple sequential inferences."""
        results = []
        latencies = []
        
        for i in range(3):
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            result = await real_client.query_action(rgb)
            
            assert result is not None, f"Inference {i} failed"
            results.append(result)
            latencies.append(result["latency_ms"])
        
        mean_latency = np.mean(latencies)
        logger.info(f"✓ {len(results)} inferences completed")
        logger.info(f"  Mean latency: {mean_latency:.1f}ms")
        logger.info(f"  Min/max: {min(latencies):.1f}ms / {max(latencies):.1f}ms")
        
        # Check stats
        stats = real_client.get_stats()
        assert stats["query_count"] == 3, "Query count wrong"
        assert stats["success_count"] == 3, "Success count wrong"
        assert stats["failure_count"] == 0, "Should have no failures"
    
    @pytest.mark.asyncio
    async def test_real_inference_different_resolutions(self, real_client):
        """Test inference with different input resolutions."""
        sizes = [(224, 224), (512, 512), (128, 128)]
        
        for h, w in sizes:
            rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result = await real_client.query_action(rgb)
            
            assert result is not None, f"Inference failed for {h}x{w}"
            logger.info(f"✓ Inference successful for {h}x{w}")


class TestDualSystemWithRealVLA:
    """Test dual system controller with real VLA queries."""
    
    @pytest.mark.asyncio
    async def test_controller_step_with_real_vla(self, real_client, controller):
        """Test that controller step works with real VLA latency."""
        q = np.array([0.0, 0.0, 0.0])
        qdot = np.array([0.0, 0.0, 0.0])
        rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        instruction = "reach forward"
        
        # Get real VLA result
        vla_result = await real_client.query_action(rgb, instruction=instruction)
        
        if vla_result is not None:
            # Update trajectory buffer with real result (action[:3] = EE position goal)
            action = vla_result["action"]
            controller.trajectory_buffer.update_subgoal(action[:3])
        
        # Controller step should NOT block (synchronous)
        start = time.time()
        tau = controller.step(q, qdot, rgb, instruction)
        step_time = (time.time() - start) * 1000
        
        assert tau is not None, "Step returned None"
        assert len(tau) == 3, "Wrong torque dimension"
        assert step_time < 50, f"Step took {step_time:.1f}ms (should be <50ms)"
        
        logger.info(f"✓ Controller step with real VLA: {step_time:.1f}ms")


class TestRealVLANonBlocking:
    """Test that real VLA queries don't block controller timing."""
    
    @pytest.mark.asyncio
    async def test_vla_latency_doesnt_affect_loop(self, real_client, controller):
        """
        Verify that VLA latency (~700ms) doesn't affect MPC loop timing.
        
        Simulates:
        - Main loop stepping controller at ~100 Hz
        - VLA queries happening in parallel (slow, ~700ms per query)
        </head>
        """
        q = np.zeros(3)
        qdot = np.zeros(3)
        
        mpc_timings = []
        vla_latencies = []
        
        # Run 20 controller steps
        for i in range(20):
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            # Measure MPC step time (should be <50ms)
            start = time.time()
            tau = controller.step(q, qdot, rgb, "reach forward")
            mpc_time = (time.time() - start) * 1000
            
            mpc_timings.append(mpc_time)
            
            # In parallel, query VLA (don't wait for it in main loop)
            # In real system, this would be in background thread
            if i % 5 == 0:
                vla_result = await real_client.query_action(rgb)
                if vla_result:
                    vla_latencies.append(vla_result["latency_ms"])
        
        mean_mpc = np.mean(mpc_timings)
        max_mpc = np.max(mpc_timings)
        mean_vla = np.mean(vla_latencies) if vla_latencies else 0
        
        logger.info(f"✓ Non-blocking test results:")
        logger.info(f"  MPC loop: mean={mean_mpc:.1f}ms, max={max_mpc:.1f}ms")
        logger.info(f"  VLA queries: mean={mean_vla:.1f}ms")
        logger.info(f"  Ratio: VLA is {mean_vla/mean_mpc:.0f}x slower than MPC")
        
        # MPC should be consistently <50ms
        assert mean_mpc < 50, f"MPC loop too slow: {mean_mpc:.1f}ms"
        assert max_mpc < 100, f"MPC peak too slow: {max_mpc:.1f}ms"


class TestGate4bValidation:
    """Gate 4b: Real SmolVLA Server Integration."""
    
    @pytest.mark.asyncio
    async def test_gate4b_server_accessible(self, real_client):
        """Gate 4b-1: Server is accessible and healthy."""
        healthy = await real_client.health_check()
        assert healthy, "Server not accessible"
    
    @pytest.mark.asyncio
    async def test_gate4b_inference_works(self, real_client):
        """Gate 4b-2: Inference produces valid outputs."""
        rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = await real_client.query_action(rgb)
        
        assert result is not None, "Inference failed"
        assert len(result["action"]) == 4, "Wrong action dimension"
        assert isinstance(result["latency_ms"], (int, float)), "Invalid latency type"
    
    @pytest.mark.asyncio
    async def test_gate4b_multiple_queries_succeed(self, real_client):
        """Gate 4b-3: 5 consecutive queries succeed with high success rate."""
        for i in range(5):
            rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            result = await real_client.query_action(rgb)
            assert result is not None, f"Query {i} failed"
        
        stats = real_client.get_stats()
        assert stats["success_rate"] >= 0.8, f"Success rate too low: {stats['success_rate']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

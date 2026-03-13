"""
Tests for SmolVLAClient async HTTP client.

Note: Tests require active Colab server. Mock endpoint can be tested locally.
For integration testing, set SMOLVLA_ENDPOINT env var to ngrok URL.
"""

import asyncio
import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch

from src.smolvla_client.async_client import SmolVLAClient, SmolVLAResponse


# Get endpoint from env (for integration tests against real Colab)
SMOLVLA_ENDPOINT = os.environ.get("SMOLVLA_ENDPOINT", "http://localhost:8000")


# Helper to create async context manager mocks
class AsyncContextManagerMock:
    """Mock for async context managers."""
    def __init__(self, return_value):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, *args):
        pass


class TestSmolVLAResponse:
    """Test dataclass initialization and properties."""
    
    def test_response_initialization(self):
        """SmolVLAResponse instantiates correctly."""
        action_chunk = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        subgoal = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        
        response = SmolVLAResponse(
            action_chunk=action_chunk,
            subgoal_xyz=subgoal,
            latency_ms=150.5,
            timestamp=time.time()
        )
        
        assert response.action_chunk.shape == (1, 7)
        assert response.subgoal_xyz.shape == (3,)
        assert response.latency_ms == 150.5
        assert response.timestamp > 0


class TestSmolVLAClientInitialization:
    """Test client initialization and configuration loading."""
    
    def test_client_init_with_explicit_url(self):
        """Initialize client with explicit endpoint URL."""
        url = "https://example.ngrok-free.dev"
        client = SmolVLAClient(endpoint_url=url)
        
        assert client.endpoint_url == url
        assert client.timeout_s == 2.0
        assert client._session is None  # Not started yet
    
    def test_client_init_with_default_timeout(self):
        """Default timeout is 2.0 seconds."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        assert client.timeout_s == 2.0
    
    def test_client_init_with_custom_timeout(self):
        """Custom timeout can be set."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000", timeout_s=5.0)
        assert client.timeout_s == 5.0


class TestSmolVLAClientSessionManagement:
    """Test async session lifecycle."""
    
    @pytest.mark.asyncio
    async def test_client_start_creates_session(self):
        """Calling start() initializes aiohttp session."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        assert client._session is None
        
        await client.start()
        assert client._session is not None
        assert not client._session.closed
        
        await client.stop()
    
    @pytest.mark.asyncio
    async def test_client_stop_closes_session(self):
        """Calling stop() closes session cleanly."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        await client.start()
        
        assert not client._session.closed
        await client.stop()
        assert client._session.closed
    
    @pytest.mark.asyncio
    async def test_client_double_start_idempotent(self):
        """Calling start() multiple times is safe."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        await client.start()
        session1 = client._session
        
        await client.start()
        session2 = client._session
        
        # Should reuse session
        assert session1 is session2
        
        await client.stop()


class TestImageEncoding:
    """Test RGB array ↔ base64 JPEG conversion."""
    
    def test_encode_rgb_to_base64(self):
        """Encode numpy RGB array to base64 JPEG."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Create simple test image: red top half, blue bottom half
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:240, :, 0] = 255  # red top
        rgb[240:, :, 2] = 255  # blue bottom
        
        # Encode
        b64 = client._encode_image(rgb)
        
        # Should be valid base64
        assert isinstance(b64, str)
        assert len(b64) > 100  # Reasonable JPEG size
        
        # Should decode back to image (allowing JPEG loss)
        decoded = client._decode_image(b64)
        assert decoded.shape[2] == 3  # RGB
    
    def test_encode_resizes_to_224x224(self):
        """Images are resized to 224×224 for SmolVLA."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Create various size images
        for h, w in [(480, 640), (100, 100), (1080, 1920)]:
            rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            b64 = client._encode_image(rgb)
            
            # Decode and check size
            decoded = client._decode_image(b64)
            assert decoded.shape == (224, 224, 3)
    
    def test_encode_handles_uint8_input(self):
        """Handles uint8 input correctly."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        b64 = client._encode_image(rgb)
        
        # Should succeed
        assert isinstance(b64, str)
        assert len(b64) > 0


class TestHealthCheck:
    """Test /health endpoint query."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Successful health check returns status dict."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Mock the response
        mock_resp_data = {"status": "ok", "model": "smolvla-base"}
        mock_response = MagicMock()
        mock_response.status = 200
        
        async def mock_json():
            return mock_resp_data
        
        mock_response.json = mock_json
        
        # Create async context manager mock
        ctx_mgr = AsyncContextManagerMock(mock_response)
        
        # Mock the session
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get.return_value = ctx_mgr
        client._session = mock_session
        
        result = await client.health_check()
        
        assert result is not None
        assert result["status"] == "ok"
    
    @pytest.mark.asyncio
    async def test_health_check_no_session(self):
        """Health check returns None if no session."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Don't start session
        assert client._session is None
        
        result = await client.health_check()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Health check returns None on timeout."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Mock timeout
        mock_session = AsyncMock()
        mock_session.get.side_effect = asyncio.TimeoutError()
        client._session = mock_session
        
        result = await client.health_check()
        assert result is None


class TestQueryAction:
    """Test /predict endpoint queries."""
    
    @pytest.mark.asyncio
    async def test_query_action_success(self):
        """Successful action query returns SmolVLAResponse."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Create test input
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        instruction = "pick up the red cube"
        joints = [0.0, 0.3, -0.5]
        
        # Mock response data
        mock_resp_data = {
            "action_chunk": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]],
            "subgoal_xyz": [0.5, 0.3, 0.2],
            "latency_ms": 150.0,
            "status": "ok"
        }
        mock_response = MagicMock()
        mock_response.status = 200
        
        async def mock_json():
            return mock_resp_data
        
        mock_response.json = mock_json
        
        # Create async context manager mock
        ctx_mgr = AsyncContextManagerMock(mock_response)
        
        # Mock the session
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = ctx_mgr
        client._session = mock_session
        
        # Query
        result = await client.query_action(rgb, instruction, joints)
        
        # Check result
        assert result is not None
        assert isinstance(result, SmolVLAResponse)
        assert result.action_chunk.shape == (1, 7)
        assert result.subgoal_xyz.shape == (3,)
        assert result.latency_ms == 150.0
        assert client._query_count == 1
        assert client._error_count == 0
    
    @pytest.mark.asyncio
    async def test_query_action_timeout(self):
        """Query timeout returns None, doesn't raise."""
        client = SmolVLAClient(
            endpoint_url="http://localhost:8000",
            timeout_s=0.1
        )
        
        # Mock timeout
        mock_session = AsyncMock()
        mock_session.post.side_effect = asyncio.TimeoutError()
        client._session = mock_session
        
        # Should return None, not raise
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = await client.query_action(rgb, "test", [0, 0, 0])
        
        assert result is None
        assert client._error_count == 1
    
    @pytest.mark.asyncio
    async def test_query_action_no_session(self):
        """Query without session returns None."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Don't start session
        assert client._session is None
        
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = await client.query_action(rgb, "test", [0, 0, 0])
        
        assert result is None
        assert client._error_count == 1
    
    @pytest.mark.asyncio
    async def test_query_stores_latest_response(self):
        """Latest response is stored for quick access."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Mock response data
        mock_resp_data = {
            "action_chunk": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]],
            "subgoal_xyz": [0.5, 0.3, 0.2],
            "latency_ms": 100.0,
            "status": "ok"
        }
        mock_response = MagicMock()
        mock_response.status = 200
        
        async def mock_json():
            return mock_resp_data
        
        mock_response.json = mock_json
        
        # Create async context manager mock
        ctx_mgr = AsyncContextManagerMock(mock_response)
        
        # Mock the session
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = ctx_mgr
        client._session = mock_session
        
        # Query
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = await client.query_action(rgb, "test", [0, 0, 0])
        
        # Check stored response
        assert client.latest_response is result
        assert client.get_latest_subgoal() is not None
        np.testing.assert_array_equal(
            client.get_latest_subgoal(),
            np.array([0.5, 0.3, 0.2], dtype=np.float32)
        )


class TestClientStats:
    """Test client statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Stats reporting is accurate."""
        client = SmolVLAClient(endpoint_url="http://localhost:8000")
        
        # Mock response data
        mock_resp_data = {
            "action_chunk": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]],
            "subgoal_xyz": [0.5, 0.3, 0.2],
            "latency_ms": 100.0,
            "status": "ok"
        }
        mock_response = MagicMock()
        mock_response.status = 200
        
        async def mock_json():
            return mock_resp_data
        
        mock_response.json = mock_json
        
        # Create async context manager mock
        ctx_mgr = AsyncContextManagerMock(mock_response)
        
        # Mock the session
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post.return_value = ctx_mgr
        client._session = mock_session
        
        # Make 3 queries
        rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(3):
            await client.query_action(rgb, "test", [0, 0, 0])
        
        # Check stats
        stats = client.get_stats()
        assert stats["query_count"] == 3
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 100.0
        assert stats["latest_latency_ms"] == 100.0


# Integration tests (require active server)
@pytest.mark.asyncio
@pytest.mark.skipif(
    SMOLVLA_ENDPOINT == "http://localhost:8000",
    reason="Skipping integration test (no live SMOLVLA endpoint)"
)
class TestSmolVLAIntegration:
    """Integration tests against live Colab server."""
    
    async def test_live_health_endpoint(self):
        """Query live /health endpoint."""
        client = SmolVLAClient(endpoint_url=SMOLVLA_ENDPOINT)
        await client.start()
        
        try:
            result = await client.health_check()
            assert result is not None
            assert "status" in result
        finally:
            await client.stop()
    
    async def test_live_single_inference(self):
        """Query live /predict endpoint with real image."""
        client = SmolVLAClient(endpoint_url=SMOLVLA_ENDPOINT)
        await client.start()
        
        try:
            # Create dummy image
            rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            result = await client.query_action(
                rgb,
                "pick up the red object",
                [0.0, 0.3, -0.5]
            )
            
            assert result is not None
            assert result.action_chunk is not None
            assert result.subgoal_xyz is not None
            assert result.latency_ms > 0
        finally:
            await client.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

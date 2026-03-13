"""
Async HTTP client for SmolVLA Colab server.

Non-blocking communication for vision-language model queries with timeout handling
and graceful degradation if server unavailable.

From techspec: System 1 (fast MPC) ↔ System 2 (slow VLA) via async buffer.
Never blocks main control thread.
"""

import asyncio
import aiohttp
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging
import base64
from PIL import Image
import io


logger = logging.getLogger(__name__)


@dataclass
class VLAResponse:
    """Response from SmolVLA server."""
    action: np.ndarray  # [dx, dy, dz, grasp_prob] ∈ [-1, 1]³ x [0, 1]
    action_std: np.ndarray  # Uncertainty estimate
    latency_ms: float  # Network + inference latency
    success: bool  # Whether request succeeded
    error: Optional[str] = None  # Error message if failed


class SmolVLAAsyncClient:
    """
    Async HTTP client for SmolVLA server on Colab.
    
    Designed for non-blocking queries with automatic timeout handling.
    Each query runs in background; main control loop continues unblocked.
    
    Attributes:
        server_url: Base URL of SmolVLA server (with protocol)
        timeout_s: Request timeout in seconds
        verify_ssl: Verify SSL certificates (False for ngrok)
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout_s: float = 1.0,
        verify_ssl: bool = False
    ):
        """
        Initialize async client.
        
        Args:
            server_url: SmolVLA server URL (e.g., "https://xxx-ngrok.io")
            timeout_s: Request timeout. Default: 1.0s (strict for 100 Hz MPC)
            verify_ssl: Skip SSL verification for ngrok tunnels
        """
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.verify_ssl = verify_ssl
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self):
        """Create aiohttp session (call once at startup)."""
        connector = aiohttp.TCPConnector(limit=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout_s)
        )
    
    async def disconnect(self):
        """Close aiohttp session (call at shutdown)."""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if server is online."""
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.get(
                f"{self.server_url}/health",
                ssl=self.verify_ssl,
                timeout=aiohttp.ClientTimeout(total=0.5)
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    async def predict(
        self,
        rgb_image: np.ndarray,
        task_embedding: Optional[np.ndarray] = None
    ) -> VLAResponse:
        """
        Query SmolVLA for action prediction.
        
        Non-blocking async call with timeout.
        
        Args:
            rgb_image: RGB image (H, W, 3) with values in [0, 255]
            task_embedding: Optional task encoding (usually None for reaching)
            
        Returns:
            VLAResponse with action, latency, success flag
        """
        import time
        start_time = time.time()
        
        try:
            if not self.session:
                await self.connect()
            
            # Encode image to base64
            image_b64 = self._encode_image(rgb_image)
            
            # Build request
            payload = {"rgb_image_b64": image_b64}
            if task_embedding is not None:
                payload["task_embedding"] = task_embedding.tolist()
            
            # Send request with timeout
            async with self.session.post(
                f"{self.server_url}/predict",
                json=payload,
                ssl=self.verify_ssl
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {error_text}")
                
                result = await resp.json()
                latency_ms = (time.time() - start_time) * 1000
                
                return VLAResponse(
                    action=np.array(result['action'], dtype=np.float32),
                    action_std=np.array(result['action_std'], dtype=np.float32),
                    latency_ms=latency_ms,
                    success=True,
                    error=None
                )
        
        except asyncio.TimeoutError:
            error = f"Request timeout (>{self.timeout_s}s)"
            logger.warning(f"VLA query timeout: {error}")
            return self._failed_response(error)
        
        except Exception as e:
            error = str(e)
            logger.warning(f"VLA query failed: {error}")
            return self._failed_response(error)
    
    def _encode_image(self, rgb_image: np.ndarray) -> str:
        """Encode numpy RGB array to base64 PNG."""
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _failed_response(self, error: str) -> VLAResponse:
        """Return default response when query fails."""
        return VLAResponse(
            action=np.zeros(4, dtype=np.float32),  # No-op [0, 0, 0, 0]
            action_std=np.ones(4, dtype=np.float32),  # High uncertainty
            latency_ms=self.timeout_s * 1000,
            success=False,
            error=error
        )


class VLAQueryThread:
    """
    Background thread for periodic VLA queries.
    
    Queries SmolVLA every N milliseconds asynchronously without blocking
    the main MPC control loop.
    
    Attributes:
        client: SmolVLAAsyncClient instance
        query_period_ms: Time between queries (~200 ms for ~5 Hz MPC)
        last_response: Most recent VLA response (None if no response yet)
    """
    
    def __init__(
        self,
        server_url: str,
        query_period_ms: int = 200
    ):
        """
        Initialize VLA query thread.
        
        Args:
            server_url: SmolVLA server URL
            query_period_ms: Milliseconds between queries. Default: 200 (5 Hz)
        """
        self.client = SmolVLAAsyncClient(server_url)
        self.query_period_ms = query_period_ms
        self.last_response: Optional[VLAResponse] = None
        self.last_image: Optional[np.ndarray] = None
        self.running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start background query loop (run in separate thread/event loop)."""
        await self.client.connect()
        self.running = True
        logger.info("VLA query thread started")
    
    async def stop(self):
        """Stop query loop and cleanup."""
        self.running = False
        await self.client.disconnect()
        logger.info("VLA query thread stopped")
    
    async def query_loop(self):
        """Main event loop for periodic queries."""
        while self.running:
            if self.last_image is not None:
                # Query VLA non-blockingly
                self.last_response = await self.client.predict(self.last_image)
                logger.debug(f"VLA response: action={self.last_response.action}, "
                            f"latency={self.last_response.latency_ms:.1f}ms")
            
            # Wait before next query
            await asyncio.sleep(self.query_period_ms / 1000.0)
    
    def update_observation(self, rgb_image: np.ndarray):
        """Update current observation for next query."""
        self.last_image = rgb_image.copy()
    
    def get_latest_action(self) -> Optional[VLAResponse]:
        """Get most recent VLA response (non-blocking)."""
        return self.last_response

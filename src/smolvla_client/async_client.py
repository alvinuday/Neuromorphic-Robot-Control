"""
Async HTTP client for SmolVLA Vision-Language-Action model.
Queries Colab FastAPI server via ngrok HTTPS tunnel (non-blocking).
Never blocks main control loop; always fails gracefully.
"""

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp
import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SmolVLAResponse:
    """Response from SmolVLA /predict endpoint.
    
    Attributes:
        action_chunk: shape [chunk_size, 7], float32
            - First 3 dims: [dx, dy, dz] EE displacement (m)
            - Next 3 dims: [droll, dpitch, dyaw] orientation change (rad)
            - Last 1 dim: grasp signal ∈ [0, 1]
        subgoal_xyz: shape [3], float32
            - Predicted end-effector target position [x, y, z] (m)
        latency_ms: float
            - Server query latency (milliseconds)
        timestamp: float
            - Wall time when response received (seconds since epoch)
    """
    action_chunk: np.ndarray      # [chunk_size, 7] float32
    subgoal_xyz: np.ndarray       # [3] float32
    latency_ms: float              # ms
    timestamp: float               # seconds


class SmolVLAClient:
    """
    Async HTTP client for SmolVLA inference server running on Colab.
    
    Architecture:
        - Colab (T4 GPU): FastAPI server running SmolVLA model
        - ngrok tunnel: HTTPS tunnel exposing :8000
        - Local (this code): aiohttp async client polling VLA
    
    Key Design:
        - All I/O is async (never blocks)
        - Sessions are pooled (reuse connection)
        - Timeouts are enforced (1-2 seconds)
        - Errors are logged, never raised (graceful failure)
        - Thread-safe latest_response storage
    
    Usage:
        client = SmolVLAClient("config/smolvla_config.yaml")
        await client.start()
        response = await client.query_action(
            rgb=rgb_array,
            instruction="pick up the red object",
            current_joints=[0.0, 0.3, -0.5]
        )
        if response:
            print(f"Subgoal: {response.subgoal_xyz}")
    """
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        config_path: str = "config/smolvla_config.yaml",
        timeout_s: float = 2.0
    ):
        """
        Initialize SmolVLA client.
        
        Args:
            endpoint_url: ngrok URL (e.g. "https://abc123.ngrok-free.dev")
                         If None, load from config_path
            config_path: YAML file with endpoint_url and timeout_s
            timeout_s: HTTP request timeout (seconds)
        """
        self.endpoint_url: str = endpoint_url or self._load_config(config_path)
        self.timeout_s = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None
        self.latest_response: Optional[SmolVLAResponse] = None
        self._query_count = 0
        self._error_count = 0
        
        logger.info(f"SmolVLAClient initialized → {self.endpoint_url}")
    
    def _load_config(self, config_path: str) -> str:
        """Load endpoint URL from YAML config file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            
            endpoint = cfg.get("endpoint_url")
            if not endpoint:
                raise ValueError("endpoint_url not found in config")
            
            return endpoint
        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}")
            return "http://localhost:8000"  # fallback
    
    async def start(self):
        """Start async HTTP session. Call before making queries."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            logger.debug("aiohttp session started")
    
    async def stop(self):
        """Close async HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("aiohttp session closed")
    
    async def health_check(self) -> Optional[dict]:
        """
        Query /health endpoint to verify server is running.
        
        Returns:
            {"status": "ok", "model": "smolvla-base", ...} or None
        """
        if self._session is None or self._session.closed:
            logger.warning("Session not active, call await start() first")
            return None
        
        try:
            async with self._session.get(
                f"{self.endpoint_url}/health",
                timeout=aiohttp.ClientTimeout(total=3.0)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"/health returned {resp.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning("Health check timeout")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Health check error: {e}")
            return None
    
    def _encode_image(self, rgb_array: np.ndarray) -> str:
        """
        Encode numpy RGB array to base64 JPEG string.
        
        Args:
            rgb_array: shape [H, W, 3], uint8, RGB order
        
        Returns:
            base64-encoded JPEG string (safe for JSON transmission)
        """
        # Ensure uint8
        rgb = rgb_array.astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(rgb)
        
        # Resize to 224×224 for SmolVLA
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Encode to JPEG buffer
        buf = io.BytesIO()
        img_resized.save(buf, format='JPEG', quality=85, optimize=False)
        
        # Base64 encode
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_b64
    
    def _decode_image(self, img_b64: str) -> np.ndarray:
        """Decode base64 JPEG back to numpy array (for testing)."""
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img, dtype=np.uint8)
    
    async def query_action(
        self,
        rgb_image: np.ndarray,
        instruction: str,
        current_joints: List[float]
    ) -> Optional[SmolVLAResponse]:
        """
        Query SmolVLA for action prediction.
        
        Non-blocking. Returns None on timeout/error (never raises).
        
        Args:
            rgb_image: shape [H, W, 3], uint8, RGB order
            instruction: natural language task (e.g. "pick up the red object")
            current_joints: list of N joint angles [rad]
        
        Returns:
            SmolVLAResponse with action_chunk and subgoal_xyz, or None if failed
        """
        if self._session is None or self._session.closed:
            logger.warning("Session not active")
            self._error_count += 1
            return None
        
        t_start = time.time()
        self._query_count += 1
        
        try:
            # Encode image to base64
            rgb_b64 = self._encode_image(rgb_image)
            
            # Build request payload
            payload = {
                "image_b64": rgb_b64,
                "instruction": instruction,
                "current_joints": current_joints
            }
            
            # Query endpoint (async, with timeout)
            async with self._session.post(
                f"{self.endpoint_url}/predict",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    # Parse response
                    response = SmolVLAResponse(
                        action_chunk=np.array(data["action_chunk"], dtype=np.float32),
                        subgoal_xyz=np.array(data["subgoal_xyz"], dtype=np.float32),
                        latency_ms=float(data["latency_ms"]),
                        timestamp=time.time()
                    )
                    
                    # Store latest
                    self.latest_response = response
                    
                    elapsed_ms = (time.time() - t_start) * 1000
                    logger.debug(
                        f"VLA query #{self._query_count}: "
                        f"server latency={response.latency_ms:.0f}ms, "
                        f"e2e latency={elapsed_ms:.0f}ms"
                    )
                    
                    return response
                else:
                    logger.warning(f"/predict returned {resp.status}")
                    self._error_count += 1
                    return None
        
        except asyncio.TimeoutError:
            elapsed_ms = (time.time() - t_start) * 1000
            logger.warning(
                f"VLA query timeout after {elapsed_ms:.0f}ms "
                f"(deadline={self.timeout_s*1000:.0f}ms)"
            )
            self._error_count += 1
            return None
        
        except aiohttp.ClientError as e:
            logger.warning(f"VLA query error: {type(e).__name__}: {e}")
            self._error_count += 1
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error in VLA query: {type(e).__name__}: {e}")
            self._error_count += 1
            return None
    
    def get_latest_subgoal(self) -> Optional[np.ndarray]:
        """
        Get latest subgoal from most recent VLA response.
        Thread-safe (atomic numpy read under GIL).
        
        Returns:
            [3] float32 array or None
        """
        return (
            self.latest_response.subgoal_xyz
            if self.latest_response is not None
            else None
        )
    
    def get_stats(self) -> dict:
        """Return client statistics for debugging."""
        return {
            "query_count": self._query_count,
            "error_count": self._error_count,
            "success_rate": (
                100 * (self._query_count - self._error_count) / max(self._query_count, 1)
            ),
            "latest_latency_ms": (
                self.latest_response.latency_ms
                if self.latest_response
                else None
            )
        }

"""
RealSmolVLAClient: Production-grade async HTTP client for SmolVLA server.

Uses actual ngrok or HTTP endpoint instead of mocks.
Integrates with DualSystemController for non-blocking queries.

Reference: tech spec §9 (SmolVLA Integration)
"""

import asyncio
import logging
import time
from typing import Optional, Tuple
import base64
import io

import aiohttp
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class RealSmolVLAClient:
    """
    Real async HTTP client for SmolVLA server running on ngrok or localhost.
    
    Default endpoint: https://symbolistically-unfutile-henriette.ngrok-free.dev/
    
    API Contract:
        POST /predict
        Content-Type: application/json
        Body: {
            "rgb_image_b64": "base64-encoded JPEG",
            "state": [q1, q2, q3, ...],  # Optional joint state vector
            "instruction": "pick up the cube", # Optional task text
        }
        Response: {
            "action": [a1, a2, a3, ...],  # Action chunk (list of deltas)
            "action_std": [...],           # Action standard deviation
            "latency_ms": 712.3,           # Server inference latency
        }
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        endpoint: str = "/predict",
        timeout_s: float = 5.0,
        max_retries: int = 3,
    ):
        """
        Initialize real VLA client.

        Args:
            server_url: localhost or remote endpoint (default: local production server, no trailing slash)
            endpoint: API endpoint path (default "/predict")
            timeout_s: HTTP timeout in seconds
            max_retries: Number of retries on transient failure
        """
        self.server_url = server_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.full_url = f"{self.server_url}/{self.endpoint}"
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        # Statistics
        self.call_count = 0
        self.success_count = 0
        self.fail_count = 0
        self.latency_ms = []
        self.last_action = None
        self.last_error = None

        logger.info(
            f"[RealSmolVLAClient] Initialized with endpoint: {self.full_url}"
        )

    async def predict(
        self,
        rgb_image: np.ndarray,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Query SmolVLA server for action prediction.

        Args:
            rgb_image: RGB image [H, W, 3] uint8
            state: Optional joint state vector (for conditioning)
            instruction: Optional task instruction string

        Returns:
            action: Predicted action [action_dim] as float32 array

        Raises:
            Exception: If server unavailable after max_retries
        """
        self.call_count += 1

        # Prepare request payload
        try:
            # Encode image to base64 JPEG
            pil_img = Image.fromarray(rgb_image.astype(np.uint8), mode="RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Build request body
            payload = {"rgb_image_b64": img_b64}
            if state is not None:
                payload["state"] = state.tolist() if isinstance(state, np.ndarray) else state
            if instruction is not None:
                payload["instruction"] = instruction

            # Query server with retries
            action = await self._query_with_retries(payload)
            self.success_count += 1
            self.last_action = action
            return action

        except Exception as e:
            self.fail_count += 1
            self.last_error = str(e)
            logger.error(f"[RealSmolVLAClient] Error: {e}")
            raise

    async def _query_with_retries(self, payload: dict) -> np.ndarray:
        """Query server with exponential backoff retries."""
        for attempt in range(self.max_retries):
            try:
                t_start = time.perf_counter()
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.full_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(
                                f"Server returned {response.status}: "
                                f"{await response.text()}"
                            )

                        response_json = await response.json()
                        elapsed_ms = (time.perf_counter() - t_start) * 1000
                        self.latency_ms.append(elapsed_ms)

                        # Extract action from response
                        action = np.array(response_json.get("action"), dtype=np.float32)
                        logger.debug(
                            f"[RealSmolVLAClient] Success (attempt {attempt + 1}): "
                            f"action_shape={action.shape}, latency={elapsed_ms:.1f}ms"
                        )
                        return action

            except asyncio.TimeoutError as e:
                logger.warning(
                    f"[RealSmolVLAClient] Timeout on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(
                        f"Failed to connect after {self.max_retries} attempts"
                    )

            except Exception as e:
                logger.warning(
                    f"[RealSmolVLAClient] Error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def health_check(self) -> bool:
        """
        Check if server is accessible by testing /predict endpoint.

        Returns:
            True if server responds, False otherwise
        """
        try:
            # Create a minimal test image
            test_image = np.zeros((84, 84, 3), dtype=np.uint8)
            test_state = np.zeros(7, dtype=np.float32)
            
            # Try a quick predict call instead of /health
            async with aiohttp.ClientSession() as session:
                # Encode test image
                img_pil = Image.fromarray(test_image)
                img_bytes = io.BytesIO()
                img_pil.save(img_bytes, format="JPEG")
                img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
                
                payload = {
                    "rgb_image_b64": img_b64,
                    "state": test_state.tolist(),
                    "instruction": "test",
                }
                
                async with session.post(
                    self.full_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as response:
                    if response.status == 200:
                        logger.info("[RealSmolVLAClient] Health check: OK")
                        return True
                    else:
                        logger.warning(
                            f"[RealSmolVLAClient] Health check failed: {response.status}"
                        )
                        return False
        except Exception as e:
            logger.warning(f"[RealSmolVLAClient] Health check error: {e}")
            return False

    def get_stats(self) -> dict:
        """Return statistics about queries."""
        mean_latency = np.mean(self.latency_ms) if self.latency_ms else 0.0
        max_latency = np.max(self.latency_ms) if self.latency_ms else 0.0

        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "success_rate": (
                self.success_count / self.call_count if self.call_count > 0 else 0.0
            ),
            "mean_latency_ms": float(mean_latency),
            "max_latency_ms": float(max_latency),
            "last_action_shape": (
                self.last_action.shape if self.last_action is not None else None
            ),
            "last_error": self.last_error,
        }


# Convenience function for testing
async def test_server_connectivity(
    server_url: str = "http://localhost:8000",
) -> Tuple[bool, dict]:
    """
    Test VLA server connectivity and return stats.

    Args:
        server_url: Server endpoint URL (default: local production server)

    Returns:
        (is_healthy, stats_dict)
    """
    client = RealSmolVLAClient(server_url=server_url)
    is_healthy = await client.health_check()
    return is_healthy, client.get_stats()

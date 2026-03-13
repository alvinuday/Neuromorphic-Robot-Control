"""
SmolVLA Server Client for Real Integration

This module provides a client interface to connect to a real SmolVLA server
running on Colab via ngrok tunnel. This is used for Gate 4b and 5 testing.

Unlike the mocked SmolVLAClient, this makes real HTTP requests to the actual
inference server.
"""

import asyncio
import aiohttp
import numpy as np
from typing import Optional, Dict, Any
import logging
import base64
from PIL import Image
import io
import time

logger = logging.getLogger(__name__)


class SmolVLAServerConfig:
    """Configuration for SmolVLA server connection."""
    
    def __init__(self,
                 server_url: str,
                 timeout_s: float = 2.0,
                 max_retries: int = 3):
        """
        Initialize server config.
        
        Args:
            server_url: Base URL (with ngrok tunnel, e.g., https://xxxx-ngrok-free.dev)
            timeout_s: Timeout for health checks and queries
            max_retries: Number of retries on failures
        """
        self.server_url = server_url
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.health_endpoint = f"{server_url}/health"
        self.predict_endpoint = f"{server_url}/predict"
    
    def is_valid(self) -> bool:
        """Check if configuration looks valid."""
        return (isinstance(self.server_url, str) and 
                (self.server_url.startswith('http://') or 
                 self.server_url.startswith('https://')))


class RealSmolVLAClient:
    """
    Real client for SmolVLA server.
    Makes actual HTTP requests to Colab-hosted inference server.
    """
    
    def __init__(self, config: SmolVLAServerConfig):
        """Initialize with server config."""
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.query_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0
        self.total_latency_ms = 0.0
        
        logger.info(f"Initialized RealSmolVLAClient targeting {config.server_url}")
    
    async def start(self):
        """Create aiohttp session for connection pooling."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"Started session to {self.config.server_url}")
    
    async def stop(self):
        """Close aiohttp session."""
        if self.session is not None:
            await self.session.close()
            self.session = None
            logger.info("Stopped session")
    
    async def health_check(self) -> bool:
        """
        Check if server is healthy.
        
        Returns:
            True if /health returns 200 OK, False otherwise
        """
        if self.session is None:
            logger.warning("No session, call start() first")
            return False
        
        try:
            start = time.time()
            async with self.session.get(self.config.health_endpoint) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    latency = (time.time() - start) * 1000
                    logger.debug(f"Health check OK ({latency:.1f}ms): {data}")
                    return True
                else:
                    logger.warning(f"Health check failed: {resp.status}")
                    return False
        except asyncio.TimeoutError:
            logger.warning("Health check timed out")
            return False
        except Exception as e:
            logger.warning(f"Health check error: {e}")
            return False
    
    async def query_action(self,
                          rgb: np.ndarray,
                          joints: Optional[np.ndarray] = None,
                          instruction: str = "reach forward") -> Optional[Dict[str, Any]]:
        """
        Query VLA server for action.
        
        Args:
            rgb: RGB image [H, W, 3] uint8
            joints: Joint state [6] float32 (optional, ignored by server)
            instruction: Task description string
        
        Returns:
            Dict with 'action' and 'latency_ms' keys, or None on error
        """
        if self.session is None:
            logger.warning("No session, call start() first")
            return None
        
        self.query_count += 1
        
        try:
            # Encode RGB image
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
            
            if rgb.shape[0] != 224 or rgb.shape[1] != 224:
                # Resize to 224x224
                img = Image.fromarray(rgb)
                img = img.resize((224, 224), Image.LANCZOS)
                rgb = np.array(img, dtype=np.uint8)
            
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            rgb_b64 = base64.b64encode(buf.getvalue()).decode()
            
            # Query server
            start = time.time()
            payload = {"rgb_image_b64": rgb_b64}
            
            async with self.session.post(self.config.predict_endpoint, json=payload) as resp:
                latency = (time.time() - start) * 1000
                
                if resp.status == 200:
                    data = await resp.json()
                    self.success_count += 1
                    self.total_latency_ms += latency
                    
                    logger.debug(f"Query success ({latency:.1f}ms)")
                    
                    return {
                        "action": data.get("action", [0.0, 0.0, 0.0, 0.0]),
                        "action_std": data.get("action_std", [0.1, 0.1, 0.1, 0.15]),
                        "latency_ms": latency,
                    }
                else:
                    self.failure_count += 1
                    error_text = await resp.text()
                    logger.warning(f"Query failed ({resp.status}): {error_text[:200]}")
                    return None
        
        except asyncio.TimeoutError:
            self.timeout_count += 1
            self.failure_count += 1
            logger.warning(f"Query timeout ({self.config.timeout_s}s)")
            return None
        
        except Exception as e:
            self.failure_count += 1
            logger.warning(f"Query error: {e}")
            return None
    
    def get_success_rate(self) -> float:
        """Get success rate (successful queries / total queries)."""
        if self.query_count == 0:
            return 0.0
        return self.success_count / self.query_count
    
    def get_mean_latency(self) -> float:
        """Get mean latency in ms."""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "query_count": self.query_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "success_rate": self.get_success_rate(),
            "mean_latency_ms": self.get_mean_latency(),
            "total_latency_ms": self.total_latency_ms,
        }

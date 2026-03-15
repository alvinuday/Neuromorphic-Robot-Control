#!/usr/bin/env python3
"""
VLA Server Warm-Start Debugging & Memory Profiling

Diagnoses hangs/resource exhaustion in vla_production_server.py
Tests long-running query sequences to identify memory leaks.
"""

import asyncio
import json
import logging
import time
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class VLAWarmStartDebugger:
    """Debug VLA server resource usage and request handling."""
    
    def __init__(self, vla_endpoint: str = "http://localhost:8000"):
        self.endpoint = vla_endpoint
        self.request_count = 0
        self.timeout_count = 0
        self.error_count = 0
        logger.info(f"Initializing VLA debugger with endpoint: {self.endpoint}")
    
    async def health_check(self) -> bool:
        """Check if VLA server is responsive."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.endpoint}/health")
                is_healthy = response.status_code == 200
                logger.info(f"Health check: {response.status_code} {'✅' if is_healthy else '❌'}")
                return is_healthy
        except asyncio.TimeoutError:
            logger.error("Health check TIMEOUT")
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def single_request(self, request_id: int, timeout_sec: float = 2.0) -> bool:
        """
        Make a single prediction request and track response time.
        
        Returns: True if successful, False if timeout/error
        """
        import httpx
        import base64
        
        try:
            # Create dummy RGB image (84x84x3)
            dummy_rgb = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
            
            # Encode to JPEG and base64
            from PIL import Image
            import io
            
            img = Image.fromarray(dummy_rgb, mode='RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            rgb_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            # Build payload
            payload = {
                "rgb_image_b64": rgb_b64,
                "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "instruction": "pick up the object",
                "language_tokens": None,
            }
            
            # Make request with timeout
            t_start = time.perf_counter()
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                response = await client.post(
                    f"{self.endpoint}/predict",
                    json=payload,
                )
            latency_ms = (time.perf_counter() - t_start) * 1000
            
            if response.status_code == 200:
                logger.info(f"Request {request_id}: ✅ {latency_ms:.1f}ms")
                self.request_count += 1
                return True
            else:
                logger.warning(f"Request {request_id}: ⚠️ Status {response.status_code}")
                self.error_count += 1
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id}: ❌ TIMEOUT (>{timeout_sec}s)")
            self.timeout_count += 1
            return False
        except Exception as e:
            logger.error(f"Request {request_id}: ❌ ERROR {type(e).__name__}: {str(e)[:50]}")
            self.error_count += 1
            return False
    
    async def stress_test(self, num_requests: int = 20, pause_between_sec: float = 0.5):
        """
        Run sequential requests with pauses to detect resource leaks.
        
        If server hangs, suggests:
        - Memory accumulation (need GC)
        - Socket leak (need connection closing)
        - Queue overflow (need request batching)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"VLA STRESS TEST: {num_requests} sequential requests")
        logger.info(f"{'='*80}")
        
        # Initial health check
        healthy = await self.health_check()
        if not healthy:
            logger.error("VLA server not healthy. Aborting.")
            return
        
        logger.info(f"Starting {num_requests} requests with {pause_between_sec}s pause...")
        
        for i in range(num_requests):
            logger.info(f"\n[{i+1}/{num_requests}] Making request...")
            
            success = await self.single_request(i+1, timeout_sec=10.0)
            
            if not success:
                logger.warning(f"⚠️  Request {i+1} failed or timed out")
                
                # Check health after failure
                await asyncio.sleep(2)
                health = await self.health_check()
                
                if not health:
                    logger.error(f"❌ Server became unresponsive after request {i+1}")
                    logger.error("DIAGNOSIS: Possible out-of-memory, CUDA OOM, or process crash")
                    break
            
            # Pause between requests
            if i < num_requests - 1:
                logger.info(f"Pausing {pause_between_sec}s before next request...")
                await asyncio.sleep(pause_between_sec)
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info("STRESS TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total requests: {self.request_count + self.error_count + self.timeout_count}")
        logger.info(f"  ✅ Successful:  {self.request_count}")
        logger.info(f"  ⚠️  Errors:      {self.error_count}")
        logger.info(f"  ❌ Timeouts:    {self.timeout_count}")
        
        if self.timeout_count > 0:
            logger.error("\n🚨 TIMEOUT DETECTED")
            logger.error("   This is the issue blocking ablation tests.")
            logger.error("   See AGENT_STATE.md 'Proposed Fix' section for solutions.")
        else:
            logger.success("\n✅ No timeouts detected - VLA server resource-stable")

async def main():
    """Run VLA warm-start debugger."""
    debugger = VLAWarmStartDebugger()
    
    # Run 20 sequential requests (simulates ~2-3 benchmark episodes worth of load)
    await debugger.stress_test(num_requests=20, pause_between_sec=1.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nDebug interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

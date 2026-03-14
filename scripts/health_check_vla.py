#!/usr/bin/env python3
"""Quick VLA server health check before benchmarking"""

import asyncio
import aiohttp
import base64
import numpy as np
import cv2
import sys
import time

async def quick_health_check():
    """Quick health check of VLA server"""
    try:
        async with aiohttp.ClientSession() as session:
            # Test simple request
            dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            _, jpg_buf = cv2.imencode('.jpg', dummy_rgb)
            rgb_b64 = base64.b64encode(jpg_buf).decode('utf-8')
            
            payload = {
                'rgb_image_b64': rgb_b64,
                'state': [0.0, 0.0, 0.0, 0.0],
                'instruction': 'test quick health'
            }
            
            t_start = time.time()
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.post(
                'http://localhost:8000/predict',
                json=payload,
                timeout=timeout
            ) as resp:
                elapsed = time.time() - t_start
                if resp.status == 200:
                    data = await resp.json()
                    print(f'✅ VLA Server Health: OK ({elapsed:.2f}s)')
                    print(f'   Response keys: {list(data.keys())}')
                    if 'action' in data:
                        print(f'   Action dimension: {len(data["action"])}')
                    if 'latency_ms' in data:
                        print(f'   Latency: {data["latency_ms"]:.1f}ms')
                    return True
                else:
                    print(f'❌ Status: {resp.status}')
                    text = await resp.text()
                    print(f'   Response: {text[:200]}')
                    return False
    except asyncio.TimeoutError:
        print('❌ Server timeout (10s)')
        return False
    except Exception as e:
        print(f'❌ Error: {type(e).__name__}: {e}')
        return False

if __name__ == "__main__":
    result = asyncio.run(quick_health_check())
    sys.exit(0 if result else 1)

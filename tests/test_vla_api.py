#!/usr/bin/env python3
"""
Test VLA API thoroughly to identify issues
"""
import asyncio
import base64
import json
import numpy as np
from pathlib import Path
import aiohttp
import cv2

# Use very long timeout for first inference (model warmup can take 4+ minutes)
LONG_TIMEOUT = aiohttp.ClientTimeout(total=600)  # 10 minutes for initial inference
SHORT_TIMEOUT = aiohttp.ClientTimeout(total=120)  # 2 minutes for subsequent requests

async def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Endpoint")
    print("="*60)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/health", timeout=SHORT_TIMEOUT) as resp:
                data = await resp.json()
                print(f"✓ Status: {resp.status}")
                print(f"✓ Response: {json.dumps(data, indent=2)}")
                return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

async def test_minimal_predict():
    """Test minimal predict request with minimal data"""
    print("\n" + "="*60)
    print("TEST 2: Minimal Predict (just images)")
    print("="*60)
    
    # Create dummy RGB image
    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    _, jpg_buf = cv2.imencode('.jpg', dummy_rgb)
    rgb_b64 = base64.b64encode(jpg_buf).decode('utf-8')
    
    payload = {
        "rgb_image_b64": rgb_b64,
        "instruction": "pick up the object"
    }
    
    print(f"Payload keys: {list(payload.keys())}")
    print("⏳ Waiting for response (first inference may take 3-5 minutes)...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://localhost:8000/predict", 
                json=payload,
                timeout=LONG_TIMEOUT  # Use long timeout for first inference
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Status: {resp.status}")
                    print(f"✓ Response: {json.dumps(data, indent=2)}")
                else:
                    print(f"✗ Status: {resp.status}")
                    data = await resp.json()
                    print(f"✗ Response: {json.dumps(data, indent=2)}")
        except asyncio.TimeoutError:
            print(f"✗ Error: Request timed out after 10 minutes - model may be stuck")
        except Exception as e:
            print(f"✗ Error: {e}")

async def test_with_state():
    """Test predict with state"""
    print("\n" + "="*60)
    print("TEST 3: Predict with State")
    print("="*60)
    
    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    _, jpg_buf = cv2.imencode('.jpg', dummy_rgb)
    rgb_b64 = base64.b64encode(jpg_buf).decode('utf-8')
    
    payload = {
        "rgb_image_b64": rgb_b64,
        "state": [0.0, 0.0, 0.0, 0.0],
        "instruction": "pick up the object"
    }
    
    print(f"Payload keys: {list(payload.keys())}")
    print("⏳ Waiting for response...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://localhost:8000/predict", 
                json=payload,
                timeout=SHORT_TIMEOUT  # Subsequent requests should be faster
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Status: {resp.status}")
                    print(f"✓ Response: {json.dumps(data, indent=2)}")
                else:
                    print(f"✗ Status: {resp.status}")
                    data = await resp.json()
                    print(f"✗ Error: {json.dumps(data, indent=2)}")
        except asyncio.TimeoutError:
            print(f"✗ Error: Request timed out after 120 seconds")
        except Exception as e:
            print(f"✗ Error: {e}")

async def test_with_language_tokens():
    """Test predict with language tokens (what it's asking for)"""
    print("\n" + "="*60)
    print("TEST 4: Predict with Language Tokens (expected by model)")
    print("="*60)
    
    dummy_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    _, jpg_buf = cv2.imencode('.jpg', dummy_rgb)
    rgb_b64 = base64.b64encode(jpg_buf).decode('utf-8')
    
    # Try with language tokens
    payload = {
        "rgb_image_b64": rgb_b64,
        "state": [0.0, 0.0, 0.0, 0.0],
        "instruction": "pick up the object",
        "language_tokens": [1, 2, 3, 4, 5]  # Dummy tokens
    }
    
    print(f"Payload keys: {list(payload.keys())}")
    print("⏳ Waiting for response...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "http://localhost:8000/predict", 
                json=payload,
                timeout=SHORT_TIMEOUT
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✓ Status: {resp.status}")
                    print(f"✓ Response: {json.dumps(data, indent=2)}")
                else:
                    print(f"✗ Status: {resp.status}")
                    data = await resp.json()
                    print(f"✗ Error: {json.dumps(data, indent=2)}")
        except asyncio.TimeoutError:
            print(f"✗ Error: Request timed out after 120 seconds")
        except Exception as e:
            print(f"✗ Error: {e}")

async def main():
    print("\n" + "="*60)
    print("VLA API TESTING SUITE")
    print("="*60)
    
    # Test 1: Health
    health_ok = await test_health()
    if not health_ok:
        print("\n✗ Health check failed, server not running")
        return
    
    # Test 2: Minimal
    await test_minimal_predict()
    
    # Test 3: With state
    await test_with_state()
    
    # Test 4: With language tokens
    await test_with_language_tokens()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

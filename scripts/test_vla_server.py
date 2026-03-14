#!/usr/bin/env python3
"""
Comprehensive VLA Server Connectivity & Prediction Test

Tests both ngrok endpoint and localhost alternatives with detailed diagnostics.
"""

import asyncio
import aiohttp
import requests
import numpy as np
from PIL import Image
import base64
import io
import json
import time


def test_health_check_sync(url: str, endpoint: str = "/predict") -> dict:
    """Test health with synchronous requests (simpler debugging)."""
    print(f"\n{'='*70}")
    print(f"SYNC HEALTH CHECK: {url}")
    print('='*70)
    
    results = {
        "url": url,
        "endpoint": endpoint,
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }
    
    # Test 1: Simple GET to root
    print(f"\n[1] Testing GET {url}/")
    try:
        resp = requests.get(f"{url}/", timeout=5)
        results["tests"]["GET_root"] = {
            "status": resp.status_code,
            "text": resp.text[:100] if resp.text else "No content"
        }
        print(f"    ✓ Status: {resp.status_code}")
    except Exception as e:
        results["tests"]["GET_root"] = {"error": str(e)}
        print(f"    ✗ Error: {type(e).__name__}: {e}")
    
    # Test 2: Dummy POST to /predict
    print(f"\n[2] Testing POST {url}{endpoint}")
    test_image = np.zeros((84, 84, 3), dtype=np.uint8)
    test_state = np.zeros(7, dtype=np.float32)
    
    img_pil = Image.fromarray(test_image)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="JPEG")
    img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    
    payload = {
        "rgb_image_b64": img_b64,
        "state": test_state.tolist(),
        "instruction": "test pick",
    }
    
    try:
        start = time.perf_counter()
        resp = requests.post(
            f"{url}{endpoint}",
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"    ✓ Status: {resp.status_code} (latency: {elapsed:.1f}ms)")
        results["tests"]["POST_predict"] = {
            "status": resp.status_code,
            "latency_ms": elapsed,
            "response": resp.json() if resp.headers.get('content-type') == 'application/json' else resp.text[:100]
        }
        
        if resp.status_code == 200:
            data = resp.json()
            if "action" in data:
                action = data["action"]
                print(f"    ✓ Action shape: {len(action)} dims")
                results["tests"]["POST_predict"]["action_dims"] = len(action)
    except requests.exceptions.Timeout as e:
        print(f"    ✗ TIMEOUT: {e}")
        results["tests"]["POST_predict"] = {"error": "timeout", "details": str(e)}
    except requests.exceptions.ConnectionError as e:
        print(f"    ✗ CONNECTION ERROR: {e}")
        results["tests"]["POST_predict"] = {"error": "connection", "details": str(e)}
    except Exception as e:
        print(f"    ✗ Error: {type(e).__name__}: {e}")
        results["tests"]["POST_predict"] = {"error": type(e).__name__, "details": str(e)}
    
    return results


async def test_health_check_async(url: str, endpoint: str = "/predict") -> dict:
    """Test health with async aiohttp (what our code uses)."""
    print(f"\n{'='*70}")
    print(f"ASYNC HEALTH CHECK: {url}")
    print('='*70)
    
    results = {
        "url": url,
        "endpoint": endpoint,
        "async": True,
        "tests": {}
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test 1: Dummy POST
            print(f"\n[1] Testing async POST {url}{endpoint}")
            test_image = np.zeros((84, 84, 3), dtype=np.uint8)
            test_state = np.zeros(7, dtype=np.float32)
            
            img_pil = Image.fromarray(test_image)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="JPEG")
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
            
            payload = {
                "rgb_image_b64": img_b64,
                "state": test_state.tolist(),
                "instruction": "test pick",
            }
            
            try:
                start = time.perf_counter()
                async with session.post(
                    f"{url}{endpoint}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    elapsed = (time.perf_counter() - start) * 1000
                    
                    print(f"    ✓ Status: {response.status} (latency: {elapsed:.1f}ms)")
                    
                    text = await response.text()
                    results["tests"]["POST_predict"] = {
                        "status": response.status,
                        "latency_ms": elapsed,
                        "response_length": len(text)
                    }
                    
                    if response.status == 200:
                        try:
                            data = json.loads(text)
                            if "action" in data:
                                action = data["action"]
                                print(f"    ✓ Action shape: {len(action)} dims")
                                results["tests"]["POST_predict"]["action_dims"] = len(action)
                        except json.JSONDecodeError:
                            print(f"    ⚠ Response not JSON: {text[:100]}")
                            results["tests"]["POST_predict"]["parse_error"] = "not json"
                    
            except asyncio.TimeoutError as e:
                print(f"    ✗ ASYNC TIMEOUT: {e}")
                results["tests"]["POST_predict"] = {"error": "timeout", "details": str(e)}
            except aiohttp.ClientError as e:
                print(f"    ✗ AIOHTTP ERROR: {e}")
                results["tests"]["POST_predict"] = {"error": "aiohttp", "details": str(e)}
                
    except Exception as e:
        print(f"✗ Session error: {type(e).__name__}: {e}")
        results["tests"]["session"] = {"error": type(e).__name__, "details": str(e)}
    
    return results


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VLA SERVER CONNECTIVITY & PREDICTION TEST")
    print("="*70)
    
    all_results = {}
    
    # Test endpoints
    endpoints = [
        ("NGROK (Current)", "https://symbolistically-unfutile-henriette.ngrok-free.dev", "/predict"),
        ("Localhost (Alternative)", "http://localhost:8001", "/predict"),
        ("Local Alt 2", "http://127.0.0.1:8001", "/predict"),
    ]
    
    # Synchronous tests
    print("\n" + "█"*70)
    print("SYNCHRONOUS TESTS (requests library)")
    print("█"*70)
    
    for name, url, endpoint in endpoints:
        print(f"\n[Testing: {name}]")
        result = test_health_check_sync(url, endpoint)
        all_results[name] = result
        time.sleep(1)  # Rate limit
    
    # Asynchronous tests
    print("\n\n" + "█"*70)
    print("ASYNCHRONOUS TESTS (aiohttp)")
    print("█"*70)
    
    async def run_async_tests():
        for name, url, endpoint in endpoints:
            print(f"\n[Testing: {name}]")
            result = await test_health_check_async(url, endpoint)
            all_results[f"{name}_async"] = result
            await asyncio.sleep(1)  # Rate limit
    
    asyncio.run(run_async_tests())
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    print("\nResults saved to: test_vla_results.json")
    with open("test_vla_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\nStatus Overview:")
    print("-" * 70)
    for name, result in all_results.items():
        post_test = result.get("tests", {}).get("POST_predict", {})
        status = post_test.get("status", "N/A")
        error = post_test.get("error", "-")
        latency = post_test.get("latency_ms", "-")
        
        if isinstance(latency, (int, float)):
            latency_str = f"{latency:.1f}ms"
        else:
            latency_str = str(latency)
        
        status_str = f"Status: {status}" if status != "N/A" else "Error"
        error_str = f" ({error})" if error != "-" else ""
        
        print(f"{name:30} {status_str:20} {latency_str:10} {error_str}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Diagnostic tool for SmolVLA server integration.
Tests various endpoint configurations and request formats.
"""

import requests
import json
import time
from urllib.parse import urljoin

SMOLVLA_URL = "https://symbolistically-unfutile-henriette.ngrok-free.dev"

print("="*70)
print("SmolVLA SERVER DIAGNOSTIC TOOL")
print("="*70)
print(f"\n🔍 Testing server: {SMOLVLA_URL}\n")

# Test 1: Basic connectivity
print("1️⃣  Testing basic connectivity...")
try:
    response = requests.get(SMOLVLA_URL, timeout=5, verify=False)
    print(f"   ✅ Server is reachable (status: {response.status_code})")
except Exception as e:
    print(f"   ❌ Server unreachable: {e}")

# Test 2: Health endpoint
print("\n2️⃣  Testing health endpoint...")
health_endpoints = [
    "/health",
    "/api/health",
    "/status",
    "/ping",
]

for endpoint in health_endpoints:
    try:
        url = urljoin(SMOLVLA_URL, endpoint)
        response = requests.get(url, timeout=5, verify=False)
        if response.status_code == 200:
            print(f"   ✅ {endpoint}: {response.status_code}")
            print(f"      Response: {response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:100]}")
            break
    except Exception as e:
        print(f"   ❌ {endpoint}: {type(e).__name__}")

# Test 3: Predict endpoints
print("\n3️⃣  Testing predict/inference endpoints...")
predict_endpoints = [
    "/predict",
    "/api/predict",
    "/infer",
    "/api/infer",
    "/forward",
    "/api/forward",
]

# Sample request payloads
request_formats = [
    {
        "format": "instruction + image_shape",
        "payload": {
            "instruction": "pick up the object and move it to the target",
            "image_shape": [480, 640, 3],
        }
    },
    {
        "format": "text only",
        "payload": {
            "text": "pick up the object and move it to the target",
        }
    },
    {
        "format": "instruction only",
        "payload": {
            "instruction": "pick up the object and move it to the target",
        }
    },
    {
        "format": "prompt",
        "payload": {
            "prompt": "pick up the object and move it to the target",
        }
    },
]

found_working = False

for endpoint in predict_endpoints:
    if found_working:
        break
    
    print(f"\n   Testing endpoint: {endpoint}")
    
    for req_format in request_formats:
        if found_working:
            break
        
        try:
            url = urljoin(SMOLVLA_URL, endpoint)
            
            t_start = time.time()
            response = requests.post(
                url,
                json=req_format["payload"],
                timeout=10,
                verify=False,
                headers={"Content-Type": "application/json"}
            )
            elapsed = time.time() - t_start
            
            if response.status_code in [200, 201]:
                print(f"      ✅ {req_format['format']}: {response.status_code} ({elapsed*1000:.1f}ms)")
                print(f"         Response: {str(response.text)[:100]}")
                found_working = True
            else:
                print(f"      ❌ {req_format['format']}: {response.status_code}")
                if response.text:
                    print(f"         Error: {response.text[:100]}")
        
        except requests.exceptions.Timeout:
            print(f"      ⏱️  {req_format['format']}: Timeout (>10s)")
        except Exception as e:
            print(f"      ❌ {req_format['format']}: {type(e).__name__}: {str(e)[:60]}")

# Test 4: Available endpoints
print("\n4️⃣  Attempting to discover available endpoints...")
common_endpoints = [
    "/",
    "/api",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/endpoints",
]

for endpoint in common_endpoints:
    try:
        url = urljoin(SMOLVLA_URL, endpoint)
        response = requests.get(url, timeout=5, verify=False)
        if response.status_code == 200:
            print(f"   ✅ {endpoint}: Available")
            if "json" in response.headers.get('content-type', ''):
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        keys = list(data.keys())[:5]
                        print(f"      Keys: {keys}")
                except:
                    pass
    except:
        pass

# Test 5: Check for CORS/ngrok headers
print("\n5️⃣  Checking response headers...")
try:
    response = requests.get(SMOLVLA_URL, timeout=5, verify=False)
    important_headers = ['Server', 'X-Forwarded-For', 'X-Ngrok', 'Access-Control-Allow-Origin', 'Content-Type']
    for header in important_headers:
        if header in response.headers:
            print(f"   {header}: {response.headers[header]}")
except Exception as e:
    print(f"   ❌ Could not read headers: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

if not found_working:
    print("""
🔧 RECOMMENDATIONS:
   1. Verify ngrok tunnel is still active: pgrep -f ngrok
   2. Check server logs for errors
   3. Try accessing server directly in browser
   4. Verify server expects POST with JSON content-type
   5. Check if server requires authentication headers
   6. Consider if server is expecting multipart/form-data with image file
   7. Look for server documentation or API specification
    """)
else:
    print("\n✅ Found working endpoint! Update the benchmark script with the correct format.")

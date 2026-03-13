#!/usr/bin/env python3
"""Direct server endpoint test."""

import requests
import numpy as np
import io
import base64
from PIL import Image
import sys

url = "https://symbolistically-unfutile-henriette.ngrok-free.dev"

print(f"Testing server endpoints...")
print(f"URL: {url}\n")

# Test 1: Health check
try:
    resp = requests.get(f"{url}/health", timeout=5)
    print(f"✅ Health check: {resp.status_code}")
    print(f"   Response: {resp.json()}\n")
except Exception as e:
    print(f"❌ Health check failed: {e}\n")
    sys.exit(1)

# Test 2: Predict endpoint with small image
try:
    # Create test image (224x224)
    rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    rgb_b64 = base64.b64encode(buf.getvalue()).decode()
    
    print(f"Sending prediction request...")
    print(f"Image size: 224x224")
    print(f"Payload size: {len(rgb_b64)} chars\n")
    
    resp = requests.post(
        f"{url}/predict",
        json={"rgb_image_b64": rgb_b64},
        timeout=15
    )
    print(f"Response status: {resp.status_code}")
    
    if resp.status_code == 200:
        print(f"✅ SUCCESS! Server is working!")
        result = resp.json()
        print(f"Response: {result}")
    else:
        print(f"❌ Server error: {resp.status_code}")
        print(f"Response (first 500 chars):\n{resp.text[:500]}")
        
except Exception as e:
    print(f"❌ Predict failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

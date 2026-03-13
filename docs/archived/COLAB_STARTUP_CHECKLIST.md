# Colab SmolVLA Server Startup Instructions

## Current Notebook Status

✅ **Already Executed (Cells 1-18):**
- Dependencies installed
- GPU verified
- SmolVLA model loaded (450M params)
- FastAPI endpoints defined
- ngrok authentication configured
- Server listening on port 8001

⏳ **Still Need to Execute (Cells 20-22):**
- Create ngrok tunnel
- Start uvicorn server
- Test endpoints

---

## IMMEDIATE ACTIONS REQUIRED

### Step 1: Open Colab Notebook

1. Go to: https://colab.research.google.com
2. Click "File" → "Open notebook"
3. Search for: `vla/smolvla_server.ipynb` or upload it
4. Verify GPU is selected: **Runtime → Change runtime type → T4 GPU**

### Step 2: Run the Missing Cells

In the Colab notebook, scroll down to **Cell 20** ("Setup ngrok and Run Server"):

```python
# Keep notebook running (infinite loop in Colab)
import time
print("Server running. Press Ctrl+C to stop.")
print(f"ngrok URL: {smolvla_url}")

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\nServer stopped.")
```

Click the **▶ Run cell** button for:
- **Cell 20:** Setup ngrok
- **Cell 21:** Start uvicorn server  
- **Cell 22:** Test endpoints locally

### Step 3: Copy the ngrok URL

After executing Cell 21, look for output like:

```
🌐 PUBLIC SERVER URL (use in local client):
   https://xxxx-xxxx-ngrok-free.dev
   
✓ Server exposed at: https://xxxx-xxxx-ngrok-free.dev
```

**Copy this URL exactly** (including `https://`)

### Step 4: Set Environment Variable

In your terminal (on this machine):

```bash
# Replace xxxx-xxxx with your actual ngrok subdomain
export SMOLVLA_SERVER_URL="https://xxxx-xxxx-ngrok-free.dev"

# Verify it's set:
echo $SMOLVLA_SERVER_URL
```

### Step 5: Run Tests Locally

Once the server is running and URL is set:

```bash
cd ~/Documents/Alvin/College/Academics/Master\'s\ Thesis/Code/Neuromorphic-Robot-Control

# Test Gate 4b (Real SmolVLA integration)
echo "🧪 Running Gate 4b tests..."
python3 -m pytest tests/test_integration_real_smolvla.py -v -s

# Test Gate 5 (Full E2E system)
echo "🧪 Running Gate 5 tests..."
python3 -m pytest tests/test_e2e_gate5.py -v -s

# Or run both:
echo "🧪 Running both Gate 4b & 5..."
python3 -m pytest tests/test_integration_real_smolvla.py tests/test_e2e_gate5.py -v -s
```

---

## Quick Reference: What Each Cell Does

| Cell | Purpose | Status |
|------|---------|--------|
| 2 | Install dependencies | ✅ Done |
| 3 | Verify GPU | ✅ Done |
| 4 | Load SmolVLA model | ✅ Done |
| 5 | Define FastAPI endpoints | ✅ Done |
| 6 | Configure ngrok auth | ✅ Done |
| 7 | Start uvicorn server | ✅ Done |
| 8 | Create ngrok tunnel | ✅ Done |
| 9 | Test health endpoint | ✅ Done |
| 10-18 | Debug and fix issues | ✅ Done |
| **20** | **Setup ngrok again** | **⏳ NEXT** |
| **21** | **Start fresh server** | **⏳ NEXT** |
| **22** | **Test endpoints** | **⏳ OPTIONAL** |

---

## Expected Colab Output

After executing cells 20-21, you should see:

```
✓ ngrok token configured
✓ Server running on port 8001

================================================== 
🌐 PUBLIC SERVER URL (use in local client):
   https://4a8c-2600-1700-xxxx-xxxx-ngrok-free.dev
================================================== 

✓ Server exposed at: https://4a8c-2600-1700-xxxx-xxxx-ngrok-free.dev
  /health endpoint: https://4a8c-2600-1700-xxxx-xxxx-ngrok-free.dev/health
  /predict endpoint: https://4a8c-2600-1700-xxxx-xxxx-ngrok-free.dev/predict

Server running. Press Ctrl+C to stop.
ngrok URL: https://4a8c-2600-1700-xxxx-xxxx-ngrok-free.dev
```

---

## Troubleshooting

**Error: "ngrok.error.PyngrokNgrokHTTPError"**
- The ngrok auth token in Cell 6 might be expired
- Get a new token: https://dashboard.ngrok.com/get-started/your-authtoken
- Update Cell 6 and re-run

**Error: "CUDA out of memory"**
- This shouldn't happen (model = 450M, T4 = 16GB)
- Try restarting the Colab kernel (Runtime → Restart runtime)

**Error: "Connection refused" from local tests**
- Make sure Cell 21 is running (look for "Server running")
- Check the ngrok URL is correct
- Try: `curl $SMOLVLA_SERVER_URL/health` locally (should work with ngrok)

---

## Keep Notebook Running

The Colab notebook will stay active for **12 hours** as long as:
1. The browser tab is open
2. You don't close the kernelThe notebook remains running in the background
4. ngrok tunnel stays active

You can minimize the tab during testing.

---

## Once Server is Running

Once you have the ngrok URL, come back here and:

1. ✅ Set the environment variable
2. ✅ Run the test commands below
3. ✅ Watch for test results

**Tests will automatically:**
- Connect to your Colab server
- Run 13 Gate 4b tests (real VLA integration)
- Run 12 Gate 5 tests (full E2E system)
- Report results in ~5-10 minutes

---

## Manual Health Check (Optional)

While Colab is running, you can test the server manually:

```bash
# Replace the URL with your actual ngrok URL
NGROK_URL="https://xxxx-xxxx-ngrok-free.dev"

# Health check
curl -X GET "$NGROK_URL/health"

# Simple prediction test (if you want to manually try)
# This requires encoding an image, so easier to use pytest
```

---

**Status:** Ready to start the server ✨  
**Next step:** Run cells 20-21 in Colab, then come back with the ngrok URL

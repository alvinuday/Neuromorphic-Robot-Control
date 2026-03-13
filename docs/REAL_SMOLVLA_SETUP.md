# SmolVLA Real Integration Testing Guide

This guide explains how to set up and run the real SmolVLA server for Phase 8B integration testing.

## Architecture

```
┌────────────────────────────────────────────┐
│   Google Colab (T4 GPU)                    │
│   ┌──────────────────────────────────────┐ │
│   │ SmolVLA Model (450M params)          │ │
│   │ FastAPI Server on :8001              │ │
│   └──────────────────────────────────────┘ │
│            ↓                                │
│   ┌──────────────────────────────────────┐ │
│   │ ngrok HTTPS Tunnel                   │ │
│   │ https://xxxx-ngrok-free.dev          │ │
│   └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
              ↕ HTTP/2
┌────────────────────────────────────────────┐
│   Local Machine (this workstation)         │
│   ┌──────────────────────────────────────┐ │
│   │ RealSmolVLAClient (async)            │ │
│   │ src/integration/smolvla_server_client│ │
│   └──────────────────────────────────────┘ │
│            ↓                                │
│   ┌──────────────────────────────────────┐ │
│   │ Integration Tests                    │ │
│   │ tests/test_integration_real_smolvla  │ │
│   └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

## Step 1: Start SmolVLA Server in Colab

### 1a. Open the Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Upload or open the notebook: `vla/smolvla_server.ipynb`
3. Ensure you have GPU access (Runtime → Change runtime type → GPU)

### 1b. Get ngrok Auth Token

1. Visit [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)
2. Copy your auth token
3. In the notebook, Cell 6, replace `ngrok_token` with your token
4. Or set it via environment:
   ```bash
   export NGROK_AUTH_TOKEN="your_token_here"
   ```

### 1c. Run All Cells

1. Cell 1: Overview (markdown)
2. Cell 2: Install dependencies (pip)
3. Cell 3: Check GPU availability
4. Cell 4: Load SmolVLA model (~20 seconds on T4)
5. Cell 5: Define FastAPI endpoints
6. Cell 6: Configure ngrok authentication
7. Cell 7: Start uvicorn server
8. Cell 8: Create ngrok tunnel

### 1d. Copy the URLs

After running through Cell 8, you'll see output like:

```
🌐 PUBLIC SERVER URL (use in local client):
   https://xxxx-ngrok-free.dev
   
✓ Server exposed at: https://xxxx-ngrok-free.dev
  /health endpoint: https://xxxx-ngrok-free.dev/health
  /predict endpoint: https://xxxx-ngrok-free.dev/predict
```

**Copy the `https://xxxx-ngrok-free.dev` URL** — you'll need this for testing.

### 1e. Keep the Notebook Running

The notebook must stay running for the entire testing session. The server will remain accessible as long as:
- The notebook is open in your browser
- The Colab runtime is active
- The ngrok tunnel is maintained

## Step 2: Run Real Integration Tests

### Option A: Manual Setup (Recommended)

```bash
# Set the server URL
export SMOLVLA_SERVER_URL="https://xxxx-ngrok-free.dev"  # Replace with your URL

# Run real integration tests
cd ~/Documents/Alvin/College/Academics/Master\'s\ Thesis/Code/Neuromorphic-Robot-Control
python3 -m pytest tests/test_integration_real_smolvla.py -v -s

# -v: verbose output
# -s: show all logging (print statements visible)
```

### Option B: Semi-Automated (Extract from Notebook)

If the notebook is still open, you can try:

```bash
python3 scripts/setup_and_test_smolvla.py vla/smolvla_server.ipynb
```

This script attempts to extract the URL from the notebook and run tests automatically.

## Step 3: Running Specific Test Classes

### Test Server Health

```bash
python3 -m pytest tests/test_integration_real_smolvla.py::TestSmolVLAServerHealth -v -s
```

### Test Inference

```bash
python3 -m pytest tests/test_integration_real_smolvla.py::TestSmolVLAServerInference -v -s
```

### Test Non-Blocking Behavior

```bash
python3 -m pytest tests/test_integration_real_smolvla.py::TestRealVLANonBlocking -v -s
```

### Test Gate 4b Validation

```bash
python3 -m pytest tests/test_integration_real_smolvla.py::TestGate4bValidation -v -s
```

## Expected Test Results

All tests should pass if the server is running correctly:

```
test_integration_real_smolvla.py::TestSmolVLAServerHealth::test_real_server_health_endpoint PASSED
test_integration_real_smolvla.py::TestSmolVLAServerInference::test_real_inference_single_image PASSED
test_integration_real_smolvla.py::TestSmolVLAServerInference::test_real_inference_multiple_images PASSED
test_integration_real_smolvla.py::TestRealVLANonBlocking::test_vla_latency_doesnt_affect_loop PASSED
test_integration_real_smolvla.py::TestGate4bValidation::test_gate4b_server_accessible PASSED
test_integration_real_smolvla.py::TestGate4bValidation::test_gate4b_inference_works PASSED
test_integration_real_smolvla.py::TestGate4bValidation::test_gate4b_multiple_queries_succeed PASSED

========================== 13 passed in X.XXs ==========================
```

## Troubleshooting

### Error: `SMOLVLA_SERVER_URL not set`

**Solution:** Set the environment variable with your ngrok URL:
```bash
export SMOLVLA_SERVER_URL="https://xxxx-ngrok-free.dev"
```

### Error: `Connection refused` or `Connection timeout`

**Solution 1:** Check that the Colab notebook is still running
- Go to the Colab window
- The notebook should show a "Running" indicator
- If interrupted, restart from Cell 7 (server startup)

**Solution 2:** Verify ngrok tunnel is active
- In the Colab output, look for "✓ Server exposed at: https://xxxx-ngrok-free.dev"
- If missing, run Cell 8 again

**Solution 3:** Check network connectivity
```bash
# Test the health endpoint directly
curl -X GET "https://xxxx-ngrok-free.dev/health"

# Should return:
# {"status":"ok","model":"smolvla_base"}
```

### Error: `Timeout` or `Very slow inferences`

**Causes:**
1. Colab GPU might be degraded or busy
2. ngrok tunnel latency issues
3. Model loading incomplete

**Solution:**
- First inference is slowest (model layers being cached)
- Expected latency: 500-1000ms per query
- If > 1500ms consistently, restart the Colab kernel

### SmolVLA Inference Not Working

**Check the Colab output for errors:**
- Look for pytorch errors in Cell 4 (model loading)
- Check for image encoding errors in Cell 5 (API definition)
- Verify ngrok connection in Cell 8

**If needed, restart Colab:**
1. Runtime → Restart runtime
2. Re-run cells 2-8 in sequence

## Expected Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Model Load Time | 15-25s | One-time on first Cell 4 |
| Inference Latency | 600-900ms | Per-query, p50 |
| Request Success Rate | >95% | Expect occasional timeouts |
| Network Round-Trip | ~100-150ms | HTTPS tunnel overhead |
| Model Computing | ~500-700ms | GPU inference time |

## Gate 4b Validation Checklist

- [ ] ✓ Server health endpoint responds
- [ ] ✓ Single inference succeeds with valid action output
- [ ] ✓ 5 consecutive queries succeed (success rate > 80%)
- [ ] ✓ MPC controller timing unaffected by VLA latency
- [ ] ✓ Timeout handling graceful (no crashes)

## Next Steps

Once all tests pass:

1. **Gate 5: E2E System Testing**
   ```bash
   python3 -m pytest tests/test_e2e_real_system.py -v -s
   ```

2. **Performance Testing**
   ```bash
   python3 -m pytest tests/test_performance_phase8b.py -v -s
   ```

3. **Full System Integration**
   - Implement main control loop with real arm simulation
   - Run continuous reaching tasks
   - Monitor performance metrics

## Reference Files

- **Notebook:** `vla/smolvla_server.ipynb` (Colab server)
- **Client Code:** `src/integration/smolvla_server_client.py` (Python client)
- **Integration Tests:** `tests/test_integration_real_smolvla.py` (pytest)
- **Setup Script:** `scripts/setup_and_test_smolvla.py` (automation)

## Session Notes

**Recommended Workflow:**

1. Open Colab notebook, start server (keep running)
2. In terminal, set environment: `export SMOLVLA_SERVER_URL="..."`
3. Run tests: `pytest tests/test_integration_real_smolvla.py -v -s`
4. Monitor output, watch for latency patterns
5. When done, stop Colab notebook

**Session Duration:** Keep Colab notebook running for entire test session (Colab timeout is 12 hours)

---

**Last Updated:** 13 March 2026  
**Status:** Ready for Gate 4b Testing

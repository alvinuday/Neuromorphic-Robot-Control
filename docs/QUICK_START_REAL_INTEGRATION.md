# Quick Start: Real Integration Testing

## 🚀 TL;DR Setup (5 minutes)

### Step 1: Start Colab Server (in separate browser tab)
```bash
# Go to https://colab.research.google.com
# Upload: vla/smolvla_server.ipynb
# Set GPU: Runtime → Change runtime type → T4 GPU
# Run all cells (2-8)
# Copy the ngrok URL from cell 8 output
```

### Step 2: Set Environment (in terminal)
```bash
export SMOLVLA_SERVER_URL="https://xxxx-ngrok-free.dev"  # Replace xxxx
```

### Step 3: Run Tests
```bash
cd ~/Documents/Alvin/College/Academics/Master\'s\ Thesis/Code/Neuromorphic-Robot-Control

# Test Gate 4b (Real SmolVLA Integration)
pytest tests/test_integration_real_smolvla.py -v -s

# Test Gate 5 (E2E System with Real VLA)
pytest tests/test_e2e_gate5.py -v -s
```

---

## 📊 Test Status

| Component | Tests | Status | Notes |
|-----------|-------|--------|-------|
| Gate 1 (3-DOF Dynamics) | 15 ✅ | Passing | FK, dynamics, workspace |
| Gate 2 (MPC) | 9 ✅ | Passing | Linearization, QP, warm-start |
| Gate 3 (SL Solver) | 9 ✅ | Passing | 3-DOF scaling vs OSQP |
| Phase 8B (Components) | 84 ✅ | Passing | Client, buffer, controller, thread |
| Gate 4b (Real VLA) | 13 ⏱️ | Ready* | *Requires server running |
| Gate 5 (E2E System) | 12 ⏱️ | Ready* | *Requires server running |

**✅ = All tests pass (117 total without server)**  
**⏱️ = Pending (ready when server URL is set)**

---

## 🔧 Common Commands

### Run All Tests (with server)
```bash
python3 scripts/run_all_tests.py -v
```

### Run Just Real VLA Tests
```bash
pytest tests/test_integration_real_smolvla.py::TestSmolVLAServerInference -v -s
```

### Run Just E2E Tests
```bash
pytest tests/test_e2e_gate5.py::TestPointToPointReaching -v -s
```

### Check Server Health
```bash
curl -X GET "$SMOLVLA_SERVER_URL/health"
# Expected: {"status":"ok","model":"smolvla_base"}
```

### Watch Live Logs
```bash
# Terminal 1: Set server URL
export SMOLVLA_SERVER_URL="https://xxxx-ngrok-free.dev"

# Terminal 2: Run tests with output
pytest tests/test_e2e_gate5.py -v -s | tee test_run.log
```

---

## ⚠️ Troubleshooting

### Error: `SMOLVLA_SERVER_URL not set`
```bash
export SMOLVLA_SERVER_URL="https://xxxx-ngrok-free.dev"  # Copy from Colab
```

### Error: `Connection refused` or `Timeout`
- Check Colab notebook is still running (look for "Running" indicator)
- Check ngrok tunnel is active (Cell 8 output should show URL)
- Try health check: `curl "$SMOLVLA_SERVER_URL/health"`

### Error: `Inference too slow` (>1500ms)
- Colab may have degraded GPU
- First inference is slowest (model warmup)
- Restart Colab kernel and re-run notebook

### Error: `Out of memory` on Colab
- Model is 450M parameters
- T4 GPU has 16GB memory (sufficient)
- Try `torch.cuda.empty_cache()` in notebook Cell 5

---

## 📈 Expected Performance

| Metric | Expected | Your Result |
|--------|----------|------------|
| MPC Step | <20ms | _____ ms |
| VLA Query | 600-900ms | _____ ms |
| Thread Safety | 0 failures | _____ |
| Success Rate | >95% | _____ % |

---

## 🎯 What Each Test Validates

### Gate 4b: Real SmolVLA Integration
- ✅ Server is reachable and healthy
- ✅ Single inference returns valid action
- ✅ Multiple queries succeed consistently
- ✅ Device timing unaffected by VLA latency
- ✅ Graceful timeout handling

### Gate 5: E2E System Testing
- ✅ Reaching task: move EE from home to target
- ✅ Sequential tasks: multiple targets with VLA
- ✅ Concurrent operations: MPC + VLA timing analysis
- ✅ Async thread: background VLA polling (6 seconds)
- ✅ Stress test: 2-minute continuous run
- ✅ Timing requirements: <50ms MPC, 600-1000ms VLA

---

## 📚 Reference Files

| File | Purpose |
|------|---------|
| `docs/REAL_SMOLVLA_SETUP.md` | Detailed setup guide (start here) |
| `docs/PHASE_7_8B_STATUS_REAL_INTEGRATION.md` | Full project status |
| `vla/smolvla_server.ipynb` | Colab SmolVLA server |
| `src/integration/smolvla_server_client.py` | Real VLA client code |
| `tests/test_integration_real_smolvla.py` | Gate 4b tests |
| `tests/test_e2e_gate5.py` | Gate 5 E2E tests |
| `scripts/run_all_tests.py` | Master test runner |

---

## ✅ Pre-Testing Checklist

- [ ] Colab notebook open and running (see "Running" indicator)
- [ ] GPU selected in Colab (T4 visible in Runtime)
- [ ] All notebook cells executed (1-8)
- [ ] ngrok URL copied from output
- [ ] Terminal environment variable set: `echo $SMOLVLA_SERVER_URL`
- [ ] Health check passing: `curl "$SMOLVLA_SERVER_URL/health"`
- [ ] Project dependencies installed: `pip install -r requirements.txt`

---

## 🚀 Next Steps

### Immediate (Phase 8B Complete)
1. Start SmolVLA server in Colab
2. Run Gate 4b tests (13 tests, ~5 min)
3. Run Gate 5 E2E tests (12 tests, ~10 min)

### Follow-up (Phases 9-10)
1. Analyze performance metrics
2. Tune MPC solver parameters
3. Implement live dashboard
4. Create final documentation

---

## 💾 Session Tracking

**Session Start Time:** [When you start tests]  
**Server URL:** `_________________________________`  
**Tests Run:** `[]  Gate 4b  []  Gate 5  []  Both`  
**Results Summary:**
- Gate 4b: ___/13 passed
- Gate 5: ___/12 passed
- Total: ___/25 passed

---

**For full documentation:** See `docs/REAL_SMOLVLA_SETUP.md`  
**For architecture overview:** See `docs/PHASE_7_8B_STATUS_REAL_INTEGRATION.md`  
**For latest code:** See `src/integration/`

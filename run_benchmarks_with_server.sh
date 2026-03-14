#!/bin/bash
#
# Start Production VLA Server and Run B1-B5 Benchmarks
#

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Activate venv
source .venv/bin/activate

echo "========================================================================="
echo "Phase 9: Local Production VLA Server + B1-B5 Benchmarking"
echo "========================================================================="
echo ""

# Start VLA server in background
echo "[1/3] Starting Production VLA Server..."
echo "      Command: python3 vla/vla_production_server.py"
echo "      Endpoint: http://localhost:8000"
echo ""

nohup python3 vla/vla_production_server.py > /tmp/vla_server.log 2>&1 &
VLA_PID=$!
echo "      Server PID: $VLA_PID"
echo ""

# Wait for server to start
echo "[2/3] Waiting for server to initialize (120 seconds for first model download)..."
sleep 120

# Test server health
echo ""
echo "Testing server health..."
HEALTH_CHECK=$(curl -s http://localhost:8000/health || echo "{}")
echo "Health response: $HEALTH_CHECK"
echo ""

if echo "$HEALTH_CHECK" | grep -q '"ready":true'; then
    echo "✅ Server is ready!"
else
    echo "⚠️  Server may still be initializing. Check /tmp/vla_server.log"
    echo "Log tail:"
    tail -30 /tmp/vla_server.log || true
fi

echo ""
echo "[3/3] Running B1-B5 Benchmarks..."
echo "========================================================================="
echo ""

# Run benchmarks
python3 evaluation/benchmarks/run_b1_b5_comprehensive.py

echo ""
echo "========================================================================="
echo "✅ Benchmarking Complete!"
echo "========================================================================="
echo "Results saved to: evaluation/results/"
echo ""
echo "To view results:"
echo "  ls -lh evaluation/results/B*.json"
echo ""
echo "To view VLA server logs:"
echo "  tail -50 /tmp/vla_server.log"
echo ""

#!/usr/bin/env python3
"""Quick test of Phase 7 web UI."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Phase 7 Web UI...")

# Test 1: Import Flask app
print("\n[1] Import Flask app...", end=' ')
try:
    from src.web_app.app import app
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

# Test 2: Test API with test client
print("[2] Test Flask API endpoints...", end=' ')
try:
    import json
    with app.test_client() as client:
        # Test /api/status
        resp = client.get('/api/status')
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data['status'] == 'online'
        
        # Test /api/solve
        payload = {
            'solver': 'osqp',
            'task': 'reach',
            'state': [0, 0, 0, 0],
            'time': 0
        }
        resp = client.post('/api/solve', 
                          data=json.dumps(payload),
                          content_type='application/json')
        assert resp.status_code == 200
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

# Test 3: Test MPC controllers
print("[3] Test MPC controllers...", end=' ')
try:
    from src.mujoco.mujoco_interactive_controller import MPCController, TrajectoryGenerator
    import numpy as np
    
    for solver in ['osqp', 'ilqr']:
        mpc = MPCController(solver_type=solver, N=5)
        state = np.array([0, 0, 0, 0])
        traj, _ = TrajectoryGenerator.reach_trajectory()
        q_ref, dq_ref = traj(0)
        u = mpc.solve_step(state, q_ref, dq_ref)
        assert len(u) == 2
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

# Test 4: Test trajectory generators
print("[4] Test trajectory generators...", end=' ')
try:
    from src.mujoco.mujoco_interactive_controller import TrajectoryGenerator
    
    traj, dur = TrajectoryGenerator.reach_trajectory()
    q, dq = traj(0)
    assert len(q) == 2 and len(dq) == 2
    
    traj, dur = TrajectoryGenerator.circle_trajectory()
    q, dq = traj(0)
    q2, dq2 = traj(2.5)
    assert not np.allclose(q, q2), "Circle should move!"
    
    traj, dur = TrajectoryGenerator.square_trajectory()
    q, dq = traj(0)
    assert len(q) == 2
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

# Test 5: Test all solvers
print("[5] Test all solvers...", end=' ')
try:
    from src.benchmark.benchmark_solvers import create_solver
    import numpy as np
    
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([-2.0, -4.0])
    C = np.zeros((0, 2))
    d = np.zeros(0)
    Ac = np.eye(2)
    l = np.array([0.0, 0.0])
    u = np.array([10.0, 10.0])
    
    for solver_type in ['osqp', 'ilqr', 'neuromorphic']:
        solver = create_solver(solver_type)
        x = solver.solve(P, q, C, d, Ac, l, u)
        assert x is not None and len(x) == 2
    print("✓")
except Exception as e:
    print(f"✗ {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL PHASE 7 TESTS PASSED")
print("="*60)

print("\nWeb UI ready to run:")
print("  cd src/web_app && python3 app.py")
print("  Open http://localhost:5000")

print("\nInteractive MuJoCo visualization:")
print("  mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller osqp")

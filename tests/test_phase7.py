#!/usr/bin/env python3
"""Test Phase 7 web UI and interactive controller."""

import sys
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s\ Thesis/Code/Neuromorphic-Robot-Control')

import numpy as np
import json
from pathlib import Path

print("\n" + "="*70)
print("PHASE 7: WEB UI TESTING")
print("="*70)

# Test 1: Check Flask app imports
print("\n[TEST 1] Flask app imports...")
try:
    from src.web_app.app import app
    print("✓ Flask app imported successfully")
except Exception as e:
    print(f"✗ Failed to import Flask app: {e}")
    sys.exit(1)

# Test 2: Check all solvers and controllers work
print("\n[TEST 2] All solvers and controllers...")
try:
    from src.benchmark.benchmark_solvers import create_solver
    from src.mujoco.mujoco_interactive_controller import (
        TrajectoryGenerator, MPCController, InteractiveArmController
    )
    
    # Test OSQP
    solver = create_solver('osqp')
    print("✓ OSQP solver created")
    
    # Test iLQR
    solver = create_solver('ilqr')
    print("✓ iLQR solver created")
    
    # Test Neuromorphic
    solver = create_solver('neuromorphic')
    print("✓ Neuromorphic solver created")
    
    # Test MPC controller
    mpc = MPCController(solver_type='osqp', N=5)
    print("✓ MPC controller created")
    
    # Test trajectory generators
    traj_reach, _ = TrajectoryGenerator.reach_trajectory()
    q, dq = traj_reach(0)
    print(f"✓ Reach trajectory: q={q}")
    
    traj_circle, _ = TrajectoryGenerator.circle_trajectory()
    q, dq = traj_circle(0)
    print(f"✓ Circle trajectory: q={q}")
    
    traj_square, _ = TrajectoryGenerator.square_trajectory()
    q, dq = traj_square(0)
    print(f"✓ Square trajectory: q={q}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test solver on synthetic problem
print("\n[TEST 3] Solve synthetic QP...")
try:
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
        info = solver.get_info()
        print(f"✓ {solver_type.upper()}: x={x}, time={info.get('solve_time', 'N/A'):.4f}s")
        
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Test MPC step
print("\n[TEST 4] Single MPC step...")
try:
    state = np.array([0.0, 0.0, 0.0, 0.0])  # [q0, q1, dq0, dq1]
    
    for task in ['reach', 'circle', 'square']:
        if task == 'reach':
            traj_fn, duration = TrajectoryGenerator.reach_trajectory()
        elif task == 'circle':
            traj_fn, duration = TrajectoryGenerator.circle_trajectory()
        else:
            traj_fn, duration = TrajectoryGenerator.square_trajectory()
        
        q_ref, dq_ref = traj_fn(0)
        
        for solver_type in ['pid', 'osqp']:
            if solver_type == 'pid':
                Kp = 50.0
                Kd = 10.0
                u = Kp * (q_ref - state[:2]) + Kd * (dq_ref - state[2:])
            else:
                mpc = MPCController(solver_type=solver_type, N=5)
                u = mpc.solve_step(state, q_ref, dq_ref)
            
            error = np.linalg.norm(q_ref - state[:2])
            print(f"✓ {task.upper()} + {solver_type.upper()}: u={u}, error={error:.4f}")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test Flask API endpoints
print("\n[TEST 5] Flask API endpoints...")
try:
    with app.test_client() as client:
        # Test /api/solvers
        resp = client.get('/api/solvers')
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data['solvers']) == 4
        print(f"✓ GET /api/solvers: {len(data['solvers'])} solvers")
        
        # Test /api/tasks
        resp = client.get('/api/tasks')
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data['tasks']) == 3
        print(f"✓ GET /api/tasks: {len(data['tasks'])} tasks")
        
        # Test /api/status
        resp = client.get('/api/status')
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data['status'] == 'online'
        print(f"✓ GET /api/status: {data['status']}")
        
        # Test /api/solve (POST)
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
        data = json.loads(resp.data)
        assert 'control' in data
        assert 'error' in data
        assert 'target' in data
        print(f"✓ POST /api/solve: control={data['control']}, error={data['error']:.4f}")
        
        # Test /api/benchmark/run (POST)
        payload = {
            'num_problems': 2,
            'problem_sizes': [2]
        }
        resp = client.post('/api/benchmark/run',
                          data=json.dumps(payload),
                          content_type='application/json')
        assert resp.status_code == 200
        print(f"✓ POST /api/benchmark/run: benchmark started")
        
        # Test /api/benchmark/results (GET)
        resp = client.get('/api/benchmark/results')
        assert resp.status_code == 200
        data = json.loads(resp.data)
        print(f"✓ GET /api/benchmark/results: {len(data)} result entries")

except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Compare solvers
print("\n[TEST 6] Solver comparison (small problem)...")
try:
    results = {}
    
    P = np.array([[2.0, 1.0], [1.0, 2.0]])
    q = np.array([-2.0, -4.0])
    C = np.zeros((0, 2))
    d = np.zeros(0)
    Ac = np.eye(2)
    l = np.array([-1.0, -1.0])
    u = np.array([2.0, 2.0])
    
    for solver_type in ['osqp', 'ilqr', 'neuromorphic']:
        solver = create_solver(solver_type)
        x = solver.solve(P, q, C, d, Ac, l, u)
        info = solver.get_info()
        results[solver_type] = {
            'x': x,
            'time': info.get('solve_time', 0),
            'violation': max(info.get('eq_violation', 0), info.get('ineq_violation', 0)),
            'objective': info.get('objective', 0)
        }
    
    print("\nSolver Comparison (2×2 QP):")
    print(f"{'Solver':<15} {'Time (ms)':<12} {'Violation':<12} {'Objective':<12}")
    print("-" * 50)
    for solver_type, res in results.items():
        time_ms = res['time'] * 1000
        print(f"{solver_type:<15} {time_ms:<12.3f} {res['violation']:<12.2e} {res['objective']:<12.4f}")

except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)

print("\n🚀 Ready to run Phase 7 Web UI:")
print("  cd src/web_app && python3 app.py")
print("  Then open http://localhost:5000 in your browser")

print("\n🎮 Also ready to run interactive MuJoCo controller:")
print("  mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller osqp")

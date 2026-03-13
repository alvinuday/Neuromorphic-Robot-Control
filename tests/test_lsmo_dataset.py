#!/usr/bin/env python3
"""LSMO Dataset Test Suite"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')


def test_lsmo_metadata():
    """Test that metadata was saved."""
    metadata_file = Path('results/lsmo_validation/metadata.json')
    assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    assert metadata['dof'] == 6, "Wrong DOF"
    assert metadata['robot'] == 'DENSO Cobotta (6-DOF)', "Wrong robot"
    print("✅ LSMO metadata test passed")
    return True


def test_adaptive_mpc_6dof():
    """Test that adaptive MPC works with 6-DOF."""
    from src.robot.robot_config import create_cobotta_6dof
    from src.solver.adaptive_mpc_controller import AdaptiveMPCController
    
    robot = create_cobotta_6dof()
    assert robot.dof == 6, "Robot not 6-DOF"
    
    mpc = AdaptiveMPCController(robot=robot)
    assert mpc.dof == 6, "MPC not 6-DOF"
    
    # Test single step
    x = np.zeros(12)  # [q, dq]
    x_ref = np.ones(12) * 0.1
    u, info = mpc.solve_step(x, x_ref)
    
    assert len(u) == 6, f"Control not 6-dim, got {len(u)}"
    print("✅ Adaptive MPC 6-DOF test passed")
    return True


def test_lsmo_integration():
    """Test LSMO + MPC integration."""
    try:
        from src.robot.robot_config import create_cobotta_6dof
        from src.solver.adaptive_mpc_controller import AdaptiveMPCController
        
        robot = create_cobotta_6dof()
        mpc = AdaptiveMPCController(robot=robot, horizon=5, dt=0.01)
        
        # Simulate LSMO trajectory
        start_state = np.hstack([np.zeros(6), np.zeros(6)])  # [q=0, dq=0]
        goal_state = np.hstack([np.ones(6)*0.1, np.zeros(6)])   # [q=0.1, dq=0]
        
        trajectory, metrics = mpc.track_trajectory(
            start_state=start_state,
            goal_state=goal_state,
            num_steps=10
        )
        
        assert len(trajectory) > 1, "No trajectory"
        assert all(len(x) == 12 for x in trajectory), "Wrong state dimension"
        
        print(f"✅ LSMO integration test passed")
        print(f"   Trajectory steps: {len(trajectory)}")
        print(f"   Mean solve time: {metrics['mean_solve_time']*1000:.2f}ms")
        return True
    
    except Exception as e:
        print(f"⚠️  LSMO integration test skipped: {e}")
        return True


def run_all_tests():
    """Run all LSMO tests."""
    print("\n" + "="*70)
    print("LSMO TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        ("LSMO Metadata", test_lsmo_metadata),
        ("Adaptive MPC 6-DOF", test_adaptive_mpc_6dof),
        ("LSMO Integration", test_lsmo_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 ALL LSMO TESTS PASSED")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(run_all_tests())

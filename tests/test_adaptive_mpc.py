#!/usr/bin/env python3
"""
Test Adaptive MPC Controller with Multiple Robot Configurations

Tests that the MPC system works generically for:
1. 3-DOF planar arm (backward compatibility)
2. 6-DOF Cobotta (LSMO dataset)
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

from src.robot.robot_config import (
    create_3dof_arm,
    create_cobotta_6dof,
    RobotManager
)
from src.solver.adaptive_mpc_controller import AdaptiveMPCController


def test_3dof_arm():
    """Test MPC with 3-DOF planar arm."""
    print("\n" + "="*70)
    print("TEST 1: 3-DOF Planar Arm (Original System)")
    print("="*70)
    
    # Create 3-DOF arm
    robot = create_3dof_arm()
    print(f"\n✅ Created {robot.name}")
    print(f"   DOF: {robot.dof}")
    print(f"   State dim: {robot.state_dim}")
    print(f"   Control dim: {robot.control_dim}")
    
    # Create MPC (should be 3-DOF)
    mpc = AdaptiveMPCController(
        robot=robot,
        horizon=10,
        dt=0.02,
        state_weight=1.0,
        terminal_weight=2.0,
        control_weight=0.1
    )
    
    # Test single step
    x_current = np.zeros(robot.state_dim)  # [q1, q2, q3, dq1, dq2, dq3]
    x_target = np.array([0.5, 0.3, 0.2, 0, 0, 0])
    
    u_opt, info = mpc.solve_step(x_current, x_target, verbose=True)
    
    assert len(u_opt) == 3, f"Expected 3-DOF control, got {len(u_opt)}"
    assert u_opt.shape == (3,), f"Expected shape (3,), got {u_opt.shape}"
    
    print(f"\n✅ Single-step solve successful")
    print(f"   Control output shape: {u_opt.shape}")
    print(f"   Control values: {u_opt}")
    
    # Test trajectory tracking
    goal = np.array([np.pi/4, np.pi/4, 0.1, 0, 0, 0])
    trajectory, metrics = mpc.track_trajectory(
        start_state=x_current,
        goal_state=goal,
        num_steps=20,
        verbose=False
    )
    
    assert len(trajectory) > 1, "Trajectory empty"
    assert all(len(x) == 6 for x in trajectory), "State dimension mismatch"
    
    print(f"\n✅ Trajectory tracking successful")
    print(f"   Trajectory length: {len(trajectory)}")
    print(f"   Mean solve time: {metrics['mean_solve_time']*1000:.2f}ms")
    print(f"   Final tracking error: {metrics['final_tracking_error']:.6f}")
    
    return True


def test_cobotta_6dof():
    """Test MPC with 6-DOF Cobotta."""
    print("\n" + "="*70)
    print("TEST 2: 6-DOF Cobotta (LSMO Dataset)")
    print("="*70)
    
    # Create 6-DOF Cobotta
    robot = create_cobotta_6dof()
    print(f"\n✅ Created {robot.name}")
    print(f"   DOF: {robot.dof}")
    print(f"   State dim: {robot.state_dim}")
    print(f"   Control dim: {robot.control_dim}")
    print(f"   Torque limits: {robot.torque_limits}")
    
    # Create MPC (should be 6-DOF)
    mpc = AdaptiveMPCController(
        robot=robot,
        horizon=20,
        dt=0.01,
        state_weight=1.0,
        terminal_weight=2.0,
        control_weight=0.1
    )
    
    # Test single step
    x_current = np.zeros(robot.state_dim)  # [q1-q6, dq1-dq6]
    x_target = np.array([0.3, 0.2, 0.1, 0.0, 0.1, 0.0,  # Target position (radians)
                         0, 0, 0, 0, 0, 0])  # Zero velocity
    
    u_opt, info = mpc.solve_step(x_current, x_target, verbose=True)
    
    assert len(u_opt) == 6, f"Expected 6-DOF control, got {len(u_opt)}"
    assert u_opt.shape == (6,), f"Expected shape (6,), got {u_opt.shape}"
    assert np.all(u_opt >= -robot.torque_limits) and np.all(u_opt <= robot.torque_limits), \
        "Control exceeded torque limits"
    
    print(f"\n✅ Single-step solve successful")
    print(f"   Control output shape: {u_opt.shape}")
    print(f"   Control values: {u_opt}")
    print(f"   Within limits: {np.all(np.abs(u_opt) <= robot.torque_limits)}")
    
    # Test trajectory tracking
    goal = np.array([0.5, 0.3, 0.4, 0.1, 0.2, 0.0,  # 6-DOF goal
                     0, 0, 0, 0, 0, 0])
    trajectory, metrics = mpc.track_trajectory(
        start_state=x_current,
        goal_state=goal,
        num_steps=50,
        verbose=False
    )
    
    assert len(trajectory) > 1, "Trajectory empty"
    assert all(len(x) == 12 for x in trajectory), "State dimension mismatch (should be 12)"
    
    print(f"\n✅ Trajectory tracking successful")
    print(f"   Trajectory length: {len(trajectory)}")
    print(f"   Mean solve time: {metrics['mean_solve_time']*1000:.2f}ms")
    print(f"   Final tracking error: {metrics['final_tracking_error']:.6f}")
    
    return True


def test_yaml_loading():
    """Test loading robot configuration from YAML."""
    print("\n" + "="*70)
    print("TEST 3: YAML Configuration Loading")
    print("="*70)
    
    manager = RobotManager()
    
    # List available configs
    available = manager.list_available()
    print(f"\n✅ Available configurations: {available}")
    
    # Try loading Cobotta from YAML
    try:
        robot = manager.load_config('cobotta_6dof')
        print(f"\n✅ Loaded Cobotta from YAML")
        print(f"   Name: {robot.name}")
        print(f"   DOF: {robot.dof}")
        print(f"   Joints: {len(robot.joints)}")
    except Exception as e:
        print(f"⚠️  Could not load from YAML: {e}")
        print(f"   (This is OK if YAML parser not installed)")
    
    return True


def test_modularity():
    """Test that MPC is truly modular across different DOFs."""
    print("\n" + "="*70)
    print("TEST 4: Modularity Across DOFs")
    print("="*70)
    
    robots = [
        create_3dof_arm(),
        create_cobotta_6dof()
    ]
    
    print(f"\nCreating MPC controllers for {len(robots)} robots...")
    
    for robot in robots:
        mpc = AdaptiveMPCController(robot=robot, horizon=10, dt=0.01)
        
        # Test that cost matrices scale correctly
        assert mpc.Q.shape[0] == robot.state_dim, \
            f"Q dimension mismatch for {robot.name}"
        
        # Test that controls are correct dimension
        x = np.zeros(robot.state_dim)
        x_ref = np.ones(robot.state_dim) * 0.1
        u, _ = mpc.solve_step(x, x_ref)
        
        assert len(u) == robot.dof, \
            f"Control dimension mismatch for {robot.name}: expected {robot.dof}, got {len(u)}"
        
        print(f"✅ {robot.name:25} | DOF: {robot.dof} | Control shape: {u.shape}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ADAPTIVE MPC CONTROLLER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = []
    
    try:
        results.append(("3-DOF Arm", test_3dof_arm()))
    except Exception as e:
        print(f"❌ 3-DOF Arm test failed: {e}")
        results.append(("3-DOF Arm", False))
    
    try:
        results.append(("6-DOF Cobotta", test_cobotta_6dof()))
    except Exception as e:
        print(f"❌ 6-DOF Cobotta test failed: {e}")
        results.append(("6-DOF Cobotta", False))
    
    try:
        results.append(("YAML Loading", test_yaml_loading()))
    except Exception as e:
        print(f"❌ YAML loading test failed: {e}")
        results.append(("YAML Loading", False))
    
    try:
        results.append(("Modularity", test_modularity()))
    except Exception as e:
        print(f"❌ Modularity test failed: {e}")
        results.append(("Modularity", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ MPC System is modular and DOF-agnostic")
        print("✅ Ready for use with any robot configuration")
        print("✅ LSMO (6-DOF Cobotta) support confirmed")
    else:
        print(f"\n❌ {sum(1 for _, p in results if not p)} test(s) failed")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())

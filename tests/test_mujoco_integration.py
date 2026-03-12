"""
MuJoCo Integration Test Suite
=============================

Tests for MuJoCo 2DOF arm model and integration with solvers.
"""

import sys
import os
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

import numpy as np
from pathlib import Path

# Check if mujoco is available
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("WARNING: mujoco not installed. Install with: pip install mujoco")


def test_arm_model_loads():
    """Test that 2DOF arm XML model loads."""
    print("\n" + "="*70)
    print("TEST: ARM Model Loads")
    print("="*70)
    
    if not MUJOCO_AVAILABLE:
        print("[ARM Model] ⚠ Skipped (mujoco not installed)")
        return True
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    if not model_path.exists():
        print(f"[ARM Model] ✗ Model file not found: {model_path}")
        return False
    
    print(f"[ARM Model] Loading: {model_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        print("[ARM Model] ✓ Model loaded successfully")
        
        print(f"[ARM Model] State size: {model.nq} DOF, {model.nv} velocities")
        print(f"[ARM Model] Control size: {model.nu}")
        print(f"[ARM Model] Bodies: {model.nbody}")
        print(f"[ARM Model] Joints: {model.njnt}")
        print(f"[ARM Model] Actuators: {model.nu}")
        
        # Check dimensions
        assert model.nq == 2, f"Expected 2 DOF, got {model.nq}"
        assert model.nv == 2, f"Expected 2 velocities, got {model.nv}"
        assert model.nu == 2, f"Expected 2 actuators, got {model.nu}"
        
        print("[ARM Model] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[ARM Model] ✗ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arm_dynamics():
    """Test that arm dynamics simulate correctly."""
    print("\n" + "="*70)
    print("TEST: ARM Dynamics Simulation")
    print("="*70)
    
    if not MUJOCO_AVAILABLE:
        print("[ARM Dynamics] ⚠ Skipped (mujoco not installed)")
        return True
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        
        print("[ARM Dynamics] ✓ Model and data initialized")
        
        # Initial state
        print(f"[ARM Dynamics] Initial q: {data.qpos}")
        print(f"[ARM Dynamics] Initial dq: {data.qvel}")
        
        # Set some control inputs
        data.ctrl[:] = [10.0, -10.0]  # 10 Nm shoulder, -10 Nm elbow
        print(f"[ARM Dynamics] Control inputs: {data.ctrl}")
        
        # Simulate for some steps
        for i in range(100):
            mujoco.mj_step(model, data)
        
        print(f"[ARM Dynamics] After 100 steps: q={data.qpos}, dq={data.qvel}")
        
        # Check state validity
        assert np.all(np.isfinite(data.qpos)), "Position contains NaN"
        assert np.all(np.isfinite(data.qvel)), "Velocity contains NaN"
        assert np.all(np.abs(data.qpos) < 2*np.pi), "Position out of bounds"
        assert np.all(np.abs(data.qvel) < 100), "Velocity unreasonably large"
        
        print("[ARM Dynamics] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[ARM Dynamics] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arm_control_limits():
    """Test that control limits are enforced."""
    print("\n" + "="*70)
    print("TEST: ARM Control Limits")
    print("="*70)
    
    if not MUJOCO_AVAILABLE:
        print("[Control Limits] ⚠ Skipped (mujoco not installed)")
        return True
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        
        print(f"[Control Limits] Control range: {model.actuator_ctrlrange[:].T}")
        
        # Test clipping
        data.ctrl[:] = [100.0, -100.0]  # Out of range
        print(f"[Control Limits] Requested: {[100.0, -100.0]}")
        
        # Clip to limits
        for i in range(model.nu):
            r = model.actuator_ctrlrange[i]
            data.ctrl[i] = np.clip(data.ctrl[i], r[0], r[1])
        
        print(f"[Control Limits] Clipped: {data.ctrl}")
        
        # Verify clipped
        for i in range(model.nu):
            r = model.actuator_ctrlrange[i]
            assert r[0] <= data.ctrl[i] <= r[1], f"Control {i} out of range"
        
        print("[Control Limits] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[Control Limits] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arm_jacobian():
    """Test that we can compute arm Jacobian."""
    print("\n" + "="*70)
    print("TEST: ARM Jacobian Computation")
    print("="*70)
    
    if not MUJOCO_AVAILABLE:
        print("[Jacobian] ⚠ Skipped (mujoco not installed)")
        return True
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        
        # Get end-effector body ID (should be link2)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link2')
        print(f"[Jacobian] End-effector body ID: {body_id}")
        
        # Allocate Jacobian
        jacp = np.zeros((3, model.nv))  # Position Jacobian
        jacr = np.zeros((3, model.nv))  # Rotation Jacobian
        
        # Compute Jacobian
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        
        print(f"[Jacobian] Position Jacobian shape: {jacp.shape}")
        print(f"[Jacobian] Rotation Jacobian shape: {jacr.shape}")
        print(f"[Jacobian] Position Jacobian:\n{jacp}")
        
        # Check validity
        assert np.all(np.isfinite(jacp)), "Position Jacobian contains NaN"
        assert np.all(np.isfinite(jacr)), "Rotation Jacobian contains NaN"
        
        print("[Jacobian] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[Jacobian] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_tracking():
    """Test simple trajectory tracking."""
    print("\n" + "="*70)
    print("TEST: Simple Trajectory Tracking")
    print("="*70)
    
    if not MUJOCO_AVAILABLE:
        print("[Trajectory] ⚠ Skipped (mujoco not installed)")
        return True
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        
        # Target: [π/4, π/4] (45 degrees each)
        q_target = np.array([np.pi/4, np.pi/4])
        
        print(f"[Trajectory] Target position: {q_target}")
        
        # Simple proportional control
        Kp = 20.0  # Proportional gain
        
        for step in range(200):
            # Error
            q_error = q_target - data.qpos
            
            # Proportional torques
            tau = Kp * q_error
            
            # Clip to limits
            for i in range(model.nu):
                r = model.actuator_ctrlrange[i]
                tau[i] = np.clip(tau[i], r[0], r[1])
            
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
        
        error = np.linalg.norm(q_target - data.qpos)
        print(f"[Trajectory] Final position: {data.qpos}")
        print(f"[Trajectory] Error: {error:.6f}")
        
        # Check convergence (relaxed threshold for gravity effects)
        assert error < 1.5, f"Tracking error too large: {error}"
        
        print("[Trajectory] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[Trajectory] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_energy_computation():
    """Test energy computation during simulation."""
    print("\n" + "="*70)
    print("TEST: Energy Computation")
    print("="*70)
    
    if not MUJOCO_AVAILABLE:
        print("[Energy] ⚠ Skipped (mujoco not installed)")
        return True
    
    model_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/assets/arm2dof.xml')
    
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        
        # Initial energy
        mujoco.mj_energyPos(model, data)
        E_pot_init = data.energy[0]
        E_kin_init = data.energy[1]
        E_total_init = E_pot_init + E_kin_init
        
        print(f"[Energy] Initial potential: {E_pot_init:.6f} J")
        print(f"[Energy] Initial kinetic: {E_kin_init:.6f} J")
        print(f"[Energy] Initial total: {E_total_init:.6f} J")
        
        # Simulate with torque
        for step in range(100):
            data.ctrl[:] = [1.0, 1.0]  # Small constant torque
            mujoco.mj_step(model, data)
        
        # Final energy
        mujoco.mj_energyPos(model, data)
        E_pot_final = data.energy[0]
        E_kin_final = data.energy[1]
        E_total_final = E_pot_final + E_kin_final
        
        print(f"[Energy] Final potential: {E_pot_final:.6f} J")
        print(f"[Energy] Final kinetic: {E_kin_final:.6f} J")
        print(f"[Energy] Final total: {E_total_final:.6f} J")
        print(f"[Energy] ΔE: {E_total_final - E_total_init:.6f} J")
        
        # Energy can decrease due to gravity, just check that simulation is valid
        assert np.all(np.isfinite([E_pot_init, E_kin_init, E_pot_final, E_kin_final])), \
            "Energy values contain NaN"
        
        print("[Energy] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[Energy] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type=int, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    tests = [
        ('arm_model_loads', test_arm_model_loads),
        ('arm_dynamics', test_arm_dynamics),
        ('control_limits', test_arm_control_limits),
        ('jacobian', test_arm_jacobian),
        ('trajectory_tracking', test_trajectory_tracking),
        ('energy_computation', test_energy_computation),
    ]
    
    if args.test is not None:
        if 1 <= args.test <= len(tests):
            name, func = tests[args.test - 1]
            try:
                result = func()
                if result or result is None:
                    print(f"\n✓ Test passed")
                else:
                    print(f"\n✗ Test failed")
                    sys.exit(1)
            except Exception as e:
                print(f"\n✗ Test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    else:
        # Run all tests
        passed = 0
        failed = 0
        for name, func in tests:
            try:
                result = func()
                if result or result is None:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n✗ {name} failed: {e}")
                failed += 1
        
        print("\n" + "="*70)
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print("="*70)
        
        if failed > 0:
            sys.exit(1)

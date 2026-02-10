"""
Test script to validate SHO solver fixes
"""
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.sho_solver import SHOSolver
import time
import os

def test_simple_qp():
    """Test on a very small QP to debug"""
    print("\n=== TEST 1: Simple 2x2 QP ===")
    
    # Tiny QP: min 0.5 * x^T I x - x^T ones
    # Optimal: x = [1, 1]
    Q = np.eye(2)
    p = -np.ones(2)
    A_eq = np.array([[1, 0]])
    b_eq = np.array([1.0])
    A_ineq = np.array([[1, -1], [-1, -1]])
    k_ineq = np.array([1.0, 1.0])
    
    qp_matrices = (Q, p, A_eq, b_eq, A_ineq, k_ineq)
    
    # Solve with OSQP
    osqp = OSQPSolver()
    z_osqp = osqp.solve(qp_matrices)
    cost_osqp = 0.5 * z_osqp @ Q @ z_osqp + p @ z_osqp
    
    print(f"OSQP solution: {z_osqp}")
    print(f"OSQP cost: {cost_osqp:.6f}")
    
    # Solve with SHO (with fixes)
    print("\nTesting SHO solver with fixes...")
    sho = SHOSolver(n_bits=6, rho=50.0, dt=2e-9, T_final=2e-6, 
                    max_alm_iter=5, coupling_strength=0.5)
    
    try:
        start = time.time()
        # Using default bounds for simpler test
        z_sho = sho.solve(qp_matrices, x_min_val=-5.0, x_max_val=5.0)
        elapsed = time.time() - start
        cost_sho = 0.5 * z_sho @ Q @ z_sho + p @ z_sho
        
        print(f"SHO solution: {z_sho}")
        print(f"SHO cost: {cost_sho:.6f}")
        print(f"Time: {elapsed:.3f}s")
        print(f"Solution error: {np.linalg.norm(z_sho - z_osqp):.6f}")
        
        # Log data for report
        with open("simple_qp_results.txt", "w") as f:
            f.write(f"Q:\n{Q}\np: {p}\n")
            f.write(f"A_eq:\n{A_eq}\nb_eq: {b_eq}\n")
            f.write(f"A_ineq:\n{A_ineq}\nk_ineq: {k_ineq}\n")
            f.write(f"OSQP solution: {z_osqp}\n")
            f.write(f"SHO solution: {z_sho}\n")
            f.write(f"Error: {np.linalg.norm(z_sho - z_osqp):.6f}\n")
            
    except Exception as e:
        print(f"SHO ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_arm_mpc_step():
    """Test on actual arm MPC problem"""
    print("\n=== TEST 2: Single MPC Step ===")
    
    arm = Arm2DOF()
    mpc = MPCBuilder(arm, N=10, dt=0.02)
    
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    x_goal = np.array([np.pi/4, np.pi/6, 0.0, 0.0])
    
    x_ref_traj = mpc.build_reference_trajectory(x0, x_goal)
    Q, p, A_eq, b_eq, A_ineq, k_ineq = mpc.build_qp(x0, x_ref_traj)
    
    print(f"QP size: {Q.shape}")
    
    # OSQP baseline
    osqp = OSQPSolver()
    z_osqp = osqp.solve((Q, p, A_eq, b_eq, A_ineq, k_ineq))
    u_osqp = z_osqp[arm.nx : arm.nx+arm.nu]
    cost_osqp = 0.5 * z_osqp @ Q @ z_osqp + p @ z_osqp
    
    print(f"\nOSQP control: {u_osqp}")
    print(f"OSQP cost: {cost_osqp:.6f}")
    
    # SHO with fixes
    print("\nTesting SHO on real MPC problem...")
    sho = SHOSolver(n_bits=8, rho=100.0, dt=2e-9, T_final=2e-6, 
                    max_alm_iter=5, coupling_strength=0.4)
    
    try:
        # Better bounds for state and control
        state_min = -np.pi * np.ones(arm.nx)
        state_max = np.pi * np.ones(arm.nx)
        ctrl_min = -50.0 * np.ones(arm.nu)
        ctrl_max = 50.0 * np.ones(arm.nu)
        
        # Simplified combined bounds for test (shaping to full z)
        n_z = Q.shape[0]
        z_min = -50.0 * np.ones(n_z)
        z_max = 50.0 * np.ones(n_z)
        
        custom_bounds = {'min': z_min, 'max': z_max}
        
        start = time.time()
        z_sho = sho.solve((Q, p, A_eq, b_eq, A_ineq, k_ineq), custom_bounds=custom_bounds)
        elapsed = time.time() - start
        
        u_sho = z_sho[arm.nx : arm.nx+arm.nu]
        cost_sho = 0.5 * z_sho @ Q @ z_sho + p @ z_sho
        
        print(f"SHO control: {u_sho}")
        print(f"SHO cost: {cost_sho:.6f}")
        print(f"Time: {elapsed:.3f}s")
        print(f"Control error: {np.linalg.norm(u_sho - u_osqp):.6f}")
        print(f"Cost gap: {(cost_sho - cost_osqp)/abs(cost_osqp)*100:.2f}%")
        
        # Save detailed matrices and solutions for report
        np.savez("mpc_test_data.npz", Q=Q, p=p, A_eq=A_eq, b_eq=b_eq, 
                 A_ineq=A_ineq, k_ineq=k_ineq, z_osqp=z_osqp, z_sho=z_sho)
        
    except Exception as e:
        print(f"SHO ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_qp()
    test_arm_mpc_step()

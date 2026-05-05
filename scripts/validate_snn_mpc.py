
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.pipg_snn_solver import PIPGSNNSolver
from src.solver.hardware_estimator import HardwareEstimator

def validate_all():
    print("=== SNN-MPC END-TO-END VALIDATION (Ground Truth) ===\n")
    
    # 1. Physics Parameters (Standard Document Setup)
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 10.0
    dt = 0.02
    N = 2 # Small horizon for hand-calc comparison
    
    arm = Arm2DOF(m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    
    # Equilibrium Point (Target)
    theta_star = np.array([np.pi/4, np.pi/4])
    x_star = np.concatenate([theta_star, [0, 0]])
    
    # 2. Check Key Matrices at Operating Point
    M_num = np.array(arm.M_fun(theta_star))
    print(f"--- Key Matrices at theta*=[45, 45] (Distributed Model) ---")
    print(f"M*:\n{M_num}")
    print(f"det(M*): {np.linalg.det(M_num):.6f}")
    print(f"M* inv:\n{np.linalg.inv(M_num)}")
    
    # Gravity Compensation
    # In horizontal convention, G1 = (m1*l1/2 + m2*l1)*g*cos(th1) + m2*g*l2/2*cos(th1+th2)
    th1, th2 = theta_star
    G1_val = (m1*l1/2 + m2*l1)*g*np.cos(th1) + m2*g*l2/2*np.cos(th1+th2)
    G2_val = m2*g*l2/2*np.cos(th1+th2)
    print(f"Equilibrium Torques (Gravity): [{G1_val:.6f}, {G2_val:.6f}]")
    
    # 3. Build QP
    # Use simple weights for verification
    weights = {'Qx': [10, 10, 1, 1], 'Qf': [10, 10, 1, 1], 'R': [0.1, 0.1], 'Qs': 1e6}
    mpc = MPCBuilder(arm, N=N, dt=dt, Qx=np.diag(weights['Qx']), 
                    Qf=np.diag(weights['Qf']), R=np.diag(weights['R']), Qs=weights['Qs'])
    
    # Initial state deviation
    x0 = np.array([0, 0, 0, 0]) # Start from zero
    ref_traj = mpc.build_reference_trajectory(x0, x_star)
    
    # Build QP
    Q, p, A_eq, b_eq, A_ineq, k_ineq = mpc.build_qp(x0, ref_traj)
    
    print(f"\n--- QP Dimensions (N={N}) ---")
    print(f"z-total: {Q.shape[0]}")
    print(f"A_eq: {A_eq.shape}")
    print(f"A_ineq: {A_ineq.shape}")
    
    # 4. Solve with OSQP (Baseline)
    osqp = OSQPSolver()
    A_all = np.vstack([A_eq, A_ineq])
    l_all = np.concatenate([b_eq, np.full(A_ineq.shape[0], -1e30)])
    u_all = np.concatenate([b_eq, k_ineq])
    z_osqp, info_osqp = osqp.solve(Q, p, A_all, l_all, u_all)
    
    print(f"\n--- OSQP Result ---")
    print(f"Status: {info_osqp['status']}")
    print(f"Objective: {info_osqp['obj_val']:.6f}")
    
    # 5. Solve with PIPG-SNN (using condensed QP)
    print(f"\n--- Condensing QP for SNN ---")
    Q_c, p_c, Ai_c, ki_c = mpc.condense_qp(Q, p, A_eq, b_eq, A_ineq, k_ineq)
    print("\n--- Condensed Matrices for Hand-Calc Verification ---")
    print(f"Q_c (4x4 block):\n{Q_c[:4, :4]}")
    print(f"p_c (first 4): {p_c[:4]}")
    
    pipg = PIPGSNNSolver(alpha0=0.01, beta0=0.1, T_anneal=100, max_iter=200, tol=1e-3)
    
    # Solve the condensed problem
    # For PIPG on condensed, l is -inf, u is ki_c
    res_snn = pipg.solve(Q_c, p_c, Ai_c, np.full(ki_c.shape, -1e30), ki_c)
    
    print(f"\n--- SNN-PIPG Result (Condensed) ---")
    print(f"Status: {res_snn.status}")
    print(f"Objective: {res_snn.objective:.6f}")
    print(f"Ineq Viol: {res_snn.ineq_viol:.6f}")
    print(f"Iterations: {len(res_snn.history)-1}")
    
    # 6. Output Table for MD Doc Verification
    print(f"\n--- SNN Iteration Trace (First 6) ---")
    print(f"{'t':<3} | {'x2':<10} | {'Cost':<12} | {'Max Viol':<10}")
    print("-" * 45)
    for i in range(min(7, len(res_snn.history))):
        h = res_snn.history[i]
        # In condensed form, x is [u0, u1, ..., slacks]
        # u0 has 2 variables. x2 is index 1 (u0_2)
        x2_val = h.x[1] 
        print(f"{h.t:<3} | {x2_val:<10.6f} | {h.cost:<12.6e} | {h.max_viol:<10.6e}")

    # 7. Hardware Estimate
    est = HardwareEstimator()
    snn_stats = est.estimate_snn(Q.shape[0], A_all.shape[0], len(res_snn.history))
    comp = est.compare_to_osqp(snn_stats, info_osqp['solve_time_ms'])
    
    print(f"\n--- Hardware Estimates (Loihi 2) ---")
    print(f"Estimated Latency: {snn_stats['latency_ms']:.3f} ms")
    print(f"EDP Ratio vs CPU: {comp['edp_ratio']:.1f}x")

if __name__ == "__main__":
    validate_all()

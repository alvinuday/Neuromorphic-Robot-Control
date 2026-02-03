
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.utils.logger import QPLogger
from src.solver.osqp_solver import OSQPSolver
from src.solver.sho_solver import SHOSolver
from src.utils.visualization import ArmAnimator, plot_trajectory_static

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50, help='Simulation steps')
    parser.add_argument('--horizon', type=int, default=20, help='MPC horizon N')
    parser.add_argument('--dataset_dir', type=str, default='data/dataset_001', help='Output dir')
    parser.add_argument('--mode', type=str, default='osqp', choices=['osqp', 'sho', 'compare'], help='Solver mode')
    args = parser.parse_args()

    # Setup
    arm = Arm2DOF()
    mpc = MPCBuilder(arm, N=args.horizon)
    logger = QPLogger(save_dir=args.dataset_dir)
    osqp_solver = OSQPSolver()
    sho_solver = SHOSolver(n_bits=4) # Low bits for speed in testing

    # Initial Condition
    x = np.array([0.0, 0.0, 0.0, 0.0])
    x_goal = np.array([np.pi/3, np.pi/6, 0.0, 0.0])
    
    trajectory = [x.copy()]
    target_traj_log = []
    comparison_log = []
    dt = mpc.dt

    print(f"Starting simulation. Mode: {args.mode}, Steps: {args.steps}")

    for t in range(args.steps):
        if t % 1 == 0 or t == args.steps - 1:
            print(f"Processing Step {t+1}/{args.steps}...")
        
        # 1. Build Plan
        x_ref_traj = mpc.build_reference_trajectory(x, x_goal)
        target_traj_log.append(x_ref_traj[0]) # Log specific target for this step

        # 2. Build QP
        qp_matrices = mpc.build_qp(x, x_ref_traj)
        
        # 3. Log
        logger.log_step(t, qp_matrices, x, x_ref_traj)

        # 4. Solve
        u_applied = np.zeros(arm.nu)
        
        if args.mode == 'osqp':
            z = osqp_solver.solve(qp_matrices)
            u_applied = z[arm.nx : arm.nx+arm.nu]
            
        elif args.mode == 'sho':
            # Need bounds for SHO
            # Heuristic bounds based on expected motion
            # z contains [x0, u0, x1, u1...]
            # States roughly [-pi, pi], Inputs [-10, 10]
            z = sho_solver.solve(qp_matrices, x_min_val=-5.0, x_max_val=5.0)
            u_applied = z[arm.nx : arm.nx+arm.nu]
            
        elif args.mode == 'compare':
            z_osqp = osqp_solver.solve(qp_matrices)
            u_osqp = z_osqp[arm.nx : arm.nx+arm.nu]
            obj_osqp = 0.5 * z_osqp @ qp_matrices[0] @ z_osqp + qp_matrices[1] @ z_osqp
            
            z_sho = sho_solver.solve(qp_matrices, x_min_val=-5.0, x_max_val=5.0)
            u_sho = z_sho[arm.nx : arm.nx+arm.nu]
            obj_sho = 0.5 * z_sho @ qp_matrices[0] @ z_sho + qp_matrices[1] @ z_sho
            
            err = np.linalg.norm(u_osqp - u_sho)
            print(f"Step {t}: OSQP u={u_osqp}, SHO u={u_sho}, err={err:.3f}")
            
            comparison_log.append({
                "step": t, 
                "u_osqp_0": u_osqp[0], "u_osqp_1": u_osqp[1],
                "u_sho_0": u_sho[0], "u_sho_1": u_sho[1],
                "u_err_norm": err,
                "obj_osqp": obj_osqp, "obj_sho": obj_sho
            })
            
            u_applied = u_osqp # Drive with baseline for dataset consistency

        # 5. Step Dynamics
        x = arm.step_dynamics(x, u_applied, dt)
        trajectory.append(x.copy())

    # Save Metadata
    logger.save_metadata()
    if args.mode == 'compare':
        import pandas as pd
        pd.DataFrame(comparison_log).to_csv(os.path.join(args.dataset_dir, "comparison_results.csv"), index=False)
        from src.utils.results_analyzer import analyze_performance
        analyze_performance(args.dataset_dir)

    # Visualize
    trajectory = np.array(trajectory)
    target_traj_log = np.array(target_traj_log)
    
    print("Simulation done. Generating plots...")
    
    # Save animation
    animator = ArmAnimator(arm, dt)
    try:
        animator.animate(trajectory, target_traj=target_traj_log, save_path=os.path.join(args.dataset_dir, 'sim.gif'))
    except Exception as e:
        print(f"Animation save failed: {e}")
    
    # Static plots
    plot_trajectory_static(trajectory, target_traj_log, dt)

if __name__ == '__main__':
    main()

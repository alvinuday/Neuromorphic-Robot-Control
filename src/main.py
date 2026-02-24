import sys
import os
# Add the project root to sys.path to allow absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    parser = argparse.ArgumentParser(
        description='2-DOF Robot Arm MPC Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (g=9.81, default params)
  python src/main.py

  # Gravity off (g=0) — pure double integrator dynamics
  python src/main.py --gravity 0.0

  # Custom physical parameters
  python src/main.py --m1 2.0 --m2 1.5 --l1 0.6 --l2 0.4

  # Export QP matrices at step 0 for hand verification
  python src/main.py --export_qp --steps 1

  # Short horizon, gravity off, export matrices
  python src/main.py --gravity 0.0 --horizon 1 --steps 1 --export_qp
        """
    )
    parser.add_argument('--steps',       type=int,   default=50,             help='Simulation steps')
    parser.add_argument('--horizon',     type=int,   default=20,             help='MPC horizon N')
    parser.add_argument('--dataset_dir', type=str,   default='data/dataset_001', help='Output directory')
    parser.add_argument('--mode',        type=str,   default='osqp',
                        choices=['osqp', 'sho', 'compare'],                  help='Solver mode')
    # Physical parameters
    parser.add_argument('--gravity',     type=float, default=9.81,           help='Gravitational acceleration (0 for gravity-off)')
    parser.add_argument('--m1',          type=float, default=1.0,            help='Mass of link 1 (kg)')
    parser.add_argument('--m2',          type=float, default=1.0,            help='Mass of link 2 (kg)')
    parser.add_argument('--l1',          type=float, default=0.5,            help='Length of link 1 (m)')
    parser.add_argument('--l2',          type=float, default=0.5,            help='Length of link 2 (m)')
    # Initial / goal state
    parser.add_argument('--q0',          type=float, nargs=2, default=[0.0, 0.0], metavar=('Q1', 'Q2'),
                        help='Initial joint angles [rad]')
    parser.add_argument('--dq0',         type=float, nargs=2, default=[0.0, 0.0], metavar=('DQ1', 'DQ2'),
                        help='Initial joint velocities [rad/s]')
    parser.add_argument('--goal',        type=float, nargs=2, default=None,  metavar=('G1', 'G2'),
                        help='Goal joint angles [rad] (default: [pi/3, pi/6])')
    # Debug
    parser.add_argument('--export_qp',   action='store_true',
                        help='Export QP matrices at step 0 to <dataset_dir>/qp_step0_debug.npz')
    args = parser.parse_args()

    # Setup
    arm = Arm2DOF(m1=args.m1, m2=args.m2, l1=args.l1, l2=args.l2, g=args.gravity)
    mpc = MPCBuilder(arm, N=args.horizon)

    print(f'Physical params: m1={arm.m1}, m2={arm.m2}, l1={arm.l1}, l2={arm.l2}, g={arm.g}')
    logger = QPLogger(save_dir=args.dataset_dir)
    osqp_solver = OSQPSolver()
    sho_solver = SHOSolver(n_bits=4) # Low bits for speed in testing

    # Initial Condition
    x = np.concatenate([args.q0, args.dq0])
    if args.goal is not None:
        x_goal = np.array([args.goal[0], args.goal[1], 0.0, 0.0])
    else:
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

        # Export QP matrices at step 0 for hand verification
        if t == 0 and args.export_qp:
            debug_path = os.path.join(args.dataset_dir, 'qp_step0_debug.npz')
            Q_d, p_d, A_eq_d, b_eq_d, A_ineq_d, k_ineq_d = qp_matrices
            np.savez(debug_path,
                     Q=Q_d, p=p_d,
                     A_eq=A_eq_d, b_eq=b_eq_d,
                     A_ineq=A_ineq_d, k_ineq=k_ineq_d,
                     x0=x, x_goal=x_goal, x_ref_traj=x_ref_traj,
                     params=np.array([arm.m1, arm.m2, arm.l1, arm.l2, arm.g]))
            print(f'QP step-0 matrices exported to {debug_path}')

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
    plot_trajectory_static(trajectory, target_traj_log, dt, save_path=os.path.join(args.dataset_dir, 'trajectory.png'))

if __name__ == '__main__':
    main()

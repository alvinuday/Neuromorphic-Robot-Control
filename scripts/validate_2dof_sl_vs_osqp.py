"""2-DOF MPC validation: Stuart-Landau vs OSQP.

This script uses the repository's canonical 2-DOF model and MPC QP builder,
then compares the SL solver against OSQP on identical QPs.

Run:
  "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" scripts/validate_2dof_sl_vs_osqp.py
"""

import argparse
from pathlib import Path
import sys
import numpy as np


# Allow running this script directly without manually exporting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


def build_osqp_form(qp6):
    """Convert (Q, p, A_eq, b_eq, A_ineq, k_ineq) to OSQP (P, q, A, l, u)."""
    P, q, A_eq, b_eq, A_ineq, k_ineq = qp6
    A = np.vstack([A_eq, A_ineq])
    l = np.concatenate([b_eq, -np.inf * np.ones(A_ineq.shape[0])])
    u = np.concatenate([b_eq, k_ineq])
    return P, q, A, l, u


def run_case(mpc, arm, x0, x_goal, osqp_solver, sl_solver):
    x_ref_traj = mpc.build_reference_trajectory(x0, x_goal)
    qp6 = mpc.build_qp(x0, x_ref_traj)
    P, q, A, l, u = build_osqp_form(qp6)

    x_osqp, info_osqp = osqp_solver.solve(P, q, A, l, u)
    x_sl, info_sl = sl_solver.solve(P, q, A, l, u)

    obj_osqp = float(0.5 * x_osqp @ P @ x_osqp + q @ x_osqp)
    obj_sl = float(0.5 * x_sl @ P @ x_sl + q @ x_sl)

    rel_obj_gap = abs(obj_sl - obj_osqp) / (abs(obj_osqp) + 1e-12)

    # Decision vector layout in current qp_builder is [x0, u0, x1, u1, ..., xN, slacks]
    u0_osqp = x_osqp[arm.nx:arm.nx + arm.nu]
    u0_sl = x_sl[arm.nx:arm.nx + arm.nu]
    u0_err_norm = float(np.linalg.norm(u0_sl - u0_osqp))

    return {
        "obj_osqp": obj_osqp,
        "obj_sl": obj_sl,
        "rel_obj_gap": float(rel_obj_gap),
        "u0_err_norm": u0_err_norm,
        "x_sl_max_abs": float(np.max(np.abs(x_sl))),
        "x_sl_finite": bool(np.all(np.isfinite(x_sl))),
        "status_sl": str(info_sl.get("status", "unknown")),
        "constraint_viol_osqp": float(info_osqp["constraint_viol"]),
        "constraint_viol_sl": float(info_sl["constraint_viol"]),
        "solve_ms_osqp": float(info_osqp["solve_time_ms"]),
        "solve_ms_sl": float(info_sl["solve_time_ms"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate 2-DOF SL solver vs OSQP")
    parser.add_argument("--cases", type=int, default=3, help="Number of predefined test cases to run")
    parser.add_argument("--t_solve", type=float, default=4.0, help="SL ODE integration horizon")
    parser.add_argument("--horizon", type=int, default=5, help="MPC horizon")
    parser.add_argument("--tau_x", type=float, default=0.1)
    parser.add_argument("--tau_lam", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--penalty", type=float, default=50.0)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--no_dual", action="store_true", help="Disable dual dynamics term")
    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Enable OSQP fallback inside SL solver (off by default)",
    )
    args = parser.parse_args()

    np.random.seed(7)

    arm = Arm2DOF()
    mpc = MPCBuilder(arm, N=args.horizon, dt=0.02)

    use_dual = not args.no_dual

    osqp_solver = OSQPSolver(verbose=False)
    sl_solver = StuartLandauLagrangeDirect(
        tau_x=args.tau_x,
        tau_lam=args.tau_lam,
        mu=args.mu,
        T_solve=args.t_solve,
        dt=args.dt,
        constraint_penalty=args.penalty,
        damping=args.damping,
        use_dual=use_dual,
        fallback_to_osqp=args.allow_fallback,
    )

    cases = [
        (np.array([0.10, -0.05, 0.02, -0.01]), np.array([0.60, -0.20, 0.0, 0.0])),
        (np.array([-0.15, 0.08, -0.03, 0.02]), np.array([0.30, 0.35, 0.0, 0.0])),
        (np.array([0.05, 0.12, 0.01, -0.02]), np.array([-0.40, 0.20, 0.0, 0.0])),
    ]

    rows = []
    selected_cases = cases[: max(1, min(args.cases, len(cases)))]
    print(
        f"running cases={len(selected_cases)} horizon={args.horizon} t_solve={args.t_solve} "
        f"tau_x={args.tau_x} tau_lam={args.tau_lam} mu={args.mu} dt={args.dt} "
        f"penalty={args.penalty} damping={args.damping} use_dual={use_dual} "
        f"fallback={args.allow_fallback}"
    )
    for i, (x0, x_goal) in enumerate(selected_cases):
        out = run_case(mpc, arm, x0, x_goal, osqp_solver, sl_solver)
        rows.append(out)
        print(
            f"case={i} rel_obj_gap={out['rel_obj_gap']:.4f} "
            f"u0_err={out['u0_err_norm']:.4f} "
            f"xmax={out['x_sl_max_abs']:.2f} finite={out['x_sl_finite']} "
            f"status_sl={out['status_sl']} "
            f"viol_osqp={out['constraint_viol_osqp']:.3e} "
            f"viol_sl={out['constraint_viol_sl']:.3e} "
            f"t_osqp_ms={out['solve_ms_osqp']:.2f} "
            f"t_sl_ms={out['solve_ms_sl']:.2f}"
        )

    print("\nsummary")
    print(f"mean_rel_obj_gap={np.mean([r['rel_obj_gap'] for r in rows]):.4f}")
    print(f"mean_u0_err={np.mean([r['u0_err_norm'] for r in rows]):.4f}")
    print(f"mean_viol_osqp={np.mean([r['constraint_viol_osqp'] for r in rows]):.3e}")
    print(f"mean_viol_sl={np.mean([r['constraint_viol_sl'] for r in rows]):.3e}")


if __name__ == "__main__":
    main()

"""Systematic no-fallback diagnosis for SL LagONN variants against OSQP.

Runs a parameter sweep over `StuartLandauLaGONN` and `StuartLandauLagONNFull`
on canonical 2-DOF MPC QPs and reports best configurations by weighted score.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from itertools import product

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.stuart_landau_lagonn import StuartLandauLaGONN
from src.solver.stuart_landau_lagonn_full import StuartLandauLagONNFull


def evaluate_solver(mpc, arm, osqp, solver, cases):
    rows = []
    for x0, xg in cases:
        xref = mpc.build_reference_trajectory(x0, xg)
        Q, p, Aeq, beq, Aineq, kineq = mpc.build_qp(x0, xref)
        A = np.vstack([Aeq, Aineq])
        l = np.concatenate([beq, -np.inf * np.ones(Aineq.shape[0])])
        u = np.concatenate([beq, kineq])

        z_osqp, _ = osqp.solve(Q, p, A, l, u)
        z = solver.solve((Q, p, Aeq, beq, Aineq, kineq), verbose=False)

        obj_osqp = float(0.5 * z_osqp @ Q @ z_osqp + p @ z_osqp)
        obj = float(0.5 * z @ Q @ z + p @ z)

        rows.append(
            {
                "rel_gap": abs(obj - obj_osqp) / (abs(obj_osqp) + 1e-12),
                "u0_err": float(np.linalg.norm(z[arm.nx : arm.nx + arm.nu] - z_osqp[arm.nx : arm.nx + arm.nu])),
                "eq_v": float(np.max(np.abs(Aeq @ z - beq))),
                "ineq_v": float(np.max(np.maximum(0.0, Aineq @ z - kineq))),
                "finite": bool(np.all(np.isfinite(z))),
            }
        )

    rel_gap = float(np.mean([r["rel_gap"] for r in rows]))
    u0_err = float(np.mean([r["u0_err"] for r in rows]))
    eq_v = float(np.mean([r["eq_v"] for r in rows]))
    ineq_v = float(np.mean([r["ineq_v"] for r in rows]))
    finite_all = bool(all(r["finite"] for r in rows))

    # Score favors feasibility first, then objective/control match.
    score = 10.0 * eq_v + 10.0 * ineq_v + rel_gap + 0.1 * u0_err
    return {
        "mean_rel_gap": rel_gap,
        "mean_u0_err": u0_err,
        "mean_eq_v": eq_v,
        "mean_ineq_v": ineq_v,
        "finite": finite_all,
        "score": float(score),
    }


def top_k(results, k=5):
    return sorted(results, key=lambda r: r["metrics"]["score"])[:k]


def main():
    parser = argparse.ArgumentParser(description="Diagnose LagONN convergence against OSQP")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--t_solve", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use reduced sweep grid")
    args = parser.parse_args()

    arm = Arm2DOF()
    mpc = MPCBuilder(arm, N=args.horizon, dt=0.02)
    osqp = OSQPSolver(verbose=False)

    cases = [
        (np.array([0.10, -0.05, 0.02, -0.01]), np.array([0.60, -0.20, 0.0, 0.0])),
        (np.array([-0.15, 0.08, -0.03, 0.02]), np.array([0.30, 0.35, 0.0, 0.0])),
        (np.array([0.05, 0.12, 0.01, -0.02]), np.array([-0.40, 0.20, 0.0, 0.0])),
    ]

    if args.quick:
        common = {
            "tau_x": [0.05, 0.1],
            "tau_eq": [0.02, 0.05],
            "tau_ineq": [0.02],
            "lagrange_scale": [1.0, 3.0],
            "eq_penalty": [1.0],
            "ineq_penalty": [1.0],
            "dual_leak": [1e-3, 5e-3],
        }
    else:
        common = {
            "tau_x": [0.05, 0.1, 0.2],
            "tau_eq": [0.02, 0.05, 0.1],
            "tau_ineq": [0.02, 0.05, 0.1],
            "lagrange_scale": [1.0, 3.0, 8.0],
            "eq_penalty": [0.5, 1.0, 3.0],
            "ineq_penalty": [0.5, 1.0, 3.0],
            "dual_leak": [1e-3, 5e-3, 1e-2],
        }

    lagonn_results = []
    full_results = []

    for vals in product(
        common["tau_x"],
        common["tau_eq"],
        common["tau_ineq"],
        common["lagrange_scale"],
        common["eq_penalty"],
        common["ineq_penalty"],
        common["dual_leak"],
    ):
        tau_x, tau_eq, tau_ineq, lagrange_scale, eq_penalty, ineq_penalty, dual_leak = vals
        kwargs = {
            "tau_x": tau_x,
            "tau_eq": tau_eq,
            "tau_ineq": tau_ineq,
            "mu_x": 0.0,
            "T_solve": args.t_solve,
            "adaptive_annealing": False,
            "lagrange_scale": lagrange_scale,
            "eq_penalty": eq_penalty,
            "ineq_penalty": ineq_penalty,
            "dual_leak": dual_leak,
            "convergence_tol": 1e-6,
        }

        s_full = StuartLandauLagONNFull(**kwargs)
        m_full = evaluate_solver(mpc, arm, osqp, s_full, cases)
        full_results.append({"params": kwargs, "metrics": m_full})

        s_simple = StuartLandauLaGONN(
            tau_x=tau_x,
            tau_eq=tau_eq,
            tau_ineq=tau_ineq,
            mu_x=0.0,
            T_solve=args.t_solve,
            dt=0.01,
            convergence_tol=1e-5,
            adaptive_annealing=False,
            eq_penalty=eq_penalty,
            ineq_penalty=ineq_penalty,
            dual_leak=dual_leak,
        )
        m_simple = evaluate_solver(mpc, arm, osqp, s_simple, cases)
        lagonn_results.append(
            {
                "params": {
                    "tau_x": tau_x,
                    "tau_eq": tau_eq,
                    "tau_ineq": tau_ineq,
                    "eq_penalty": eq_penalty,
                    "ineq_penalty": ineq_penalty,
                    "dual_leak": dual_leak,
                },
                "metrics": m_simple,
            }
        )

    print("=== TOP LagONNFull ===")
    for i, row in enumerate(top_k(full_results, k=args.top)):
        print(f"rank={i+1} params={row['params']}")
        print(f"  metrics={row['metrics']}")

    print("\n=== TOP LagONN ===")
    for i, row in enumerate(top_k(lagonn_results, k=args.top)):
        print(f"rank={i+1} params={row['params']}")
        print(f"  metrics={row['metrics']}")


if __name__ == "__main__":
    main()

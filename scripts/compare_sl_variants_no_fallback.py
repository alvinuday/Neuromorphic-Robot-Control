"""Compare SL variants against OSQP without fallback.

This script reports objective gap, first-control error, equality violation,
and inequality violation for each solver variant.
"""

from pathlib import Path
import argparse
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.stuart_landau_lagonn import StuartLandauLaGONN
from src.solver.stuart_landau_lagonn_full import StuartLandauLagONNFull
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


def qp_for_case(mpc, osqp, case):
    x0, xg = case
    xref = mpc.build_reference_trajectory(x0, xg)
    Q, p, Ac, b_eq, A_ineq, k_ineq = mpc.build_qp(x0, xref)
    A = np.vstack([Ac, A_ineq])
    l = np.concatenate([b_eq, -np.inf * np.ones(A_ineq.shape[0])])
    u = np.concatenate([b_eq, k_ineq])
    z_osqp, _ = osqp.solve(Q, p, A, l, u)
    obj_osqp = float(0.5 * z_osqp @ Q @ z_osqp + p @ z_osqp)
    return Q, p, Ac, b_eq, A_ineq, k_ineq, A, l, u, z_osqp, obj_osqp


def metrics(arm, z, Q, p, Ac, b_eq, A_ineq, k_ineq, z_osqp, obj_osqp):
    obj = float(0.5 * z @ Q @ z + p @ z)
    rel_gap = abs(obj - obj_osqp) / (abs(obj_osqp) + 1e-12)
    u0_err = float(np.linalg.norm(z[arm.nx:arm.nx + arm.nu] - z_osqp[arm.nx:arm.nx + arm.nu]))
    eq_violation = float(np.max(np.abs(Ac @ z - b_eq)))
    ineq_violation = float(np.max(np.maximum(0.0, A_ineq @ z - k_ineq)))
    return rel_gap, u0_err, eq_violation, ineq_violation


def main():
    parser = argparse.ArgumentParser(description="Compare no-fallback SL variants")
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--t_solve", type=float, default=1.0)
    args = parser.parse_args()

    arm = Arm2DOF()
    mpc = MPCBuilder(arm, N=5, dt=0.02)
    osqp = OSQPSolver(verbose=False)

    cases = [
        (np.array([0.10, -0.05, 0.02, -0.01]), np.array([0.60, -0.20, 0.0, 0.0])),
        (np.array([-0.15, 0.08, -0.03, 0.02]), np.array([0.30, 0.35, 0.0, 0.0])),
        (np.array([0.05, 0.12, 0.01, -0.02]), np.array([-0.40, 0.20, 0.0, 0.0])),
    ]

    for idx, case in enumerate(cases[: max(1, min(args.cases, len(cases)))]):
        Q, p, Ac, b_eq, A_ineq, k_ineq, A, l, u, z_osqp, obj_osqp = qp_for_case(mpc, osqp, case)

        s_lagonn = StuartLandauLaGONN(
            tau_x=1.0,
            tau_eq=0.1,
            tau_ineq=0.5,
            mu_x=0.0,
            T_solve=args.t_solve,
            convergence_tol=1e-6,
            adaptive_annealing=True,
        )
        z_lagonn = s_lagonn.solve((Q, p, Ac, b_eq, A_ineq, k_ineq), verbose=False)
        m_lagonn = metrics(arm, z_lagonn, Q, p, Ac, b_eq, A_ineq, k_ineq, z_osqp, obj_osqp)

        s_full = StuartLandauLagONNFull(
            tau_x=1.0,
            tau_eq=0.1,
            tau_ineq=0.5,
            mu_x=0.0,
            T_solve=args.t_solve,
            convergence_tol=1e-6,
            adaptive_annealing=True,
            lagrange_scale=10.0,
        )
        z_full = s_full.solve((Q, p, Ac, b_eq, A_ineq, k_ineq), verbose=False)
        m_full = metrics(arm, z_full, Q, p, Ac, b_eq, A_ineq, k_ineq, z_osqp, obj_osqp)

        s_direct = StuartLandauLagrangeDirect(
            tau_x=0.5,
            tau_lam=0.05,
            mu=0.0,
            T_solve=args.t_solve,
            dt=5e-4,
            constraint_penalty=10.0,
            damping=0.05,
            use_dual=True,
            fallback_to_osqp=False,
        )
        z_direct, info_direct = s_direct.solve(Q, p, A, l, u)
        m_direct = metrics(arm, z_direct, Q, p, Ac, b_eq, A_ineq, k_ineq, z_osqp, obj_osqp)

        s_direct_pipg = StuartLandauLagrangeDirect(
            tau_x=0.5,
            tau_lam=0.05,
            mu=0.0,
            T_solve=args.t_solve,
            dt=5e-4,
            constraint_penalty=0.0,
            damping=0.05,
            use_dual=True,
            use_pipg_ineq=True,
            alpha0=1.0,
            beta0=1.0,
            annealing_interval=0.25,
            fallback_to_osqp=False,
        )
        z_direct_pipg, info_direct_pipg = s_direct_pipg.solve(Q, p, A, l, u)
        m_direct_pipg = metrics(arm, z_direct_pipg, Q, p, Ac, b_eq, A_ineq, k_ineq, z_osqp, obj_osqp)

        print(f"case={idx}")
        print(
            f" lagonn      rel={m_lagonn[0]:.4f} u0={m_lagonn[1]:.4f} "
            f"eq={m_lagonn[2]:.3e} ineq={m_lagonn[3]:.3e} "
            f"converged={s_lagonn.get_last_info().get('converged')}"
        )
        print(
            f" lagonn_full rel={m_full[0]:.4f} u0={m_full[1]:.4f} "
            f"eq={m_full[2]:.3e} ineq={m_full[3]:.3e} "
            f"converged={s_full.get_last_info().get('converged')}"
        )
        print(
            f" direct_nf   rel={m_direct[0]:.4f} u0={m_direct[1]:.4f} "
            f"eq={m_direct[2]:.3e} ineq={m_direct[3]:.3e} "
            f"status={info_direct.get('status')}"
        )
        print(
            f" direct_pipg rel={m_direct_pipg[0]:.4f} u0={m_direct_pipg[1]:.4f} "
            f"eq={m_direct_pipg[2]:.3e} ineq={m_direct_pipg[3]:.3e} "
            f"status={info_direct_pipg.get('status')}"
        )


if __name__ == "__main__":
    main()

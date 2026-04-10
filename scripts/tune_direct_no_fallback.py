"""Grid search for best no-fallback StuartLandauLagrangeDirect settings."""

from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


arm = Arm2DOF()
mpc = MPCBuilder(arm, N=5, dt=0.02)
osqp = OSQPSolver(verbose=False)

cases = [
    (np.array([0.10, -0.05, 0.02, -0.01]), np.array([0.60, -0.20, 0.0, 0.0])),
    (np.array([-0.15, 0.08, -0.03, 0.02]), np.array([0.30, 0.35, 0.0, 0.0])),
    (np.array([0.05, 0.12, 0.01, -0.02]), np.array([-0.40, 0.20, 0.0, 0.0])),
]


def build(case):
    x0, xg = case
    xref = mpc.build_reference_trajectory(x0, xg)
    Q, p, Ac, b_eq, A_ineq, k_ineq = mpc.build_qp(x0, xref)
    A = np.vstack([Ac, A_ineq])
    l = np.concatenate([b_eq, -np.inf * np.ones(A_ineq.shape[0])])
    u = np.concatenate([b_eq, k_ineq])
    z_osqp, _ = osqp.solve(Q, p, A, l, u)
    obj_osqp = float(0.5 * z_osqp @ Q @ z_osqp + p @ z_osqp)
    return Q, p, Ac, b_eq, A_ineq, k_ineq, A, l, u, z_osqp, obj_osqp


def metrics(z, Q, p, Ac, b_eq, A_ineq, k_ineq, z_osqp, obj_osqp):
    obj = float(0.5 * z @ Q @ z + p @ z)
    rel_gap = abs(obj - obj_osqp) / (abs(obj_osqp) + 1e-12)
    u0_err = float(np.linalg.norm(z[arm.nx:arm.nx + arm.nu] - z_osqp[arm.nx:arm.nx + arm.nu]))
    eq_violation = float(np.max(np.abs(Ac @ z - b_eq)))
    ineq_violation = float(np.max(np.maximum(0.0, A_ineq @ z - k_ineq)))
    return rel_gap, u0_err, eq_violation, ineq_violation


best = None

for tau_x in [0.1, 0.2, 0.5, 1.0]:
    for tau_lam in [0.01, 0.02, 0.05, 0.1]:
        for penalty in [2.0, 5.0, 10.0, 20.0, 50.0]:
            for damping in [0.0, 0.01, 0.05, 0.1]:
                for use_dual in [False, True]:
                    rels, uerrs, eqs, ineqs = [], [], [], []
                    statuses = []
                    try:
                        for case in cases:
                            Q, p, Ac, b_eq, A_ineq, k_ineq, A, l, u, z_osqp, obj_osqp = build(case)
                            solver = StuartLandauLagrangeDirect(
                                tau_x=tau_x,
                                tau_lam=tau_lam,
                                mu=0.0,
                                T_solve=2.0,
                                dt=5e-4,
                                constraint_penalty=penalty,
                                damping=damping,
                                use_dual=use_dual,
                                fallback_to_osqp=False,
                            )
                            z, info = solver.solve(Q, p, A, l, u)
                            rel, u0, eqv, ineqv = metrics(z, Q, p, Ac, b_eq, A_ineq, k_ineq, z_osqp, obj_osqp)
                            rels.append(rel)
                            uerrs.append(u0)
                            eqs.append(eqv)
                            ineqs.append(ineqv)
                            statuses.append(info.get("status", ""))
                    except Exception:
                        continue

                    score = (np.mean(rels), np.mean(eqs), np.mean(uerrs))
                    row = {
                        "tau_x": tau_x,
                        "tau_lam": tau_lam,
                        "penalty": penalty,
                        "damping": damping,
                        "use_dual": use_dual,
                        "mean_rel": float(np.mean(rels)),
                        "mean_eq": float(np.mean(eqs)),
                        "mean_u0": float(np.mean(uerrs)),
                        "max_eq": float(np.max(eqs)),
                        "statuses": statuses,
                    }
                    if best is None or score < (best["mean_rel"], best["mean_eq"], best["mean_u0"]):
                        best = row
                        print("NEW_BEST", best)

print("FINAL_BEST", best)

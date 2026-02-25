"""
MPC QP Web Inspector — Flask REST API

Wraps existing Arm2DOF, MPCBuilder, OSQPSolver, and qp_inspector
into JSON endpoints for the browser-based inspector frontend.

Run:
    python web_app/server.py          # serves at http://localhost:5050
"""

import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver
from src.utils.qp_inspector import (
    CANONICAL_CASES, build_case, inspect_linearisation, verify_solution
)

app = Flask(__name__, static_folder='static')


def _np(obj):
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        v = float(obj)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return 1e30 if v > 0 else -1e30
        return v
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np(v) for v in obj]
    return obj


def _parse_config(data):
    """Extract arm/mpc/qp config from a JSON request body."""
    params = data.get('params', {})
    m1 = float(params.get('m1', 1.0))
    m2 = float(params.get('m2', 1.0))
    l1 = float(params.get('l1', 0.5))
    l2 = float(params.get('l2', 0.5))
    g  = float(params.get('g', 9.81))

    N  = int(data.get('N', 1))
    dt = float(data.get('dt', 0.1))

    x0 = np.array(data.get('x0', [0, 0, 0, 0]), dtype=float)
    x_goal = np.array(data.get('x_goal', [np.pi/3, np.pi/6, 0, 0]), dtype=float)

    weights = data.get('weights', {})
    Qx = np.diag(weights['Qx']) if 'Qx' in weights else None
    Qf = np.diag(weights['Qf']) if 'Qf' in weights else None
    R  = np.diag(weights['R'])  if 'R'  in weights else None

    bnd = data.get('bounds', {})
    tau_max = float(bnd.get('tau_max', 50.0))
    tau_min = float(bnd.get('tau_min', -50.0))
    theta_max_deg = bnd.get('theta_max', 180.0)
    theta_min_deg = bnd.get('theta_min', -180.0)
    theta_max = np.array([np.deg2rad(theta_max_deg)] * 2)
    theta_min = np.array([np.deg2rad(theta_min_deg)] * 2)

    bounds = {
        'theta_min': theta_min,
        'theta_max': theta_max,
        'tau_min':   np.array([tau_min, tau_min]),
        'tau_max':   np.array([tau_max, tau_max]),
    }

    arm = Arm2DOF(m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    mpc = MPCBuilder(arm, N=N, dt=dt, Qx=Qx, Qf=Qf, R=R, bounds=bounds)

    return arm, mpc, x0, x_goal, N, dt


def _z_labels(N, nx=4, nu=2, nq=2):
    """Generate human-readable labels for every element of z."""
    state_names = ['q1', 'q2', 'dq1', 'dq2']
    ctrl_names  = ['tau1', 'tau2']
    labels = []
    for k in range(N):
        for s in state_names:
            labels.append(f'x{k}[{s}]')
        for c in ctrl_names:
            labels.append(f'u{k}[{c}]')
    for s in state_names:
        labels.append(f'x{N}[{s}]')
    for k in range(N + 1):
        for j in range(nq):
            labels.append(f's{k}[{j}]')
    return labels


def _decompose_z(z_star, N, nx=4, nu=2, nq=2):
    """Split z* into per-step x_k, u_k and slacks."""
    x_steps = []
    u_steps = []
    for k in range(N):
        start = k * (nx + nu)
        x_steps.append(z_star[start:start + nx].tolist())
        u_steps.append(z_star[start + nx:start + nx + nu].tolist())
    start_xN = N * (nx + nu)
    x_steps.append(z_star[start_xN:start_xN + nx].tolist())
    n_z = N * (nx + nu) + nx
    slacks = z_star[n_z:].tolist()
    return {'x_k': x_steps, 'u_k': u_steps, 'slacks': slacks}


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/cases', methods=['GET'])
def get_cases():
    cases = []
    for name, cfg in CANONICAL_CASES.items():
        cases.append({
            'name':        name,
            'description': cfg.get('description', ''),
            'g':           cfg.get('g', 9.81),
            'q0':          cfg.get('q0', [0, 0]),
            'dq0':         cfg.get('dq0', [0, 0]),
            'x_goal':      _np(cfg.get('x_goal', [np.pi/3, np.pi/6, 0, 0])),
            'N':           cfg.get('N', 1),
            'dt':          cfg.get('dt', 0.1),
            'tau_max':     cfg.get('tau_max', 50.0),
        })
    return jsonify(cases=cases)


@app.route('/api/build', methods=['POST'])
def api_build():
    data = request.get_json(force=True)
    try:
        arm, mpc, x0, x_goal, N, dt = _parse_config(data)
        ref_traj = mpc.build_reference_trajectory(x0, x_goal)
        qp_matrices = mpc.build_qp(x0, ref_traj)
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices

        nx, nu, nq = arm.nx, arm.nu, arm.nq
        n_z = N * (nx + nu) + nx
        n_slack = (N + 1) * nq
        n_z_total = n_z + n_slack

        # Linearisation at step k=0
        x_bar = ref_traj[0]
        q_bar = x_bar[:nq]
        th1, th2 = q_bar[0], q_bar[1]
        m1, m2, l1, l2, g = arm.m1, arm.m2, arm.l1, arm.l2, arm.g
        G1 = (m1*l1/2 + m2*l1)*g*np.sin(th1) + m2*l2/2*g*np.sin(th1+th2)
        G2 = m2*l2/2*g*np.sin(th1+th2)
        u_bar = np.array([G1, G2])

        f_val = np.array(mpc.f_fun(x_bar, u_bar)).flatten()
        Ac = np.array(mpc.A_fun(x_bar, u_bar))
        Bc = np.array(mpc.B_fun(x_bar, u_bar))
        Ad = np.eye(nx) + dt * Ac
        Bd = dt * Bc
        c_k = dt * (f_val - Ac @ x_bar - Bc @ u_bar)

        M_num = np.array(arm.M_fun(q_bar))
        try:
            M_inv = np.linalg.inv(M_num)
        except np.linalg.LinAlgError:
            M_inv = np.full_like(M_num, np.nan)

        G_vec = np.array([G1, G2])
        dG11 = (m1*l1/2 + m2*l1)*g*np.cos(th1) + m2*l2/2*g*np.cos(th1+th2)
        dG12 = m2*l2/2*g*np.cos(th1+th2)
        dG21 = dG12
        dG22 = dG12
        dG_dq = np.array([[dG11, dG12], [dG21, dG22]])

        z_labels = _z_labels(N, nx, nu, nq)

        fk_points = []
        for k in range(N + 1):
            pts = arm.forward_kinematics(ref_traj[k][:nq])
            fk_points.append(pts.tolist())

        result = {
            'dimensions': {
                'nx': nx, 'nu': nu, 'nq': nq, 'N': N, 'dt': dt,
                'n_z': n_z, 'n_slack': n_slack, 'n_z_total': n_z_total,
                'n_eq': A_eq.shape[0], 'n_ineq': A_ineq.shape[0],
            },
            'linearisation': {
                'x_bar': x_bar, 'u_bar': u_bar, 'f_val': f_val,
                'Ac': Ac, 'Bc': Bc, 'Ad': Ad, 'Bd': Bd, 'c_k': c_k,
                'M': M_num, 'M_inv': M_inv, 'G': G_vec, 'dG_dq': dG_dq,
            },
            'qp': {
                'Q': Q, 'p': p,
                'A_eq': A_eq, 'b_eq': b_eq,
                'A_ineq': A_ineq, 'k_ineq': k_ineq,
            },
            'ref_traj': ref_traj,
            'ref_fk': fk_points,
            'x0': x0, 'x_goal': x_goal,
            'z_labels': z_labels,
            'params': {'m1': arm.m1, 'm2': arm.m2, 'l1': arm.l1, 'l2': arm.l2, 'g': arm.g},
        }
        return jsonify(_np(result))

    except Exception as e:
        return jsonify(error=str(e)), 400


@app.route('/api/solve', methods=['POST'])
def api_solve():
    data = request.get_json(force=True)
    try:
        arm, mpc, x0, x_goal, N, dt = _parse_config(data)
        ref_traj = mpc.build_reference_trajectory(x0, x_goal)
        qp_matrices = mpc.build_qp(x0, ref_traj)
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices

        nx, nu, nq = arm.nx, arm.nu, arm.nq
        solver = OSQPSolver()
        z_star = solver.solve(qp_matrices)

        if z_star is None:
            return jsonify(error='OSQP returned infeasible/unsolved', status='infeasible'), 200

        vr = verify_solution(z_star, qp_matrices, tol=1e-4, verbose=False)

        decomp = _decompose_z(z_star, N, nx, nu, nq)

        pred_fk = []
        for xk in decomp['x_k']:
            pts = arm.forward_kinematics(np.array(xk[:nq]))
            pred_fk.append(pts.tolist())

        result = {
            'status': 'solved',
            'z_star': z_star,
            'z_decomposed': decomp,
            'predicted_fk': pred_fk,
            'objective': vr['objective'],
            'eq_norm': vr['eq_norm'],
            'ineq_viol': vr['ineq_viol'],
            'kkt_resid': vr['kkt_resid'],
            'comp_slack': vr['comp_slack'],
            'passed': vr['passed'],
        }
        return jsonify(_np(result))

    except Exception as e:
        return jsonify(error=str(e), status='error'), 400


@app.route('/api/fk', methods=['POST'])
def api_fk():
    data = request.get_json(force=True)
    theta = np.array(data.get('theta', [0, 0]), dtype=float)
    params = data.get('params', {})
    l1 = float(params.get('l1', 0.5))
    l2 = float(params.get('l2', 0.5))
    arm = Arm2DOF(l1=l1, l2=l2, g=0)
    pts = arm.forward_kinematics(theta)
    return jsonify(points=_np(pts))


if __name__ == '__main__':
    print('MPC QP Web Inspector → http://localhost:5050')
    app.run(host='0.0.0.0', port=5050, debug=True)

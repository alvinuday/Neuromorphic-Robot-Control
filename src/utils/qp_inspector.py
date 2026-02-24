"""
QP Inspector — Canonical test-case builder and matrix verification tool.

Usage:
    from src.utils.qp_inspector import build_case, print_matrices, verify_solution, run_all_cases

    # Build Case A and inspect matrices
    result = build_case('A')
    print_matrices(result['qp_matrices'], result['N'])

    # Run all canonical cases and verify OSQP solutions
    run_all_cases()
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver

# ---------------------------------------------------------------------------
# Canonical case definitions
# ---------------------------------------------------------------------------
# Each entry is used as keyword arguments to build_case().
# Override any field with build_case('A', N=2, dt=0.02, ...).
CANONICAL_CASES = {
    'A': {
        'g': 0.0,
        'q0': [0.0, 0.0],
        'dq0': [0.0, 0.0],
        'x_goal': [np.pi / 3, np.pi / 6, 0.0, 0.0],
        'N': 1,
        'dt': 0.1,
        'description': (
            'g=0, q=[0,0], dq=[0,0]  →  True pure double integrator. '
            'G=0, ∂G/∂q=0, C=0, Ac=diag-block integrator, c_k=0.'
        ),
    },
    'B': {
        'g': 9.81,
        'q0': [0.0, 0.0],
        'dq0': [0.0, 0.0],
        'x_goal': [np.pi / 3, np.pi / 6, 0.0, 0.0],
        'N': 1,
        'dt': 0.1,
        'description': (
            'g=9.81, q=[0,0], dq=[0,0]  →  Hanging equilibrium. '
            'G=0, u_bar=0, but gravity gradient ∂G/∂q ≠ 0 (uses cos, not sin). '
            'Ac lower-left is non-zero. c_k=0.'
        ),
    },
    'C': {
        'g': 9.81,
        'q0': [np.pi / 6, 0.0],
        'dq0': [0.0, 0.0],
        'x_goal': [np.pi / 3, np.pi / 6, 0.0, 0.0],
        'N': 1,
        'dt': 0.1,
        'description': (
            'g=9.81, q=[π/6,0], dq=[0,0]  →  G≠0 (gravity torque active). '
            'u_bar=G=[4.905, 1.226] Nm. Gravity gradient uses cos(π/6). '
            'c_k≠0 because x_bar≠0.'
        ),
    },
    'D': {
        'g': 9.81,
        'q0': [0.0, np.pi / 2],
        'dq0': [0.0, 0.0],
        'x_goal': [np.pi / 3, np.pi / 6, 0.0, 0.0],
        'N': 1,
        'dt': 0.1,
        'description': (
            'g=9.81, q=[0,π/2], dq=[0,0]  →  cos(q2)=0 → M is "decoupled" form. '
            'M⁻¹=[[3,-3],[-3,15]]. Gradient row 2 is zero (cos(π/2)=0). '
            'Bc changes shape vs Cases A/B/C.'
        ),
    },
    'E': {
        'g': 9.81,
        'q0': [0.0, np.pi / 4],
        'dq0': [1.0, -0.5],
        'x_goal': [np.pi / 3, np.pi / 6, 0.0, 0.0],
        'N': 1,
        'dt': 0.1,
        'description': (
            'g=9.81, q=[0,π/4], dq=[1,-0.5]  →  Full general snapshot. '
            'Coriolis C≠0, lower-right block of Ac≠0. '
            'Full ∂(Cq̇)/∂q and ∂(Cq̇)/∂q̇ needed (NOT simply −M⁻¹C).'
        ),
    },
    'A_N2': {
        'g': 0.0,
        'q0': [0.0, 0.0],
        'dq0': [0.0, 0.0],
        'x_goal': [np.pi / 3, np.pi / 6, 0.0, 0.0],
        'N': 2,
        'dt': 0.1,
        'description': 'Case A (g=0) with N=2 for hand derivation of two-step QP.',
    },
    'B_N2': {
        'g': 9.81,
        'q0': [0.0, 0.0],
        'dq0': [0.0, 0.0],
        'x_goal': [np.pi / 3, np.pi / 6, 0.0, 0.0],
        'N': 2,
        'dt': 0.1,
        'description': 'Case B (g=9.81, hanging) with N=2 for two-step QP.',
    },
    'C_tight': {
        'g': 9.81,
        'q0': [np.pi / 6, 0.0],
        'dq0': [0.0, 0.0],
        'x_goal': [np.pi / 2, np.pi / 4, 0.0, 0.0],
        'N': 1,
        'dt': 0.1,
        'tau_max': 2.0,
        'description': (
            'Case C with tight τ_max=2 Nm  →  torque constraint becomes active. '
            'Use to study KKT multipliers and active-constraint behaviour.'
        ),
    },
}

# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_case(name='A', **overrides):
    """
    Build the complete QP for a canonical case and return all matrices.

    Parameters
    ----------
    name : str
        Key in CANONICAL_CASES ('A', 'B', 'C', 'D', 'E', 'A_N2', 'B_N2', 'C_tight').
    **overrides
        Any key from the case dict or extra MPCBuilder constructor arguments
        (Qx, Qf, R, Qs, theta_min, theta_max, tau_min, tau_max).

    Returns
    -------
    dict with keys:
        arm           : Arm2DOF instance
        mpc           : MPCBuilder instance
        qp_matrices   : (Q, p, A_eq, b_eq, A_ineq, k_ineq)
        x0            : np.ndarray [nx]  initial state
        x_ref_traj    : np.ndarray [N+1, nx]  reference trajectory
        x_goal        : np.ndarray [nx]
        N             : int
        dt            : float
        n_z           : int  decision vars (no slacks)
        n_slack       : int  slack variables
        n_z_total     : int  total decision vars
        n_eq          : int  equality constraint rows
        n_ineq        : int  inequality constraint rows
        description   : str
        params        : dict  arm physical parameters
    """
    if name not in CANONICAL_CASES:
        raise ValueError(f"Unknown case '{name}'. Options: {list(CANONICAL_CASES.keys())}")

    cfg = dict(CANONICAL_CASES[name])
    cfg.update(overrides)

    # Physical parameters
    g       = cfg.pop('g', 9.81)
    m1      = cfg.pop('m1', 1.0)
    m2      = cfg.pop('m2', 1.0)
    l1      = cfg.pop('l1', 0.5)
    l2      = cfg.pop('l2', 0.5)

    # Initial / goal states
    q0      = np.array(cfg.pop('q0', [0.0, 0.0]))
    dq0     = np.array(cfg.pop('dq0', [0.0, 0.0]))
    x0      = np.concatenate([q0, dq0])
    x_goal  = np.array(cfg.pop('x_goal', [np.pi / 3, np.pi / 6, 0.0, 0.0]))

    # Horizon / timestep
    N   = cfg.pop('N', 1)
    dt  = cfg.pop('dt', 0.1)
    description = cfg.pop('description', '')

    # Optional weight / bound overrides
    Qx = cfg.pop('Qx', None)
    Qf = cfg.pop('Qf', None)
    R  = cfg.pop('R', None)

    tau_max_val = cfg.pop('tau_max', 50.0)
    tau_min_val = cfg.pop('tau_min', -50.0)
    theta_max   = cfg.pop('theta_max', np.array([np.pi, np.pi]))
    theta_min   = cfg.pop('theta_min', np.array([-np.pi, -np.pi]))

    bounds = {
        'theta_min': np.atleast_1d(theta_min),
        'theta_max': np.atleast_1d(theta_max),
        'tau_min':   np.atleast_1d(tau_min_val) * np.ones(2),
        'tau_max':   np.atleast_1d(tau_max_val) * np.ones(2),
    }

    # Build objects
    arm = Arm2DOF(m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    mpc = MPCBuilder(arm, N=N, dt=dt, Qx=Qx, Qf=Qf, R=R, bounds=bounds)

    x_ref_traj  = mpc.build_reference_trajectory(x0, x_goal)
    qp_matrices = mpc.build_qp(x0, x_ref_traj)

    Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
    nx, nu, nq = arm.nx, arm.nu, arm.nq
    n_z       = N * (nx + nu) + nx
    n_slack   = (N + 1) * nq
    n_z_total = n_z + n_slack

    return {
        'arm':         arm,
        'mpc':         mpc,
        'qp_matrices': qp_matrices,
        'x0':          x0,
        'x_goal':      x_goal,
        'x_ref_traj':  x_ref_traj,
        'N':           N,
        'dt':          dt,
        'n_z':         n_z,
        'n_slack':     n_slack,
        'n_z_total':   n_z_total,
        'n_eq':        A_eq.shape[0],
        'n_ineq':      A_ineq.shape[0],
        'description': description,
        'params':      dict(m1=m1, m2=m2, l1=l1, l2=l2, g=g),
    }


# ---------------------------------------------------------------------------
# Linearisation inspector (for hand-derivation verification)
# ---------------------------------------------------------------------------

def inspect_linearisation(case_name='A', **overrides):
    """
    Evaluate Ac, Bc, f, G, M at the operating point for a given case.

    Returns a dict with:
        x_bar, u_bar, f_val, Ac, Bc, Ad, Bd, c_k,
        M_num, M_inv_num, G_num, dG_dq_num
    """
    r = build_case(case_name, **overrides)
    arm, mpc, x_ref_traj = r['arm'], r['mpc'], r['x_ref_traj']
    dt = r['dt']

    x_bar  = x_ref_traj[0]
    q_bar  = x_bar[:arm.nq]
    dq_bar = x_bar[arm.nq:]

    # Gravity compensation torque at operating point
    th1, th2 = q_bar[0], q_bar[1]
    m1, m2, l1, l2, g = arm.m1, arm.m2, arm.l1, arm.l2, arm.g
    G1 = (m1 * l1 / 2 + m2 * l1) * g * np.sin(th1) + m2 * l2 / 2 * g * np.sin(th1 + th2)
    G2 = m2 * l2 / 2 * g * np.sin(th1 + th2)
    u_bar = np.array([G1, G2])

    # CasADi evaluations
    f_val = np.array(mpc.f_fun(x_bar, u_bar)).flatten()
    Ac    = np.array(mpc.A_fun(x_bar, u_bar))
    Bc    = np.array(mpc.B_fun(x_bar, u_bar))

    # Discretisation (Euler, matching build_qp)
    Ad  = np.eye(arm.nx) + dt * Ac
    Bd  = dt * Bc
    c_k = dt * (f_val - Ac @ x_bar - Bc @ u_bar)

    # Mass matrix and gravity gradient (numerical)
    import casadi as ca
    M_num   = np.array(arm.M_fun(q_bar))

    # Gravity gradient ∂G/∂q via finite differences (for verification)
    eps = 1e-6

    def G_func(q):
        t1, t2 = q[0], q[1]
        _G1 = (m1*l1/2 + m2*l1)*g*np.sin(t1) + m2*l2/2*g*np.sin(t1+t2)
        _G2 = m2*l2/2*g*np.sin(t1+t2)
        return np.array([_G1, _G2])

    dG_dq = np.zeros((arm.nq, arm.nq))
    for j in range(arm.nq):
        eq = q_bar.copy(); eq[j] += eps
        dG_dq[:, j] = (G_func(eq) - G_func(q_bar)) / eps

    # Analytical gravity gradient
    dG11 = (m1*l1/2 + m2*l1)*g*np.cos(th1) + m2*l2/2*g*np.cos(th1+th2)
    dG12 = m2*l2/2*g*np.cos(th1+th2)
    dG21 = m2*l2/2*g*np.cos(th1+th2)
    dG22 = m2*l2/2*g*np.cos(th1+th2)
    dG_dq_analytic = np.array([[dG11, dG12], [dG21, dG22]])

    try:
        M_inv = np.linalg.inv(M_num)
    except np.linalg.LinAlgError:
        M_inv = None

    return {
        'x_bar':           x_bar,
        'u_bar':           u_bar,
        'G_num':           G_func(q_bar),
        'f_val':           f_val,
        'Ac':              Ac,
        'Bc':              Bc,
        'Ad':              Ad,
        'Bd':              Bd,
        'c_k':             c_k,
        'M_num':           M_num,
        'M_inv':           M_inv,
        'dG_dq_num':       dG_dq,
        'dG_dq_analytic':  dG_dq_analytic,
        'M_inv_times_dG':  None if M_inv is None else M_inv @ dG_dq_analytic,
    }


# ---------------------------------------------------------------------------
# Matrix printer
# ---------------------------------------------------------------------------

def print_matrices(qp_matrices, N, nx=4, nu=2, nq=2, precision=4, case_name=''):
    """
    Pretty-print all six QP matrices with index annotations.
    """
    Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
    n_z_total = Q.shape[0]
    n_z       = N * (nx + nu) + nx
    n_slack   = (N + 1) * nq

    sep = '─' * 70
    fmt = f'{{:.{precision}f}}'

    def _hdr(title):
        print(f'\n{sep}')
        print(f'  {title}')
        print(sep)

    def _mat(M, row_labels=None, col_labels=None, max_rows=None, max_cols=None):
        r, c = M.shape
        if max_rows and r > max_rows:
            print(f'  [matrix {r}×{c} — showing first {max_rows} rows]')
            M = M[:max_rows, :]
            r = max_rows
        if max_cols and c > max_cols:
            M = M[:, :max_cols]
        for i in range(r):
            lbl = f'  [{row_labels[i]:>14}]' if row_labels else '  '
            vals = '  '.join(fmt.format(v) for v in M[i])
            print(lbl + vals)

    header = f'Case {case_name}' if case_name else ''
    print(f'\n{"="*70}')
    print(f'  QP MATRICES  {header}')
    print(f'  N={N}, nx={nx}, nu={nu}, nq={nq}')
    print(f'  n_z_total={n_z_total}  (n_z={n_z} + n_slack={n_slack})')
    print(f'  Rows: A_eq={A_eq.shape[0]}, A_ineq={A_ineq.shape[0]}')
    print(f'{"="*70}')

    # z-variable layout
    print('\n  z-variable layout:')
    for k in range(N + 1):
        x_start = k * (nx + nu)
        print(f'    x_{k}: z[{x_start}:{x_start+nx}]', end='')
        if k < N:
            u_start = x_start + nx
            print(f'   u_{k}: z[{u_start}:{u_start+nu}]', end='')
        print()
    print(f'  Slacks: slack_start={n_z}')
    for k in range(N + 1):
        s_start = n_z + k * nq
        print(f'    s_{k}: z[{s_start}:{s_start+nq}]')

    # Cost Hessian diagonal (compact)
    _hdr('Q (cost Hessian) — diagonal entries')
    diag_Q = np.diag(Q)
    print('  [NOTE: code stores 2×Qx, 2×R, 2×Qf, 2×Qs so OSQP\'s ½z\'Qz = Qx-weighted cost]')
    for i in range(n_z_total):
        role = _z_role(i, N, nx, nu, nq, n_z)
        print(f'  Q[{i:3d},{i:3d}] = {fmt.format(diag_Q[i])}   ({role})')

    # Linear cost vector
    _hdr('p (linear cost vector)')
    for i in range(n_z_total):
        if abs(p[i]) > 1e-12:
            role = _z_role(i, N, nx, nu, nq, n_z)
            print(f'  p[{i:3d}] = {fmt.format(p[i])}   ({role})')
    if np.all(np.abs(p) < 1e-12):
        print('  (all zeros)')

    # Equality constraints
    _hdr(f'A_eq ({A_eq.shape[0]}×{A_eq.shape[1]}) and b_eq')
    row_labels_eq = []
    for k_blk in range(N + 1):
        prefix = 'IC' if k_blk == 0 else f'dyn k={k_blk-1}'
        for dim in range(nx):
            row_labels_eq.append(f'{prefix} row{dim}')
    for i in range(A_eq.shape[0]):
        nonz_cols = np.where(np.abs(A_eq[i]) > 1e-12)[0]
        lbl = row_labels_eq[i] if i < len(row_labels_eq) else f'row{i}'
        col_desc = ', '.join(
            f'z[{j}]={fmt.format(A_eq[i,j])} ({_z_role(j,N,nx,nu,nq,n_z)})'
            for j in nonz_cols
        )
        print(f'  [{lbl:>12}]  b={fmt.format(b_eq[i])}   {col_desc}')

    # Inequality constraints
    _hdr(f'A_ineq ({A_ineq.shape[0]}×{A_ineq.shape[1]}) and k_ineq')
    for i in range(A_ineq.shape[0]):
        nonz_cols = np.where(np.abs(A_ineq[i]) > 1e-12)[0]
        col_desc = ', '.join(
            f'z[{j}]={fmt.format(A_ineq[i,j])} ({_z_role(j,N,nx,nu,nq,n_z)})'
            for j in nonz_cols
        )
        print(f'  [ineq {i:3d}]  ≤ {fmt.format(k_ineq[i])}   {col_desc}')


def _z_role(idx, N, nx, nu, nq, n_z):
    """Human-readable label for a z-vector index."""
    if idx >= n_z:
        s_offset = idx - n_z
        k_s = s_offset // nq
        dim = s_offset % nq
        return f's_{k_s}[{dim}]'
    step_width = nx + nu
    k = idx // step_width
    pos  = idx % step_width
    if k <= N:
        if pos < nx:
            state_names = ['q1', 'q2', 'dq1', 'dq2']
            if pos < len(state_names):
                return f'x_{k}[{state_names[pos]}]'
            return f'x_{k}[{pos}]'
        else:
            u_pos = pos - nx
            return f'u_{k}[tau{u_pos+1}]'
    return f'z[{idx}]'


# ---------------------------------------------------------------------------
# Solution verifier
# ---------------------------------------------------------------------------

def verify_solution(z_star, qp_matrices, tol=1e-4, verbose=True):
    """
    Verify an OSQP solution z_star against all QP components.

    Checks:
      1. Dynamics feasibility: ||A_eq z* - b_eq||
      2. Inequality satisfaction: max(A_ineq z* - k_ineq)
      3. KKT stationarity: ||Q z* + p + A_eq^T λ_eq + A_ineq^T λ_ineq||
         (λ computed from KKT via least-squares)
      4. Objective value ½ z*^T Q z* + p^T z*

    Returns dict with all residual norms and pass/fail booleans.
    """
    Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices

    # 1. Equality residual
    eq_resid = A_eq @ z_star - b_eq
    eq_norm  = np.linalg.norm(eq_resid)

    # 2. Inequality violation
    ineq_val = A_ineq @ z_star - k_ineq
    ineq_viol = np.max(ineq_val)
    ineq_max_idx = int(np.argmax(ineq_val))

    # 3. Objective value
    obj = 0.5 * z_star @ Q @ z_star + p @ z_star

    # 4. KKT stationarity (approximate λ via least-squares on combined constraints)
    A_all = np.vstack([A_eq, A_ineq])
    grad_obj = Q @ z_star + p
    # Stationarity: grad_obj + A_all^T λ = 0 → λ = -(A_all A_all^T)^{-1} A_all grad_obj
    # Use least-squares
    lam_ls, _, _, _ = np.linalg.lstsq(A_all.T, -grad_obj, rcond=None)
    kkt_resid = np.linalg.norm(grad_obj + A_all.T @ lam_ls)

    # Complementarity slackness for inequalities
    n_eq = A_eq.shape[0]
    lam_ineq = lam_ls[n_eq:]
    slack   = k_ineq - A_ineq @ z_star
    comp_slack = np.abs(lam_ineq * slack)
    comp_norm  = np.max(comp_slack) if len(comp_slack) > 0 else 0.0

    passed = (
        eq_norm   < tol and
        ineq_viol < tol and
        kkt_resid < tol * 100   # KKT residual uses least-squares approx, looser
    )

    results = {
        'eq_norm':       eq_norm,
        'ineq_viol':     ineq_viol,
        'ineq_max_idx':  ineq_max_idx,
        'kkt_resid':     kkt_resid,
        'comp_slack':    comp_norm,
        'objective':     obj,
        'passed':        passed,
    }

    if verbose:
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'  {status}   eq_norm={eq_norm:.2e}  ineq_viol={ineq_viol:.2e}'
              f'  kkt={kkt_resid:.2e}  obj={obj:.6f}')

    return results


# ---------------------------------------------------------------------------
# NPZ exporter
# ---------------------------------------------------------------------------

def export_case_npz(case_name='A', save_path=None, solve=True, **overrides):
    """
    Build a case, optionally solve it with OSQP, and save all matrices to .npz.

    Parameters
    ----------
    case_name : str
    save_path : str or None  (default: 'data/qp_inspector/<case_name>.npz')
    solve     : bool  — also save z_star and verification metrics
    **overrides : passed to build_case()

    Returns
    -------
    str: path to saved file
    """
    r = build_case(case_name, **overrides)
    Q, p, A_eq, b_eq, A_ineq, k_ineq = r['qp_matrices']

    save_dict = dict(
        Q=Q, p=p,
        A_eq=A_eq, b_eq=b_eq,
        A_ineq=A_ineq, k_ineq=k_ineq,
        x0=r['x0'],
        x_goal=r['x_goal'],
        x_ref_traj=r['x_ref_traj'],
    )

    if solve:
        solver = OSQPSolver()
        z_star = solver.solve(r['qp_matrices'])
        if z_star is not None:
            vr = verify_solution(z_star, r['qp_matrices'], verbose=False)
            save_dict['z_star']     = z_star
            save_dict['obj']        = np.array([vr['objective']])
            save_dict['eq_norm']    = np.array([vr['eq_norm']])
            save_dict['ineq_viol']  = np.array([vr['ineq_viol']])
            save_dict['kkt_resid']  = np.array([vr['kkt_resid']])

    if save_path is None:
        out_dir = os.path.join('data', 'qp_inspector')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'case_{case_name}.npz')

    np.savez(save_path, **save_dict)
    print(f'  Saved: {save_path}')
    return save_path


# ---------------------------------------------------------------------------
# Full verification run
# ---------------------------------------------------------------------------

def run_all_cases(cases=None, verbose_matrices=False):
    """
    Run all canonical cases through OSQP and print a verification summary.

    Parameters
    ----------
    cases : list of str or None (all cases)
    verbose_matrices : bool  — print full matrix blocks for each case
    """
    if cases is None:
        cases = list(CANONICAL_CASES.keys())

    solver = OSQPSolver()
    all_passed = True

    print('\n' + '=' * 70)
    print('  QP INSPECTOR — Canonical Case Verification')
    print('=' * 70)

    for name in cases:
        cfg = CANONICAL_CASES[name]
        print(f'\n── Case {name} ─────────────────────────────────────────────────')
        print(f'  {cfg["description"]}')

        r = build_case(name)
        Q, p, A_eq, b_eq, A_ineq, k_ineq = r['qp_matrices']

        print(f'  z layout: n_z_total={r["n_z_total"]} '
              f'(n_z={r["n_z"]} + n_slack={r["n_slack"]})')
        print(f'  A_eq: {A_eq.shape}   A_ineq: {A_ineq.shape}')
        print(f'  x0={np.round(r["x0"], 4)}   x_goal={np.round(r["x_goal"], 4)}')

        # Linearisation summary
        lin = inspect_linearisation(name)
        print(f'  Operating point: x_bar={np.round(lin["x_bar"], 4)}'
              f'  u_bar={np.round(lin["u_bar"], 4)}')
        print(f'  G(q_bar)={np.round(lin["G_num"], 4)}')
        print(f'  ||f(x_bar,u_bar)||={np.linalg.norm(lin["f_val"]):.2e}'
              f'  (should be ~0 at equilibrium)')
        print(f'  ||c_k||={np.linalg.norm(lin["c_k"]):.4f}')

        Ac = lin['Ac']
        print(f'  Ac lower-left:\n    {np.round(Ac[2:, :2], 4).tolist()}')
        print(f'  Ac lower-right:\n    {np.round(Ac[2:, 2:], 4).tolist()}')

        dG = lin['dG_dq_analytic']
        print(f'  ∂G/∂q (analytic):\n    {np.round(dG, 4).tolist()}')
        dG_err = np.max(np.abs(lin['dG_dq_num'] - lin['dG_dq_analytic']))
        print(f'  ∂G/∂q finite-diff error: {dG_err:.2e}')

        if verbose_matrices:
            print_matrices(r['qp_matrices'], r['N'], case_name=name)

        # Solve and verify
        print(f'\n  Solving with OSQP ...')
        z_star = solver.solve(r['qp_matrices'])

        if z_star is None:
            print('  OSQP: INFEASIBLE / FAILED')
            all_passed = False
            continue

        nx, nu = r['arm'].nx, r['arm'].nu
        u0_star = z_star[nx: nx + nu]
        print(f'  u0* = {np.round(u0_star, 6)}')

        vr = verify_solution(z_star, r['qp_matrices'], verbose=True)

        if not vr['passed']:
            all_passed = False

    print('\n' + '=' * 70)
    print(f'  Overall: {"ALL PASSED ✓" if all_passed else "SOME FAILED ✗"}')
    print('=' * 70)
    return all_passed


# ---------------------------------------------------------------------------
# Hand-derivation helper: print linearisation step-by-step
# ---------------------------------------------------------------------------

def print_linearisation_steps(case_name='A', dt=None, **overrides):
    """
    Print a full step-by-step linearisation report for hand verification.
    """
    if dt is not None:
        overrides['dt'] = dt

    r   = build_case(case_name, **overrides)
    lin = inspect_linearisation(case_name, **overrides)
    arm = r['arm']
    N   = r['N']
    _dt = r['dt']

    sep  = '─' * 60
    def s(label, val, fmt='.6f'):
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                print(f'  {label} = {np.round(val, 6).tolist()}')
            else:
                print(f'  {label} =')
                for row in val:
                    print(f'    {np.round(row, 6).tolist()}')
        else:
            print(f'  {label} = {val:{fmt}}')

    print(f'\n{"="*60}')
    print(f'  Linearisation Steps — Case {case_name}')
    print(f'{"="*60}')

    p = r['params']
    print(f'\n  Physical parameters:')
    print(f'    m1={p["m1"]}, m2={p["m2"]}, l1={p["l1"]}, l2={p["l2"]}, g={p["g"]}')
    print(f'    N={N}, dt={_dt}')

    print(f'\n{sep}  Step 1: Operating point')
    s('x_bar', lin['x_bar'])
    s('q_bar', lin['x_bar'][:arm.nq])
    s('dq_bar', lin['x_bar'][arm.nq:])

    print(f'\n{sep}  Step 2: Inertia matrix M(q_bar)')
    s('M', lin['M_num'])
    det_M = np.linalg.det(lin['M_num'])
    s('det(M)', det_M)
    s('M_inv', lin['M_inv'])

    print(f'\n{sep}  Step 3: Gravity vector G(q_bar) and equilibrium torque u_bar')
    s('G(q_bar)', lin['G_num'])
    s('u_bar = G', lin['u_bar'])

    print(f'\n{sep}  Step 4: Gravity gradient ∂G/∂q (uses cosines, NOT sines!)')
    s('∂G/∂q (analytic)', lin['dG_dq_analytic'])
    s('∂G/∂q (finite-diff check)', lin['dG_dq_num'])
    s('max error', np.max(np.abs(lin['dG_dq_num'] - lin['dG_dq_analytic'])))

    print(f'\n{sep}  Step 5: f(x_bar, u_bar) = ẋ at operating point')
    s('f_val', lin['f_val'])
    print('  (should be ≈ 0 at static equilibrium, non-zero if dq_bar ≠ 0)')

    print(f'\n{sep}  Step 6: Ac = ∂f/∂x  (4×4 Jacobian)')
    s('Ac', lin['Ac'])
    print('  Block structure:  Ac = [[0, I], [-M⁻¹ ∂G/∂q − ∂(Cq̇)/∂q,  −M⁻¹ ∂(Cq̇)/∂q̇]]')

    print(f'\n{sep}  Step 7: Bc = ∂f/∂u  (4×2 Jacobian)')
    s('Bc', lin['Bc'])
    print('  Bc = [0₂ₓ₂; M⁻¹]  — only acceleration rows depend on τ')

    print(f'\n{sep}  Step 8: Euler discretisation (dt={_dt})')
    s('Ad = I + dt*Ac', lin['Ad'])
    s('Bd = dt*Bc', lin['Bd'])
    s('c_k = dt*(f - Ac*x_bar - Bc*u_bar)', lin['c_k'])
    print('  Equilibrium check: Ad*x_bar + Bd*u_bar + c_k == x_bar?')
    lhs = lin['Ad'] @ lin['x_bar'] + lin['Bd'] @ lin['u_bar'] + lin['c_k']
    s('  lhs', lhs)
    s('  x_bar', lin['x_bar'])
    s('  error', np.max(np.abs(lhs - lin['x_bar'])))

    print(f'\n{"="*60}')


# ---------------------------------------------------------------------------
# Direct operating-point inspector (bypasses reference trajectory)
# ---------------------------------------------------------------------------

def inspect_at_point(arm, x_bar, u_bar, dt=0.1):
    """
    Evaluate linearisation at an arbitrary (x_bar, u_bar) — e.g. a moving arm.

    This bypasses MPCBuilder's reference trajectory (which always has dq=0)
    and lets you evaluate at any x_bar including non-zero velocity.

    Returns the same dict structure as inspect_linearisation().
    """
    import numpy as np

    q_bar  = x_bar[:arm.nq]
    dq_bar = x_bar[arm.nq:]
    th1, th2 = q_bar[0], q_bar[1]
    m1, m2, l1, l2, g = arm.m1, arm.m2, arm.l1, arm.l2, arm.g

    # CasADi evaluations
    f_val = np.array(arm.f_fun(x_bar, u_bar)).flatten()
    Ac    = np.array(arm.A_fun(x_bar, u_bar))
    Bc    = np.array(arm.B_fun(x_bar, u_bar))

    Ad  = np.eye(arm.nx) + dt * Ac
    Bd  = dt * Bc
    c_k = dt * (f_val - Ac @ x_bar - Bc @ u_bar)

    M_num = np.array(arm.M_fun(q_bar))
    try:
        M_inv = np.linalg.inv(M_num)
    except np.linalg.LinAlgError:
        M_inv = None

    def G_func(q):
        t1, t2 = q[0], q[1]
        _G1 = (m1*l1/2 + m2*l1)*g*np.sin(t1) + m2*l2/2*g*np.sin(t1+t2)
        _G2 = m2*l2/2*g*np.sin(t1+t2)
        return np.array([_G1, _G2])

    eps = 1e-6
    dG_dq_num = np.zeros((arm.nq, arm.nq))
    for j in range(arm.nq):
        eq = q_bar.copy(); eq[j] += eps
        dG_dq_num[:, j] = (G_func(eq) - G_func(q_bar)) / eps

    dG11 = (m1*l1/2 + m2*l1)*g*np.cos(th1) + m2*l2/2*g*np.cos(th1+th2)
    dG12 = m2*l2/2*g*np.cos(th1+th2)
    dG_dq_analytic = np.array([[dG11, dG12], [dG12, dG12]])

    return {
        'x_bar':          x_bar,
        'u_bar':          u_bar,
        'G_num':          G_func(q_bar),
        'f_val':          f_val,
        'Ac':             Ac,
        'Bc':             Bc,
        'Ad':             Ad,
        'Bd':             Bd,
        'c_k':            c_k,
        'M_num':          M_num,
        'M_inv':          M_inv,
        'dG_dq_num':      dG_dq_num,
        'dG_dq_analytic': dG_dq_analytic,
        'M_inv_times_dG': None if M_inv is None else M_inv @ dG_dq_analytic,
    }


def print_case_e_full(dt=0.1):
    """
    Print the full Case E (moving arm) linearisation using inspect_at_point,
    which correctly captures non-zero Coriolis in Ac.
    """
    arm = Arm2DOF(g=9.81)
    q_bar  = np.array([0.0, np.pi / 4])
    dq_bar = np.array([1.0, -0.5])
    x_bar  = np.concatenate([q_bar, dq_bar])

    # Equilibrium torque: u_bar = C(q,dq)*dq + G(q)  (holds ddq=0 at current velocity)
    m1, m2, l1, l2, g = arm.m1, arm.m2, arm.l1, arm.l2, arm.g
    th1, th2, dth1, dth2 = x_bar
    G1 = (m1*l1/2 + m2*l1)*g*np.sin(th1) + m2*l2/2*g*np.sin(th1+th2)
    G2 = m2*l2/2*g*np.sin(th1+th2)
    h  = -m2*l1*l2*np.sin(th2)
    Cdq1 = h*dth2*dth1 + h*(dth1+dth2)*dth2
    Cdq2 = -h*dth1**2
    u_bar = np.array([Cdq1 + G1, Cdq2 + G2])

    res = inspect_at_point(arm, x_bar, u_bar, dt=dt)

    print(f'\n{"="*60}')
    print(f'  Case E — Moving Arm (Full Jacobian with Coriolis)')
    print(f'  q=[0, π/4], dq=[1, -0.5], g=9.81, dt={dt}')
    print(f'{"="*60}')

    def s(label, val):
        if isinstance(val, np.ndarray):
            if val.ndim == 1:
                print(f'  {label} = {np.round(val, 6).tolist()}')
            else:
                print(f'  {label} =')
                for row in val:
                    print(f'    {np.round(row, 6).tolist()}')
        else:
            print(f'  {label} = {val}')

    s('x_bar',   res['x_bar'])
    s('u_bar',   res['u_bar'])
    s('G(q_bar)', res['G_num'])
    s('f_val (non-zero because dq≠0)', res['f_val'])
    s('M',       res['M_num'])
    s('M_inv',   res['M_inv'])
    s('∂G/∂q',  res['dG_dq_analytic'])
    s('Ac (full with Coriolis blocks)', res['Ac'])
    s('Bc', res['Bc'])
    s('Ad', res['Ad'])
    s('Bd', res['Bd'])
    s('c_k', res['c_k'])

    print(f'\n  NOTE: Ac lower-right = -M⁻¹·∂(Cq̇)/∂q̇ ≠ 0 because dq≠0')
    print(f'  (The code linearizes around x_ref_traj which has dq=0,')
    print(f'   so this block is always 0 in MPCBuilder.build_qp.)')
    print(f'{"="*60}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='QP Inspector for 2DOF MPC')
    parser.add_argument('--case', type=str, default=None,
                        help='Case name (A/B/C/D/E/A_N2/B_N2/C_tight) or "all"')
    parser.add_argument('--linearise', action='store_true',
                        help='Print step-by-step linearisation for --case')
    parser.add_argument('--matrices', action='store_true',
                        help='Print full QP matrices for --case')
    parser.add_argument('--export', action='store_true',
                        help='Export case to NPZ')
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--N', type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.dt is not None:
        overrides['dt'] = args.dt
    if args.N is not None:
        overrides['N'] = args.N

    if args.case is None or args.case == 'all':
        run_all_cases()
    else:
        if args.linearise:
            print_linearisation_steps(args.case, **overrides)
        if args.matrices:
            r = build_case(args.case, **overrides)
            print_matrices(r['qp_matrices'], r['N'], case_name=args.case)
        if args.export:
            export_case_npz(args.case, **overrides)
        if not (args.linearise or args.matrices or args.export):
            run_all_cases(cases=[args.case])

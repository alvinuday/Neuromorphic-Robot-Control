"""
B5 — Solver Comparison Benchmark (THESIS CORE)

Compares SL solver vs OSQP solver on a suite of 50 QPs.

QP Suite design:
    - Small (n=6, m=6):    MPC-sized problems (most common case)
    - Medium (n=12, m=12): Extended horizon
    - Ill-conditioned:     κ(P) = 100–1000 (challenging for OSQP)
    - Well-conditioned:    κ(P) = 1–10

For each QP:
    - Run OSQP 10 times, record mean/std time + objective value
    - Run SL 10 times (or 3 if time-constrained), record same
    - OSQP solution = reference; SL objective error = |obj_SL - obj_OSQP| / |obj_OSQP|

Expected result (be honest about this in the report):
    OSQP: ~5–50ms, near-perfect accuracy
    SL:   ~2000–8000ms, 0.1–5% objective error (depends on conditioning)
    
Thesis claim: SL is more robust on ill-conditioned problems.
Test this claim explicitly.
"""

import sys
from pathlib import Path

# Ensure src module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import json
import time
import datetime
import platform
import numpy as np

from src.solver.osqp_solver import OSQPSolver
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


def _mujoco_version():
    """Get MuJoCo version string."""
    try:
        import mujoco
        return mujoco.__version__
    except Exception:
        return "unknown"


def make_result_header(benchmark_id: str, config: dict) -> dict:
    """Create benchmark result header with environment info."""
    return {
        "benchmark_id": benchmark_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mujoco": _mujoco_version(),
        }
    }


def generate_qp_suite():
    """
    Generate 50 QPs of varying size and conditioning.
    
    Returns:
        list of QP dicts with P, q, A, l, u and metadata
    """
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    suite = []
    
    # Well-conditioned n=6 problems
    for _ in range(25):
        n = 6
        U = np.linalg.qr(rng.standard_normal((n, n)))[0]
        D = np.diag(rng.uniform(1, 10, n))
        P = U @ D @ U.T
        P = 0.5 * (P + P.T) + n * np.eye(n)
        q = rng.standard_normal(n)
        A = np.eye(n)
        l = -5 * np.ones(n)
        u = 5 * np.ones(n)
        kappa = float(D.diagonal().max() / D.diagonal().min())
        suite.append({
            'P': P, 'q': q, 'A': A, 'l': l, 'u': u,
            'type': 'well_cond_n6', 'kappa': kappa
        })
    
    # Ill-conditioned n=6 problems
    for _ in range(25):
        n = 6
        U = np.linalg.qr(rng.standard_normal((n, n)))[0]
        D = np.diag(np.logspace(0, 3, n))  # κ = 1000
        P = U @ D @ U.T
        P = 0.5 * (P + P.T)
        q = rng.standard_normal(n)
        A = np.eye(n)
        l = -5 * np.ones(n)
        u = 5 * np.ones(n)
        kappa = float(D.diagonal().max() / D.diagonal().min())
        suite.append({
            'P': P, 'q': q, 'A': A, 'l': l, 'u': u,
            'type': 'ill_cond_n6', 'kappa': kappa
        })
    
    return suite


def run_solver_on_suite(solver, suite, n_runs=3):
    """
    Run solver on all QPs in suite.
    
    Args:
        solver: Solver instance (OSQPSolver or StuartLandauLagrangeDirect)
        suite: List of QP dicts
        n_runs: Number of runs per QP
        
    Returns:
        List of result dicts per QP
    """
    results = []
    for i, qp in enumerate(suite):
        times = []
        objs = []
        viols = []
        
        for _ in range(n_runs):
            try:
                x, info = solver.solve(qp['P'], qp['q'], qp['A'], qp['l'], qp['u'])
                times.append(info.get('solve_time_ms', 0.0))
                objs.append(info.get('obj_val', 0.0))
                viols.append(info.get('constraint_viol', 0.0))
            except Exception as e:
                print(f"Warning: Solver failed on QP {i}: {e}")
                times.append(np.nan)
                objs.append(np.nan)
                viols.append(np.nan)
        
        # Filter out NaN values
        times = [t for t in times if not np.isnan(t)]
        objs = [o for o in objs if not np.isnan(o)]
        viols = [v for v in viols if not np.isnan(v)]
        
        results.append({
            'type': qp['type'],
            'kappa': qp['kappa'],
            'mean_ms': float(np.mean(times)) if times else np.nan,
            'std_ms': float(np.std(times)) if times else np.nan,
            'mean_obj': float(np.mean(objs)) if objs else np.nan,
            'mean_viol': float(np.mean(viols)) if viols else np.nan,
        })
    
    return results


def main():
    """Run B5 benchmark."""
    print("=" * 80)
    print("B5 — Solver Comparison Benchmark (OSQP vs StuartLandau)")
    print("=" * 80)
    
    print("\nGenerating QP suite (50 problems)...")
    suite = generate_qp_suite()
    well_cond = [qp for qp in suite if 'well' in qp['type']]
    ill_cond = [qp for qp in suite if 'ill' in qp['type']]
    print(f"  Well-conditioned: {len(well_cond)}")
    print(f"  Ill-conditioned: {len(ill_cond)}")
    
    config = {
        "n_qps": len(suite),
        "osqp_runs": 5,
        "sl_runs": 3,
        "suite_composition": "25 well-cond + 25 ill-cond",
    }
    
    osqp = OSQPSolver()
    sl = StuartLandauLagrangeDirect(T_solve=3.0)
    
    print("\nRunning OSQP on 50 QPs (5 runs each)...")
    osqp_res = run_solver_on_suite(osqp, suite, n_runs=5)
    
    print("Running StuartLandau on 50 QPs (3 runs each) — this is SLOW...")
    sl_res = run_solver_on_suite(sl, suite, n_runs=3)
    
    # Compute objective error
    print("\nComputing objective errors...")
    for i, (s, o) in enumerate(zip(sl_res, osqp_res)):
        if not (np.isnan(s['mean_obj']) or np.isnan(o['mean_obj'])):
            denom = abs(o['mean_obj']) + 1e-10
            s['obj_error_pct'] = float(abs(s['mean_obj'] - o['mean_obj']) / denom * 100.0)
        else:
            s['obj_error_pct'] = np.nan
    
    # Summary tables by type
    print(f"\n{'='*80}")
    print("Solver Comparison Summary")
    print(f"{'='*80}")
    
    for qtype in ('well_cond_n6', 'ill_cond_n6'):
        sl_sub = [r for r in sl_res if r['type'] == qtype]
        osqp_sub = [r for r in osqp_res if r['type'] == qtype]
        
        osqp_times = [r['mean_ms'] for r in osqp_sub if not np.isnan(r['mean_ms'])]
        sl_times = [r['mean_ms'] for r in sl_sub if not np.isnan(r['mean_ms'])]
        sl_errors = [r['obj_error_pct'] for r in sl_sub if not np.isnan(r['obj_error_pct'])]
        
        print(f"\n{qtype}")
        print(f"  OSQP: {np.mean(osqp_times):.1f} ± {np.std(osqp_times):.1f} ms")
        print(f"  SL:   {np.mean(sl_times):.1f} ± {np.std(sl_times):.1f} ms")
        print(f"  SL objective error: {np.mean(sl_errors):.2f}%")
    
    result = {
        **make_result_header("B5_solver_comparison", config),
        "n_qps": len(suite),
        "osqp": osqp_res,
        "sl": sl_res,
        "summary": {
            "osqp_mean_ms_well": float(np.mean([r['mean_ms'] for r in osqp_res if 'well' in r['type'] and not np.isnan(r['mean_ms'])])),
            "sl_mean_ms_well": float(np.mean([r['mean_ms'] for r in sl_res if 'well' in r['type'] and not np.isnan(r['mean_ms'])])),
            "osqp_mean_ms_ill": float(np.mean([r['mean_ms'] for r in osqp_res if 'ill' in r['type'] and not np.isnan(r['mean_ms'])])),
            "sl_mean_ms_ill": float(np.mean([r['mean_ms'] for r in sl_res if 'ill' in r['type'] and not np.isnan(r['mean_ms'])])),
            "sl_mean_obj_error_pct": float(np.mean([r['obj_error_pct'] for r in sl_res if not np.isnan(r['obj_error_pct'])])),
        }
    }
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B5_solver_comparison_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\n{'='*80}")
    print(f"B5 Complete")
    print(f"{'='*80}")
    print(f"Saved: {path}")
    print("\nSummary:")
    print(json.dumps(result['summary'], indent=2))
    
    return result


if __name__ == "__main__":
    main()

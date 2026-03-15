#!/usr/bin/env python
"""Verify Phase 9 gate compliance."""
import json
from pathlib import Path

results_dir = Path('evaluation/results')
benchmarks = ['B1_mpc_solo', 'B2_vla_solo', 'B3_dual_system', 'B4_sensor_ablation', 'B5_solver_comparison']

print('=' * 80)
print('PHASE 9 GATE COMPLIANCE VERIFICATION')
print('=' * 80)

all_pass = True

for b in benchmarks:
    files = list(results_dir.glob(f'{b}*.json'))
    if not files:
        print(f'\n❌ {b}: NO JSON FILE')
        all_pass = False
        continue
    
    latest = sorted(files)[-1]
    with open(latest) as f:
        data = json.load(f)
    
    benchmark_id = data.get('benchmark_id', 'N/A')
    
    # Check for non-zero metrics
    has_metrics = False
    if b in ['B1_mpc_solo', 'B2_vla_solo', 'B3_dual_system']:
        if 'results' in data and isinstance(data['results'], dict):
            if 'rmse' in data['results']:
                rmse = data['results']['rmse']
                has_metrics = rmse > 0
                print(f'\n✅ {b}')
                print(f'   File: {latest.name} ({latest.stat().st_size / 1024:.1f} KB)')
                print(f'   Mean RMSE: {rmse:.4f} rad')
            elif 'episodes' in data['results']:
                ep = data['results']['episodes'][0]
                if 'rmse' in ep:
                    rmse = ep['rmse']
                    has_metrics = rmse > 0
                    print(f'\n✅ {b}')
                    print(f'   File: {latest.name} ({latest.stat().st_size / 1024:.1f} KB)')
                    print(f'   Sample Episode RMSE: {rmse:.4f} rad')
    
    elif b == 'B4_sensor_ablation':
        modes = data.get('results', {})
        if modes:
            has_metrics = True
            print(f'\n✅ {b}')
            print(f'   File: {latest.name} ({latest.stat().st_size / 1024:.1f} KB)')
            print(f'   Modes tested: {len(modes)}')
            for mode_name in list(modes.keys())[:5]:
                mode = modes[mode_name]
                rmse = mode.get('mean_rmse', 'N/A')
                print(f'     - {mode_name}: RMSE={rmse}')
    
    elif b == 'B5_solver_comparison':
        summary = data.get('summary', [])
        if summary:
            has_metrics = True
            print(f'\n✅ {b}')
            print(f'   File: {latest.name} ({latest.stat().st_size / 1024:.1f} KB)')
            print(f'   QP problems tested: {len(summary)}')
            
            # Show well-conditioned stats
            well_cond = [s for s in summary if s.get('conditioning') == 'well-conditioned']
            if well_cond:
                avg_osqp = sum(s.get('osqp_time_ms', 0) for s in well_cond) / len(well_cond)
                avg_sl = sum(s.get('sl_time_ms', 0) for s in well_cond) / len(well_cond)
                print(f'   Well-conditioned ({len(well_cond)} problems):')
                print(f'     OSQP avg: {avg_osqp:.2f} ms')
                print(f'     SL avg: {avg_sl:.2f} ms')
    
    if not has_metrics:
        print(f'\n⚠️  {b}: No metrics found')
        all_pass = False

print('\n' + '=' * 80)
if all_pass:
    print('✅ PHASE 9 GATE: PASSED - All 5 benchmarks valid')
else:
    print('❌ PHASE 9 GATE: FAILED - Some benchmarks missing')
print('=' * 80)

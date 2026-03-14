#!/usr/bin/env python3
"""
Phase 12: Quick Benchmark Execution Wrapper
Runs B1-B5 benchmarks with controlled episode counts
"""

import asyncio
import sys
import json
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import benchmark runner
from evaluation.benchmarks.run_b1_b5_comprehensive import (
    BenchmarkRunner, 
    log_and_print
)

async def run_phase12_benchmarks():
    """Execute Phase 12 benchmarks with reasonable episode counts"""
    
    print("\n" + "="*80)
    print("PHASE 12: BENCHMARKING & VALIDATION STAGE 1")
    print("="*80)
    print(f"Timestamp: {Path('evaluation/results').resolve()}")
    
    # Create results directory
    results_dir = project_root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")
    
    # Initialize runner
    try:
        runner = BenchmarkRunner(
            dataset_id="lerobot/utokyo_xarm_pick_and_place",  # Use correct dataset
            results_dir=results_dir,
            max_episode_steps=500,
        )
        log_and_print("[INIT] ✓ BenchmarkRunner initialized successfully")
    except Exception as e:
        log_and_print(f"[INIT] ✗ Failed to initialize runner: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    all_results = {}
    
    # ─────────────────────────────────────────────────────────────────────────────
    # B1: Dataset Replay with MPC Solo (3 episodes for quick test)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    log_and_print("STARTING B1: Dataset Replay with MPC Solo")
    print("="*80 + "\n")
    
    try:
        b1_results = runner.run_b1_dataset_replay_mpc_solo(n_episodes=3)
        all_results["B1"] = b1_results
        
        b1_file = results_dir / "B1_dataset_replay_mpc_solo.json"
        b1_results.save(b1_file)
        log_and_print(f"\n[B1] ✓ Complete - Results saved to: {b1_file.name}")
        log_and_print(f"[B1] Success Rate: {b1_results.success_rate*100:.1f}%")
        log_and_print(f"[B1] Mean Tracking Error: {b1_results.mean_tracking_error:.6f} rad")
        
    except Exception as e:
        log_and_print(f"\n[B1] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # B2: VLA Prediction Accuracy (3 episodes)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    log_and_print("STARTING B2: VLA Prediction Accuracy")
    print("="*80 + "\n")
    
    try:
        b2_results = await runner.run_b2_vla_prediction_accuracy(n_episodes=3)
        all_results["B2"] = b2_results
        
        b2_file = results_dir / "B2_vla_prediction_accuracy.json"
        b2_results.save(b2_file)
        log_and_print(f"\n[B2] ✓ Complete - Results saved to: {b2_file.name}")
        log_and_print(f"[B2] Queries: {sum(e.vla_queries for e in b2_results.episodes)}")
        log_and_print(f"[B2] Mean VLA Latency: {b2_results.mean_vla_latency_ms:.1f} ms")
        
    except Exception as e:
        log_and_print(f"\n[B2] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # B3: Full Dual-System (3 episodes)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    log_and_print("STARTING B3: Full Dual-System (VLA + MPC)")
    print("="*80 + "\n")
    
    try:
        b3_results = await runner.run_b3_full_dual_system(n_episodes=3)
        all_results["B3"] = b3_results
        
        b3_file = results_dir / "B3_full_dual_system.json"
        b3_results.save(b3_file)
        log_and_print(f"\n[B3] ✓ Complete - Results saved to: {b3_file.name}")
        log_and_print(f"[B3] Success Rate: {b3_results.success_rate*100:.1f}%")
        log_and_print(f"[B3] Mean Tracking Error: {b3_results.mean_tracking_error:.6f} rad")
        
    except Exception as e:
        log_and_print(f"\n[B3] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # B4: MPC-Only Baseline (2 episodes)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    log_and_print("STARTING B4: MPC-Only Baseline")
    print("="*80 + "\n")
    
    try:
        b4_results = runner.run_b4_mpc_only_baseline(n_episodes=2)
        all_results["B4"] = b4_results
        
        b4_file = results_dir / "B4_mpc_only_baseline.json"
        b4_results.save(b4_file)
        log_and_print(f"\n[B4] ✓ Complete - Results saved to: {b4_file.name}")
        log_and_print(f"[B4] Success Rate: {b4_results.success_rate*100:.1f}%")
        
    except Exception as e:
        log_and_print(f"\n[B4] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("="*80 + "\n")
    
    for name, results in all_results.items():
        summary = results.summary()
        print(f"[{name}] {results.name}")
        print(f"  Episodes:           {summary['n_episodes']}")
        print(f"  Success Rate:       {summary['success_rate']*100:.1f}%")
        print(f"  Tracking Error:     {summary['mean_tracking_error_rad']:.6f} rad")
        print(f"  VLA Latency:        {summary['mean_vla_latency_ms']:.1f} ms")
        print(f"  Duration:           {summary['mean_duration_s']:.2f} s\n")
    
    print("="*80)
    print(f"✓ Phase 12 Stage 1: Benchmark Execution Complete")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(run_phase12_benchmarks())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

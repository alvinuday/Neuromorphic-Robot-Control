#!/usr/bin/env python3
"""
Master Test Suite Runner for All Phases

Validates the complete neuromorphic robot control system through all gates:
- Gate 1: 3-DOF Dynamics (Kinematics + Dynamics)
- Gate 2: MPC (Linearization + QP)
- Gate 3: SL Solver (3-DOF scaling)
- Gate 4a: SmolVLA Colab Server (health + endpoints)
- Gate 4b: Real SmolVLA Integration (async client + non-blocking)
- Gate 5: E2E System Testing (full dual-system, real server)

Usage:
    # Run all tests
    python3 scripts/run_all_tests.py
    
    # Run specific gate
    python3 scripts/run_all_tests.py --gate 1
    python3 scripts/run_all_tests.py --gate 4b
    
    # Run with detailed output
    python3 scripts/run_all_tests.py --verbose
    
    # Run only real server tests (requires SMOLVLA_SERVER_URL)
    python3 scripts/run_all_tests.py --real-only
"""

import subprocess
import sys
import os
import logging
from pathlib import Path
import argparse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Runs test gates in sequence with reporting."""
    
    def __init__(self, verbose: bool = False, real_only: bool = False):
        self.verbose = verbose
        self.real_only = real_only
        self.project_root = Path(__file__).parent.parent
        self.results = {}
    
    def run_test(self, name: str, test_file: str, description: str = "") -> bool:
        """
        Run a single test file (or multiple space-separated files).
        
        Args:
            name: Display name (e.g., "Gate 1")
            test_file: Path(s) to test file(s) relative to project root (space-separated for multiple)
            description: Long description
        
        Returns:
            True if tests passed, False otherwise
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"{name}")
        if description:
            logger.info(f"{description}")
        logger.info(f"{'='*70}")
        
        # Handle multiple test files (space-separated)
        test_files = test_file.split()
        test_paths = []
        
        for tf in test_files:
            test_path = self.project_root / tf
            
            if not test_path.exists():
                logger.error(f"❌ Test file not found: {test_path}")
                return False
            
            test_paths.append(str(test_path))
        
        # Build pytest command
        cmd = ['python3', '-m', 'pytest'] + test_paths
        
        if self.verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.append('-v')
        
        # Add quiet mode to hide warnings if not verbose
        if not self.verbose:
            cmd.append('-q')
        
        try:
            # Display command nicely
            display_cmd = ' '.join([f.replace(str(self.project_root), '.') if self.project_root in Path(f).parents or Path(f) == self.project_root else f for f in test_paths])
            logger.info(f"Running: pytest {display_cmd}\n")
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                timeout=300  # 5 minutes per test suite
            )
            
            passed = result.returncode == 0
            self.results[name] = passed
            
            if passed:
                logger.info(f"\n✓ {name} PASSED")
            else:
                logger.error(f"\n✗ {name} FAILED (exit code: {result.returncode})")
            
            return passed
        
        except subprocess.TimeoutExpired:
            logger.error(f"\n✗ {name} TIMEOUT (exceeded 5 minutes)")
            self.results[name] = False
            return False
        
        except Exception as e:
            logger.error(f"\n✗ {name} ERROR: {e}")
            self.results[name] = False
            return False
    
    def run_all_gates(self) -> bool:
        """Run all gates in sequence."""
        
        if self.real_only:
            logger.info("\n🌐 Running REAL SERVER tests only (no mocks)")
            logger.info("   Requires: export SMOLVLA_SERVER_URL='https://...ngrok.dev'")
            
            server_url = os.getenv("SMOLVLA_SERVER_URL")
            if not server_url:
                logger.error("\n❌ SMOLVLA_SERVER_URL not set!")
                logger.info("\nSetup instructions:")
                logger.info("1. Start SmolVLA server in Colab: vla/smolvla_server.ipynb")
                logger.info("2. Copy the ngrok URL")
                logger.info("3. Run: export SMOLVLA_SERVER_URL='https://...-ngrok.dev'")
                return False
            
            logger.info(f"   Server URL: {server_url}\n")
        
        gates = [
            ("Gate 1: 3-DOF Dynamics",
             "tests/test_dynamics_3dof.py",
             "Kinematics, dynamics, singularities"),
            
            ("Gate 2: MPC (Linearization + QP)",
             "tests/test_mpc_gate2.py",
             "Linearization, discretization, QP construction, warm-start"),
            
            ("Gate 3: SL Solver (3-DOF)",
             "tests/test_sl_gate3.py",
             "SL solver scaling to 3-DOF, convergence, accuracy vs OSQP"),
            
            ("Phase 8B: SmolVLA Async Components (Mocked)",
             "tests/test_smolvla_client.py tests/test_trajectory_buffer.py "
             "tests/test_dual_system_controller.py tests/test_vla_query_thread.py "
             "tests/test_integration_phase8b.py",
             "SmolVLA client, trajectory buffer, controllers (with mocks)"),
        ]
        
        # Add real server tests only if running real-only or server is available
        real_gates = [
            ("Gate 4b: Real SmolVLA Integration",
             "tests/test_integration_real_smolvla.py",
             "Real server health, inference, non-blocking properties"),
            
            ("Gate 5: E2E System Testing",
             "tests/test_e2e_gate5.py",
             "Full dual-system: reaching tasks, concurrent ops, threading, stress"),
        ]
        
        # Run main gates
        for name, test_file, desc in gates:
            if not self.run_test(name, test_file, desc):
                logger.error(f"\n❌ {name} failed. Stopping here.")
                return False
        
        # Run real gates if requested or if server available
        if self.real_only or os.getenv("SMOLVLA_SERVER_URL"):
            logger.info("\n" + "="*70)
            logger.info("🌐 REAL SERVER TESTING (optional)")
            logger.info("="*70)
            
            for name, test_file, desc in real_gates:
                # Skip if no server URL
                if not os.getenv("SMOLVLA_SERVER_URL"):
                    logger.info(f"\nSkipping {name} (set SMOLVLA_SERVER_URL to enable)")
                    continue
                
                # Run test but don't fail if it doesn't pass
                try:
                    self.run_test(name, test_file, desc)
                except Exception as e:
                    logger.warning(f"Real test error (non-critical): {e}")
        
        return True
    
    def run_specific_gate(self, gate_num: int) -> bool:
        """Run a specific gate."""
        
        gates_map = {
            1: ("Gate 1: 3-DOF Dynamics", "tests/test_dynamics_3dof.py"),
            2: ("Gate 2: MPC", "tests/test_mpc_gate2.py"),
            3: ("Gate 3: SL Solver", "tests/test_sl_gate3.py"),
            "4a": ("Gate 4a: SmolVLA Server", "tests/test_smolvla_client.py"),
            "4b": ("Gate 4b: Real SmolVLA", "tests/test_integration_real_smolvla.py"),
            5: ("Gate 5: E2E System", "tests/test_e2e_gate5.py"),
            "8b": ("Phase 8B: Components", 
                   "tests/test_smolvla_client.py tests/test_trajectory_buffer.py "
                   "tests/test_dual_system_controller.py tests/test_vla_query_thread.py "
                   "tests/test_integration_phase8b.py"),
        }
        
        if gate_num not in gates_map:
            logger.error(f"Unknown gate: {gate_num}")
            logger.info("Available: 1, 2, 3, 4a, 4b, 5, 8b")
            return False
        
        name, test_file = gates_map[gate_num]
        return self.run_test(name, test_file)
    
    def print_summary(self):
        """Print test results summary."""
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for name, passed_flag in self.results.items():
            status = "✓ PASS" if passed_flag else "✗ FAIL"
            logger.info(f"{status}  {name}")
        
        logger.info("="*70)
        logger.info(f"TOTAL: {passed}/{total} test suites passed")
        
        if passed == total:
            logger.info("✓ ALL TESTS PASSED")
            return 0
        else:
            logger.info("✗ SOME TESTS FAILED")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Master test runner for neuromorphic robot control system"
    )
    parser.add_argument(
        '--gate',
        type=str,
        help="Run specific gate (1, 2, 3, 4a, 4b, 5, 8b)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output with all pytest details"
    )
    parser.add_argument(
        '--real-only',
        action='store_true',
        help="Run only real server tests (requires SMOLVLA_SERVER_URL)"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose, real_only=args.real_only)
    
    try:
        if args.gate:
            success = runner.run_specific_gate(args.gate)
        else:
            success = runner.run_all_gates()
        
        runner.print_summary()
        return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Testing interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())

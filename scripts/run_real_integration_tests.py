#!/usr/bin/env python3
"""
SmolVLA Real Integration Test Runner

This script:
1. Takes the ngrok URL from Colab as input
2. Sets the environment variable
3. Runs Gate 4b + 5 tests with real server
4. Reports comprehensive results

Usage:
    python3 scripts/run_real_integration_tests.py "https://xxxx-xxxx-ngrok-free.dev"
    
Or interactively:
    python3 scripts/run_real_integration_tests.py
    (will prompt for URL)
"""

import subprocess
import sys
import os
import logging
from pathlib import Path
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def validate_url(url: str) -> bool:
    """Validate ngrok URL format."""
    return (url.startswith('https://') and 
            'ngrok' in url and 
            url.endswith('.dev'))


def health_check(url: str) -> bool:
    """Quick health check to verify server is reachable."""
    import requests
    try:
        logger.info(f"🔍 Testing server connectivity...")
        response = requests.get(f"{url}/health", timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Server is healthy: {data}")
            return True
        else:
            logger.error(f"❌ Server returned {response.status_code}")
            return False
    
    except requests.exceptions.Timeout:
        logger.error("❌ Server health check timed out (3s)")
        return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


def run_tests(server_url: str) -> bool:
    """Run Gate 4b and 5 tests with the server URL."""
    
    # Set environment variable
    os.environ['SMOLVLA_SERVER_URL'] = server_url
    logger.info(f"\n✓ Environment variable set: SMOLVLA_SERVER_URL={server_url}\n")
    
    project_root = Path(__file__).parent.parent
    
    tests = [
        ("Gate 4b: Real SmolVLA Integration", 
         "tests/test_integration_real_smolvla.py",
         13),
        ("Gate 5: E2E System Testing",
         "tests/test_e2e_gate5.py",
         12),
    ]
    
    results = {}
    total_tests = 0
    total_passed = 0
    
    logger.info("="*70)
    logger.info("RUNNING REAL INTEGRATION TESTS")
    logger.info("="*70)
    
    for gate_name, test_file, expected_count in tests:
        logger.info(f"\n🧪 {gate_name}")
        logger.info(f"   Running: {test_file}")
        logger.info(f"   Expected: {expected_count} tests")
        logger.info("-" * 70)
        
        test_path = project_root / test_file
        
        if not test_path.exists():
            logger.error(f"❌ Test file not found: {test_path}")
            results[gate_name] = False
            continue
        
        # Run pytest
        cmd = [
            'python3', '-m', 'pytest',
            str(test_path),
            '-v',
            '--tb=short'
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes per test  suite
            )
            
            elapsed = time.time() - start_time
            
            # Parse output
            output = result.stdout + result.stderr
            
            # Look for test summary
            if 'passed' in output:
                # Extract numbers
                import re
                match = re.search(r'(\d+)\s+passed', output)
                if match:
                    passed_count = int(match.group(1))
                    total_tests += expected_count
                    total_passed += passed_count
                    
                    status = "✅ PASSED" if passed_count == expected_count else "⚠️ PARTIAL"
                    logger.info(f"{status} - {passed_count}/{expected_count} tests passed in {elapsed:.1f}s")
                    results[gate_name] = (passed_count == expected_count)
                    
                    # Show any skipped
                    if 'skipped' in output:
                        skip_match = re.search(r'(\d+)\s+skipped', output)
                        if skip_match:
                            logger.info(f"   (Note: {skip_match.group(1)} tests skipped)")
                else:
                    logger.error(f"❌ Could not parse test results")
                    results[gate_name] = False
            else:
                logger.error(f"❌ Tests failed or didn't run")
                logger.error(output[-500:])  # Last 500 chars
                results[gate_name] = False
        
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Tests timed out (exceeded 5 minutes)")
            results[gate_name] = False
        except Exception as e:
            logger.error(f"❌ Error running tests: {e}")
            results[gate_name] = False
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for gate_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}  {gate_name}")
    
    logger.info("="*70)
    logger.info(f"TOTAL: {total_passed}/{total_tests} tests passed")
    
    if all(results.values()):
        logger.info("✅ ALL REAL INTEGRATION TESTS PASSED!")
        return True
    else:
        logger.error("⚠️ Some tests failed. Check output above.")
        return False


def main():
    """Main entry point."""
    
    # Get server URL from argument or prompt
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        logger.info("\n🔗 SmolVLA Server URL Required")
        logger.info("   Copy from Colab notebook Cell 21 output")
        logger.info("   Example: https://4a8c-2600-1700-xxxx-ngrok-free.dev\n")
        
        server_url = input("Enter ngrok URL: ").strip()
        
        if not server_url:
            logger.error("❌ No URL provided")
            return 1
    
    # Validate URL format
    if not validate_url(server_url):
        logger.error(f"❌ Invalid ngrok URL format: {server_url}")
        logger.error("   Expected format: https://xxxx-xxxx-ngrok-free.dev")
        return 1
    
    logger.info(f"🌐 Server URL: {server_url}\n")
    
    # Health check
    if not health_check(server_url):
        logger.error("\n❌ Server is not reachable. Check:")
        logger.error("   1. Colab notebook is still running")
        logger.error("   2. ngrok tunnel is active")
        logger.error("   3. URL is correct (copy from Cell 21)")
        return 1
    
    # Run tests
    logger.info("\n✅ Server is healthy. Starting tests...\n")
    time.sleep(1)
    
    success = run_tests(server_url)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

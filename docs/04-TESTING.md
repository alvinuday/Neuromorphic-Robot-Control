# Testing Guide

How to run tests and understand results.

## Quick Test Run

```bash
# Run all tests (takes ~15 seconds)
python3 -m pytest tests/ -v

# Run one specific test
python3 -m pytest tests/test_lagrange_direct.py::test_solve_simple_qp -v
```

## Test Suite Overview

| Test File | Tests | Purpose |
|-----------|-------|---------|
| `test_lagrange_direct.py` | 5 | Basic solver correctness |
| `test_phase3_kkt.py` | 3 | KKT optimality conditions |
| `test_benchmark_suite.py` | 7 | Solver benchmarking |
| `test_mujoco_integration.py` | 6 | arm dynamics & control |
| `test_integration_e2e.py` | 4 | End-to-end workflows |

**Total**: 25 tests, all passing ✅

## Understanding Test Output

### Passing Test
```
test_solve_simple_qp PASSED                                           [60%]
```
✅ Good - test passed

### Failing Test
```
test_solve_simple_qp FAILED                                           [60%]
AssertionError: Constraint violation too large: 1.5e-5 > 1e-6
```
❌ Bad - test failed with reason shown

### Running Specific Tests

```bash
# All tests in a file
python3 -m pytest tests/test_lagrange_direct.py -v

# Specific test function
python3 -m pytest tests/test_lagrange_direct.py::test_solver_initialization -v

# All tests matching pattern
python3 -m pytest tests/ -k "constraint" -v

# Show print statements
python3 -m pytest tests/test_benchmark_suite.py -v -s
```

## Test Categories

### Unit Tests (Fast)
Test individual components:
- Solver initialization
- Single QP solve
- Constraint checking

```bash
python3 -m pytest tests/test_lagrange_direct.py -v
```

### Integration Tests (Medium)
Test components together:
- Arm model loading
- Control closed-loop
- Benchmarking pipeline

```bash
python3 -m pytest tests/test_mujoco_integration.py -v
```

### System Tests (Full)
Test entire workflow:
- End-to-end control task
- Multiple solvers
- Full benchmarking suite

```bash
python3 -m pytest tests/test_integration_e2e.py -v
python3 -m pytest tests/test_benchmark_suite.py -v -s
```

## Interpreting Metrics

Tests check four main things:

### 1. Constraint Satisfaction
```python
# Example test
assert eq_violation < 1e-6, f"Eq violation {eq_violation} too large"
```
✅ **Good**: < 1e-6 (machine precision)  
⚠️ **Warn**: 1e-6 to 1e-4  
❌ **Bad**: > 1e-4

### 2. Optimality
```python
# How close to optimal solution
optimality_gap = (f_solver - f_optimal) / |f_optimal|
assert optimality_gap < 0.01  # Within 1%
```
✅ **Good**: < 1% gap  
⚠️ **Warn**: 1-5%  
❌ **Bad**: > 5%

### 3. Solve Time
```python
assert solve_time < 0.1  # Under 100ms
```
✅ **Good**: < 100ms (real-time @ 10Hz)  
⚠️ **Warn**: 100-500ms  
❌ **Bad**: > 500ms

### 4. Convergence
```python
assert num_iterations < 1000, "Too many iterations"
```
✅ **Good**: < 1000 ODE steps  
⚠️ **Warn**: 1000-5000  
❌ **Bad**: > 5000

## Example: Running Benchmark Tests

```bash
python3 -m pytest tests/test_benchmark_suite.py -v -s
```

Expected output:
```
TEST: OSQP Solver
[OSQP] ✓ Solver created
[OSQP] Solve time: 0.0082s
[OSQP] Constraint violation: 1.3e-7
[OSQP] ✓ PASSED

TEST: iLQR Solver
[iLQR] ✓ Solver created
[iLQR] Solve time: 0.0125s
[iLQR] ✓ PASSED

TEST: Neuromorphic Solver
[Neuro] ✓ Solver created
[Neuro] Solve time: 0.0984s
[Neuro] Constraint violation: 8.2e-7
[Neuro] ✓ PASSED
```

## Debugging a Failing Test

1. **Re-run with output**:
   ```bash
   python3 -m pytest tests/test_lagrange_direct.py::test_solve_simple_qp -v -s
   ```

2. **Check error message**: Scroll up to see the assertion that failed

3. **Run in Python directly** for more control:
   ```python
   python3
   >>> from tests.test_lagrange_direct import test_solve_simple_qp
   >>> test_solve_simple_qp()
   ```

4. **Check system dependencies**:
   ```bash
   python3 -c "import mujoco, osqp, scipy; print('✓ OK')"
   ```

---

## CI/CD Use

Full test suite for continuous integration:

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
```

---

**Next:** See [05-VISUALIZATION.md](05-VISUALIZATION.md) to understand the interactive viewer.

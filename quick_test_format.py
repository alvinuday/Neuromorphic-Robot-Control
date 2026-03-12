#!/usr/bin/env python3
"""Quick test to verify 6-tuple format support in SL solver."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from solver.stuart_landau_lagonn import StuartLandauLaGONN

# Simple 2x2 QP: minimize 0.5 ||x||^2 - [2, 3]^T x
# with no constraints (empty A_eq, A_ineq)

P = np.eye(2)
q = -np.array([2.0, 3.0])

# Empty equality constraints
A_eq = np.zeros((0, 2))
b_eq = np.zeros(0)

# Empty inequality constraints
A_ineq = np.zeros((0, 2))
k_ineq = np.zeros(0)

# Test the 6-tuple format
qp_6tuple = (P, q, A_eq, b_eq, A_ineq, k_ineq)

print("Testing 6-tuple format support...")
print(f"QP format: 6 elements")
print(f"P shape: {P.shape}")
print(f"q shape: {q.shape}")
print(f"A_eq shape: {A_eq.shape}")
print(f"b_eq shape: {b_eq.shape}")
print(f"A_ineq shape: {A_ineq.shape}")
print(f"k_ineq shape: {k_ineq.shape}")

try:
    solver = StuartLandauLaGONN(tau_x=1.0, mu_x=0.0, T_solve=20.0, convergence_tol=1e-4)
    x_star = solver.solve(qp_6tuple, verbose=True)
    print(f"\n✓ 6-tuple format accepted!")
    print(f"Solution: {x_star}")
    print(f"Expected: [2.0, 3.0]")
    print(f"Error: {np.linalg.norm(x_star - np.array([2.0, 3.0])):.2e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

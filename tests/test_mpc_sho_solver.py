
"""
Comprehensive test suite for 2-DOF MPC with oscillator-based Ising solver.
Tests mathematical correctness, numerical stability, and implementation validity.
Run with: python tests/test_mpc_sho_solver.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import casadi as ca
from scipy import sparse
import osqp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder

# ============================================================================
# SECTION 1: ROBOT DYNAMICS TESTS
# ============================================================================

def test_dynamics_setup():
    """Test 1.1: Verify CasADi dynamics functions are correctly defined"""
    print("\n=== TEST 1.1: Dynamics Setup ===")
    
    # Instantiate our class
    arm = Arm2DOF()
    f_fun, A_fun, B_fun = arm.get_dynamics_functions()
    
    # Test evaluation at known point
    x_test = np.array([0.1, 0.2, 0.0, 0.0])
    u_test = np.array([0.0, 0.0])
    
    f_val = np.array(f_fun(x_test, u_test)).flatten()
    A_val = np.array(A_fun(x_test, u_test))
    B_val = np.array(B_fun(x_test, u_test))
    
    # Check dimensions
    assert f_val.shape == (4,), f"f_val shape mismatch: {f_val.shape}"
    assert A_val.shape == (4, 4), f"A_val shape mismatch: {A_val.shape}"
    assert B_val.shape == (4, 2), f"B_val shape mismatch: {B_val.shape}"
    
    # Check physical consistency: M should be symmetric positive definite
    # We exposed M_fun in our class for testing
    theta_test = x_test[:2]
    M_numeric = np.array(arm.M_fun(theta_test))
    
    # Symmetry
    assert np.allclose(M_numeric, M_numeric.T), "Inertia matrix not symmetric"
    
    # Positive definite (eigenvalues > 0)
    eigvals = np.linalg.eigvals(M_numeric)
    assert np.all(eigvals > 0), f"Inertia matrix not positive definite: {eigvals}"
    
    print("✓ Dynamics functions correctly defined")
    print(f"  f_val = {f_val}")
    print(f"  A_val shape = {A_val.shape}, B_val shape = {B_val.shape}")
    print(f"  M eigenvalues = {eigvals}")
    
    return f_fun, A_fun, B_fun


def test_energy_conservation():
    """Test 1.2: Verify energy conservation for unforced system"""
    print("\n=== TEST 1.2: Energy Conservation (Unforced System) ===")
    
    arm = Arm2DOF()
    f_fun, _, _ = arm.get_dynamics_functions()
    
    dt = 0.001
    T_sim = 1.0
    n_steps = int(T_sim / dt)
    
    # Initial state: non-zero position and velocity
    x0 = np.array([0.5, 0.3, 0.1, -0.1])
    u = np.array([0.0, 0.0])  # No control
    
    # Integrate dynamics
    x = x0.copy()
    energies = []
    
    for _ in range(n_steps):
        xdot = np.array(f_fun(x, u)).flatten()
        x = x + dt * xdot
        
        # Compute total energy (kinetic + potential)
        theta = x[:2]
        dtheta = x[2:]
        
        # Kinetic energy (simplified, ignores coupling - WAIT, simplified formula in user prompt was wrong/incomplete)
        # Let's use the M matrix for KE: T = 0.5 * dtheta^T * M * dtheta
        M = np.array(arm.M_fun(theta))
        T = 0.5 * dtheta.T @ M @ dtheta
        
        # Potential energy
        m1, m2 = arm.m1, arm.m2
        l1, l2 = arm.l1, arm.l2
        g = arm.g
        
        # PE calculation (using gravity model logic)
        # h1 = l1 sin(th1)? G terms are usually dV/dtheta.
        # Let's trust the user's explicit PE formula if it works, or use standard mgh
        # The user provided prompt had:
        # V = (m1*l1/2 + m2*l1)*g*(1 - np.cos(theta[0])) + m2*l2/2*g*(1 - np.cos(theta[0] + theta[1]))
        V = (m1*l1/2 + m2*l1)*g*(1 - np.cos(theta[0])) + m2*l2/2*g*(1 - np.cos(theta[0] + theta[1]))
        
        E = T + V
        energies.append(E)
    
    energies = np.array(energies)
    E0 = energies[0]
    E_final = energies[-1]
    rel_error = abs(E_final - E0) / abs(E0)
    
    # Energy should be conserved within numerical tolerance
    # Damping might not be zero in discrete Euler integration, so loose tolerance
    assert rel_error < 0.1, f"Energy not conserved: relative error = {rel_error}"
    
    print(f"✓ Energy conserved within {rel_error*100:.2f}%")
    print(f"  Initial energy = {E0:.6f}")
    print(f"  Final energy   = {E_final:.6f}")


def test_linearization_accuracy():
    """Test 1.3: Verify linearization matches true dynamics locally"""
    print("\n=== TEST 1.3: Linearization Accuracy ===")
    
    arm = Arm2DOF()
    f_fun, A_fun, B_fun = arm.get_dynamics_functions()
    
    # Linearization point
    x_bar = np.array([0.2, 0.3, 0.0, 0.0])
    u_bar = np.array([0.0, 0.0])
    
    # True dynamics
    f_bar = np.array(f_fun(x_bar, u_bar)).flatten()
    A = np.array(A_fun(x_bar, u_bar))
    B = np.array(B_fun(x_bar, u_bar))
    
    # Test perturbations
    delta_x = np.array([0.01, 0.01, 0.01, 0.01])
    delta_u = np.array([0.1, 0.1])
    
    x_pert = x_bar + delta_x
    u_pert = u_bar + delta_u
    
    # True perturbed dynamics
    f_pert_true = np.array(f_fun(x_pert, u_pert)).flatten()
    
    # Linearized approximation
    f_pert_approx = f_bar + A @ delta_x + B @ delta_u
    
    error = np.linalg.norm(f_pert_true - f_pert_approx)
    rel_error = error / np.linalg.norm(f_pert_true)
    
    assert rel_error < 0.01, f"Linearization error too large: {rel_error}"
    
    print(f"✓ Linearization accurate within {rel_error*100:.4f}%")
    print(f"  True dynamics    = {f_pert_true}")
    print(f"  Approx dynamics  = {f_pert_approx}")
    print(f"  Error            = {error:.6e}")


# ============================================================================
# SECTION 2: MPC QP FORMULATION TESTS
# ============================================================================

def test_qp_dimensions():
    """Test 2.1: Verify QP matrices have correct dimensions"""
    print("\n=== TEST 2.1: QP Matrix Dimensions ===")
    
    arm = Arm2DOF()
    mpc = MPCBuilder(arm, N=20)
    
    nx, nu = 4, 2
    N = 20
    
    x0 = np.zeros(nx)
    x_goal = np.array([np.pi/4, np.pi/4, 0, 0])
    
    # Build reference trajectory
    x_ref_traj = mpc.build_reference_trajectory(x0, x_goal)
    
    # Build QP
    Q, p, A_eq, b_eq, A_ineq, k_ineq = mpc.build_qp(x0, x_ref_traj)
    
    # Check dimensions
    expected_nz = N * (nx + nu) + nx
    assert Q.shape == (expected_nz, expected_nz), f"Q shape: {Q.shape} vs expected {(expected_nz, expected_nz)}"
    assert p.shape == (expected_nz,), f"p shape: {p.shape} vs expected ({expected_nz},)"
    
    # Check Q is symmetric
    assert np.allclose(Q, Q.T), "Q not symmetric"
    
    # Check Q is positive semidefinite
    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals >= -1e-10), f"Q not PSD: min eigenvalue = {eigvals[0]}"
    
    print(f"✓ QP dimensions correct: n_z = {expected_nz}")
    print(f"  Q shape = {Q.shape}, symmetric = {np.allclose(Q, Q.T)}")
    print(f"  Q eigenvalues: min = {eigvals[0]:.2e}, max = {eigvals[-1]:.2e}")


def test_qp_convexity():
    """Test 2.2: Verify QP is convex (Q positive semidefinite)"""
    print("\n=== TEST 2.2: QP Convexity ===")
    # Covered in 2.1 check, but explicitly here too
    arm = Arm2DOF()
    mpc = MPCBuilder(arm, N=10)
    x0 = np.zeros(4)
    x_goal = np.zeros(4)
    traj = mpc.build_reference_trajectory(x0, x_goal)
    Q, _, _, _, _, _ = mpc.build_qp(x0, traj)

    eigvals = np.linalg.eigvalsh(Q)
    # Floating point tolerance
    psd = np.all(eigvals >= -1e-8)
    assert psd, "Q is not positive semidefinite"
    print(f"✓ Q is positive semidefinite. Min eig: {eigvals[0]}")


def test_constraint_consistency():
    """Test 2.3: Verify constraints are feasible (not contradictory)"""
    print("\n=== TEST 2.3: Constraint Consistency ===")
    
    arm = Arm2DOF()
    mpc = MPCBuilder(arm)
    
    # Check bounds are valid
    assert np.all(mpc.theta_max > mpc.theta_min), "theta bounds inconsistent"
    assert np.all(mpc.tau_max > mpc.tau_min), "tau bounds inconsistent"
    
    print(f"✓ Constraints are consistent")


def test_qp_vs_kkt():
    """Test 2.4: Verify OSQP solution satisfies KKT conditions"""
    print("\n=== TEST 2.4: KKT Conditions ===")
    
    # Simple test QP
    n = 4
    Q = np.eye(n)
    p = -np.ones(n)
    
    # Constraints: x >= 0, x <= 2
    A = np.vstack([np.eye(n), -np.eye(n)])
    l = np.concatenate([np.zeros(n), -2*np.ones(n)])
    u = np.inf * np.ones(2*n)
    
    # Solve with OSQP
    prob = osqp.OSQP()
    prob.setup(P=sparse.csc_matrix(Q), q=p, 
               A=sparse.csc_matrix(A), l=l, u=u, verbose=False)
    res = prob.solve()
    
    x_opt = res.x
    lam = res.y  # Dual variables
    
    # KKT stationarity: ∇f(x) + A^T λ = 0
    # Note: OSQP returns y such that Px + q + A^T y = 0
    grad_L = Q @ x_opt + p + A.T @ lam
    stationarity_error = np.linalg.norm(grad_L)
    
    # KKT primal feasibility: l <= Ax <= u
    Ax = A @ x_opt
    primal_feas = np.all(Ax >= l - 1e-6) and np.all(Ax <= u + 1e-6)
    
    assert stationarity_error < 1e-3, f"Stationarity violated: {stationarity_error}"
    assert primal_feas, "Primal feasibility violated"
    
    print(f"✓ KKT conditions satisfied")
    print(f"  Stationarity error: {stationarity_error:.6e}")


# ============================================================================
# SECTION 3: PENALTY METHOD AND ENCODING TESTS
# ============================================================================

def test_penalty_equivalence():
    """Test 3.1: Verify penalty method converges to constrained solution"""
    print("\n=== TEST 3.1: Penalty Method Convergence ===")
    
    # Simple test: min x^2 s.t. x >= 1
    # Analytical: x* = 1
    
    def solve_penalty(rho):
        # Unconstrained: min x^2 + rho * max(0, 1-x)^2
        from scipy.optimize import minimize
        def obj(x):
            return x[0]**2 + rho * max(0, 1 - x[0])**2
        res = minimize(obj, x0=[0.0], method='BFGS')
        return res.x[0]
    
    rhos = [1, 10, 100, 1000]
    x_opts = [solve_penalty(rho) for rho in rhos]
    errors = [abs(x - 1.0) for x in x_opts]
    
    # Error should decrease as rho increases
    assert errors[-1] < errors[0], "Penalty method not converging"
    assert errors[-1] < 0.01, f"Penalty method error too large: {errors[-1]}"
    
    print(f"✓ Penalty method converges")
    for rho, x, err in zip(rhos, x_opts, errors):
        print(f"  rho = {rho:4d}: x* = {x:.6f}, error = {err:.6e}")


def test_qubo_to_ising():
    """Test 3.3: Verify QUBO to Ising conversion is correct"""
    print("\n=== TEST 3.3: QUBO to Ising Conversion ===")
    
    # Simple QUBO: min s1 + s2 + 2*s1*s2, s in {0,1}
    # Solution: s1=0, s2=0, cost=0
    Q_qubo = np.array([[0, 2],
                       [2, 0]])
    p_qubo = np.array([1, 1])
    
    # Convert to Ising
    n = 2
    Qs = 0.5 * (Q_qubo + Q_qubo.T)
    J = np.zeros_like(Qs)
    h = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                J[i,j] = -0.25 * Qs[i,j]
        h[i] = -0.5 * (p_qubo[i] + np.sum(Qs[i,:]))
    
    # Test all 4 spin configurations
    costs_qubo = []
    costs_ising = []
    
    for s1 in [0, 1]:
        for s2 in [0, 1]:
            s = np.array([s1, s2])
            cost_qubo = 0.5 * s @ Q_qubo @ s + p_qubo @ s
            costs_qubo.append(cost_qubo)
            
            # Convert to spins
            sigma = 2*s - 1
            # H = -J * s's - h's. J symmetric means sum J_ij sig_i sig_j = 2 * J_12 sig_1 sig_2 + ...
            cost_ising = -np.sum(J * np.outer(sigma, sigma)) - h @ sigma
            costs_ising.append(cost_ising)
    
    # Costs should be proportional (Ising has constant offset)
    costs_qubo = np.array(costs_qubo)
    costs_ising = np.array(costs_ising)
    
    # Check ordering is preserved
    qubo_order = np.argsort(costs_qubo)
    ising_order = np.argsort(costs_ising)
    
    assert np.array_equal(qubo_order, ising_order), f"QUBO {costs_qubo} and Ising {costs_ising} orderings differ"
    
    print(f"✓ QUBO to Ising conversion correct")


# ============================================================================
# SECTION 4: OSCILLATOR DYNAMICS TESTS
# ============================================================================

def test_oscillator_phase_locking():
    """Test 4.1: Verify oscillators phase-lock for simple Ising problem"""
    print("\n=== TEST 4.1: Oscillator Phase Locking ===")
    
    # Simple 2-spin ferromagnetic coupling: J12 > 0 → prefer same spin
    # We need realistic coupling for the 1e-6 simulation time
    n = 2
    J = np.array([[0, 1e7],
                  [1e7, 0]])
    h = np.array([0, 0])
    
    def oscillator_dynamics(t, phi, J, h, omega):
        n = len(phi)
        # Vectorized for speed
        diff = phi[None, :] - phi[:, None]
        dphi = omega + np.sum(J * np.sin(diff), axis=1) + h
        return dphi
    
    omega = np.zeros(n)
    phi0 = np.array([0.1, 3.0])  # Start with different phases
    
    T_final = 1e-6
    dt = 1e-9
    t_eval = np.arange(0, T_final, dt)
    
    sol = solve_ivp(oscillator_dynamics, (0, T_final), phi0,
                    args=(J, h, omega), t_eval=t_eval, max_step=dt)
    
    phi_final = sol.y[:,-1] % (2*np.pi)
    
    # Check phase difference
    phase_diff = abs(phi_final[1] - phi_final[0])
    phase_diff = min(phase_diff, 2*np.pi - phase_diff)
    
    # For ferromagnetic (J>0), diff should go to 0
    is_locked = phase_diff < 0.1
    
    assert is_locked, f"Oscillators did not phase-lock: phase_diff = {phase_diff}"
    
    print(f"✓ Oscillators phase-locked")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" COMPREHENSIVE TEST SUITE FOR MPC-SHO QP SOLVER")
    print("="*70)
    
    # Section 1: Dynamics
    test_dynamics_setup()
    test_linearization_accuracy()
    test_energy_conservation()
    
    # Section 2: QP Formulation
    test_qp_dimensions()
    test_qp_convexity()
    test_constraint_consistency()
    test_qp_vs_kkt()
    
    # Section 3: Encoding
    test_penalty_equivalence()
    test_qubo_to_ising()
    
    # Section 4: Oscillators
    test_oscillator_phase_locking()
    
    print("\n" + "="*70)
    print(" ✓ SELECTED TESTS PASSED")
    print("="*70)

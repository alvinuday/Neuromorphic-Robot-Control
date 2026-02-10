
import numpy as np
from scipy.integrate import solve_ivp

class SHOSolver:
    def __init__(self, n_bits=16, rho=50.0, dt=2e-9, T_final=2e-6, max_alm_iter=5, coupling_strength=0.5):
        """
        n_bits: resolution (bits) per continuous variable
        rho: penalty parameter for constraints
        dt: time step for oscillator simulation (increased for stability)
        T_final: duration of oscillator simulation (increased for convergence)
        max_alm_iter: Augmented Lagrangian iterations
        coupling_strength: K in phase dynamics dphi/dt = omega + K * sum J_ij sin(phi_j - phi_i) + h
        """
        self.n_bits = n_bits
        self.rho = rho
        self.dt = dt
        self.T_final = T_final
        self.max_alm_iter = max_alm_iter
        self.coupling_strength = coupling_strength

    def solve(self, qp_matrices, x_min_val=-10.0, x_max_val=10.0, custom_bounds=None):
        """
        Solves QP using simulated Oscillator Ising Machine with Augmented Lagrangian.
        qp_matrices: (Q, p, A_eq, b_eq, A_ineq, k_ineq)
        """
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
        
        A = np.vstack([A_eq, A_ineq]) if A_ineq.size else A_eq
        b = np.concatenate([b_eq, k_ineq]) if A_ineq.size else b_eq
        
        n_z = Q.shape[0]
        n_m = A.shape[0]
        
        # Dual variables (Lagrange multipliers)
        lam = np.zeros(n_m)
        z_sol = np.zeros(n_z)
        
        if custom_bounds is not None:
            x_min = custom_bounds['min']
            x_max = custom_bounds['max']
        else:
            x_min = x_min_val * np.ones(n_z)
            x_max = x_max_val * np.ones(n_z)

        # Augmented Lagrangian Loop
        constraint_violation_history = []
        
        for i_alm in range(self.max_alm_iter):
            # min 0.5 z'Qz + p'z + lam'(Az - b) + 0.5 * rho * ||Az - b||^2
            # Linear term becomes: p + A'lam - rho A'b
            # Quadratic term: Q + rho A'A
            
            Q_c = Q + self.rho * (A.T @ A)
            p_c = p + A.T @ lam - self.rho * (A.T @ b)

            # 2. Continuous -> QUBO
            Q_qubo, p_qubo, C = self._continuous_to_qubo(Q_c, p_c, x_min, x_max)

            # 3. QUBO -> Ising
            J, h = self._qubo_to_ising(Q_qubo, p_qubo)

            # 4. Solve Oscillator Dynamics
            print(f"  SHO: ALM Iter {i_alm+1}/{self.max_alm_iter} | {h.size} spins...")
            spins = self._solve_ising_oscillator(J, h)
            
            # 5. Decode
            z_sol = self._decode_spins(spins, C, x_min)
            
            # Compute constraint violation
            constr_viol = A @ z_sol - b
            violation_norm = np.linalg.norm(np.maximum(0, constr_viol))
            constraint_violation_history.append(violation_norm)
            
            print(f"    Constraint violation: {violation_norm:.6e}")
            
            # 6. Update Duals: lam = lam + rho * (Az - b)
            # Use np.maximum(0, ...) if these were inequality constraints, 
            # but standard ALM for equalities usually doesn't clip. 
            # However, for robot MPC with inequalities stacked, we might need refinement.
            lam = lam + self.rho * constr_viol
            
            # Adaptive penalty: increase if not converging
            if i_alm > 0:
                if violation_norm > 0.5 * constraint_violation_history[i_alm-1]:
                    self.rho *= 2.0
                    print(f"    Penalty increased to rho={self.rho:.2e}")
            
            # Early stopping
            if violation_norm < 1e-3:
                print(f"    Converged at iteration {i_alm+1}")
                break
            
        print("  SHO: Done.")
        return z_sol

    def _build_penalty_qp(self, Q, p, A, b):
        # min 0.5 z'Qz + p'z + 0.5 * rho * || Az - b ||^2
        # ||Az-b||^2 = (Az-b)'(Az-b) = z'A'Az - 2b'Az + b'b
        # Cost = 0.5 z'(Q + rho A'A)z + (p - rho A'b)'z
        Q_aug = Q + self.rho * (A.T @ A)
        p_aug = p - self.rho * (A.T @ b)
        return Q_aug, p_aug

    def _continuous_to_qubo(self, Q_c, p_c, x_min, x_max):
        n_z = Q_c.shape[0]
        n_bits = self.n_bits
        n_binary = n_z * n_bits
        
        C = np.zeros((n_z, n_binary))
        scales = x_max - x_min
        
        # z = x_min + C * s
        powers = 2.0**(-np.arange(1, n_bits + 1))
        for i in range(n_z):
            C[i, i*n_bits : (i+1)*n_bits] = scales[i] * powers
                
        # transform 0.5 z'Qz + p'z
        # z = m + Cs
        # 0.5 (m+Cs)'Q(m+Cs) + p'(m+Cs)
        # = 0.5 (s'C'QC s + 2m'QC s + m'Qm) + p'Cs + p'm
        # Quadratic in s: 0.5 s'(C'QC)s
        # Linear in s: (m'Q C + p'C) s
        
        m = x_min
        Q_qubo = C.T @ Q_c @ C
        # Note: Q is symmetric, so m'QC = (QCm)' = m'Q'C = m'QC. 
        # p_qubo term: C.T @ (Q_c @ m + p_c)
        p_qubo = C.T @ (Q_c @ m + p_c)
        
        return Q_qubo, p_qubo, C

    def _qubo_to_ising(self, Q_qubo, p_qubo):
        """
        Convert QUBO to Ising using s = (sigma + 1)/2 transformation.
        Corrected scaling: h_i = -0.25 * (p_i + sum_j Q_ij)
        """
        n = Q_qubo.shape[0]
        Qs = 0.5 * (Q_qubo + Q_qubo.T) # Ensure symmetry
        
        J = -0.25 * Qs
        np.fill_diagonal(J, 0)
        
        # Corrected scaling from MIT Thesis 2021 (McGoldrick)
        h = -0.25 * (p_qubo + np.sum(Qs, axis=1))
        
        return J, h

    def _solve_ising_oscillator(self, J, h):
        n = len(h)
        omega = np.zeros(n)
        phi0 = np.random.rand(n) * 2 * np.pi
        
        t_eval = np.arange(0, self.T_final, self.dt)
        
        sol = solve_ivp(
            self._oscillator_dynamics, 
            (0, self.T_final), 
            phi0,
            args=(J, h, omega, self.coupling_strength), 
            t_eval=t_eval, 
            max_step=self.dt,
            method='RK45', 
            rtol=1e-6, 
            atol=1e-8
        )
        
        phi_final = sol.y[:, -1] % (2*np.pi)
        # Decode spin: cos(phi) > 0 -> +1, else -1
        spins = np.where(np.cos(phi_final) >= 0, 1, -1)
        return spins

    @staticmethod
    def _oscillator_dynamics(t, phi, J, h, omega, K):
        # phi is (n,)
        # diff[i, j] = phi[j] - phi[i]
        diff = phi[None, :] - phi[:, None]
        interaction = K * np.sum(J * np.sin(diff), axis=1)
        return omega + interaction + h

    def _decode_spins(self, spins, C, x_min):
        s = (spins + 1) / 2.0
        z = x_min + C @ s
        return z


import numpy as np
from scipy.integrate import solve_ivp

class SHOSolver:
    def __init__(self, n_bits=4, rho=50.0, dt=1e-9, T_final=5e-7, max_alm_iter=5):
        """
        n_bits: resolution (bits) per continuous variable
        rho: penalty parameter for constraints
        dt: time step for oscillator simulation
        T_final: duration of oscillator simulation
        max_alm_iter: Augmented Lagrangian iterations
        """
        self.n_bits = n_bits
        self.rho = rho
        self.dt = dt
        self.T_final = T_final
        self.max_alm_iter = max_alm_iter

    def solve(self, qp_matrices, x_min_val=-10.0, x_max_val=10.0):
        """
        Solves QP using simulated Oscillator Ising Machine with Augmented Lagrangian.
        qp_matrices: (Q, p, A_eq, b_eq, A_ineq, k_ineq)
        """
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
        
        # Combine all constraints into equality/inequality forms for ALM
        # For a truly correct neuromorphic solver, we handle equalities via ALM
        # and inequalities via slack variables or simple penalty.
        # Here we use ALM for all constraints treated as equalities as a baseline.
        
        A = np.vstack([A_eq, A_ineq]) if A_ineq.size else A_eq
        b = np.concatenate([b_eq, k_ineq]) if A_ineq.size else b_eq
        
        n_z = Q.shape[0]
        n_m = A.shape[0]
        
        # Dual variables (Lagrange multipliers)
        lam = np.zeros(n_m)
        z_sol = np.zeros(n_z)
        
        x_min = x_min_val * np.ones(n_z)
        x_max = x_max_val * np.ones(n_z)

        # Augmented Lagrangian Loop
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
            
            # 6. Update Duals: lam = lam + rho * (Az - b)
            # This is the core of ALM to achieve exact constraint satisfaction.
            lam = lam + self.rho * (A @ z_sol - b)
            
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
        # QUBO: 0.5 s'Qs + p's, s in {0,1}
        # s = (sigma + 1)/2
        n = Q_qubo.shape[0]
        Qs = 0.5 * (Q_qubo + Q_qubo.T) # Ensure symmetry
        
        J = -0.25 * Qs
        np.fill_diagonal(J, 0)
        
        # Consistent scaling with J = -0.25 * Q
        # Linear term derivation: (p_i/2 + sum_j Q_ij/2) * sigma_i
        h = -0.5 * (p_qubo + np.sum(Qs, axis=1))
        
        return J, h

    def _solve_ising_oscillator(self, J, h):
        n = len(h)
        # Kuramoto-like dynamics
        # phi_dot_i = sum J_ij sin(phi_j - phi_i) + h_i (?)
        # User code used: dphi[i] += J[i,j] * sin(phi[j] - phi[i]) + h[i]
        # And omega = 0.
        
        omega = np.zeros(n)
        phi0 = np.random.rand(n) * 2 * np.pi
        
        # High-freq simulation requires small steps
        t_eval = np.arange(0, self.T_final, self.dt)
        
        sol = solve_ivp(self._oscillator_dynamics, (0, self.T_final), phi0,
                        args=(J, h, omega), t_eval=t_eval, max_step=self.dt)
        
        phi_final = sol.y[:, -1] % (2*np.pi)
        # Decode spin: cos(phi) > 0 -> +1, else -1
        spins = np.where(np.cos(phi_final) >= 0, 1, -1)
        return spins

    @staticmethod
    def _oscillator_dynamics(t, phi, J, h, omega):
        # phi is (n,)
        # diff[i, j] = phi[j] - phi[i]
        diff = phi[None, :] - phi[:, None]
        interaction = np.sum(J * np.sin(diff), axis=1)
        return omega + interaction + h

    def _decode_spins(self, spins, C, x_min):
        s = (spins + 1) / 2.0
        z = x_min + C @ s
        return z

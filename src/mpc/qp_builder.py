
import numpy as np
import casadi as ca

class MPCBuilder:
    def __init__(self, arm_model, N=20, dt=0.02, 
                 Qx=None, Qf=None, R=None, 
                 bounds=None):
        self.arm = arm_model
        self.f_fun, self.A_fun, self.B_fun = self.arm.get_dynamics_functions()
        self.N = N
        self.dt = dt
        self.nx = self.arm.nx
        self.nu = self.arm.nu

        # Default weights
        self.Qx = np.diag([100, 100, 1, 1]) if Qx is None else Qx
        self.Qf = np.diag([200, 200, 2, 2]) if Qf is None else Qf
        self.R  = np.diag([0.1, 0.1])      if R is None else R

        # Default bounds
        if bounds is None:
            self.theta_min = np.array([-np.pi, -np.pi])
            self.theta_max = np.array([ np.pi,  np.pi])
            self.tau_min = np.array([-10.0, -10.0])
            self.tau_max = np.array([ 10.0,  10.0])
        else:
            self.theta_min = bounds.get('theta_min')
            self.theta_max = bounds.get('theta_max')
            self.tau_min = bounds.get('tau_min')
            self.tau_max = bounds.get('tau_max')

    def build_qp(self, x0, x_ref_traj):
        """
        Builds the QP matrices Q, p, A_eq, b_eq, A_ineq, k_ineq
        for the given initial state x0 and reference trajectory.
        """
        N = self.N
        nx = self.nx
        nu = self.nu
        dt = self.dt
        n_z = N * (nx + nu) + nx

        # Helper indices
        def idx_x(k): return k * (nx + nu)
        def idx_u(k): return k * (nx + nu) + nx

        # --- Cost Function (Q, p) ---
        Q = np.zeros((n_z, n_z))
        p = np.zeros(n_z)

        for k in range(N):
            # Block H_k = diag(Qx, R)
            H = np.block([
                [self.Qx, np.zeros((nx, nu))],
                [np.zeros((nu, nx)), self.R]
            ])
            start = idx_x(k)
            Q[start:start+nx+nu, start:start+nx+nu] += H

            # Linear term p_k: -2 * Qx * x_ref
            xr = x_ref_traj[k]
            # Structure: [x_k, u_k]. Linear term for x_k is -x_ref^T Qx - x_ref^T Qx^T (symmetric) -> -2 Qx x_ref
            # Linear term for u_k is 0 if ref u is 0.
            lin = np.concatenate([-2 * self.Qx @ xr, np.zeros(nu)])
            p[start:start+nx+nu] += lin

        # Terminal cost
        start = N * (nx + nu)
        Q[start:start+nx, start:start+nx] += self.Qf
        xrN = x_ref_traj[N]
        p[start:start+nx] += -2 * self.Qf @ xrN

        # --- Equality Constraints (Dynamics) ---
        # x0 = x_init
        # x_{k+1} = A_k x_k + B_k u_k + c_k
        
        A_eq_rows = []
        b_eq_rows = []

        # Initial condition constraint: I * x0 = x_init
        row0 = np.zeros((nx, n_z))
        row0[:, :nx] = np.eye(nx)
        A_eq_rows.append(row0)
        b_eq_rows.append(x0)

        for k in range(N):
            # Linearize around reference (or nominal)
            x_bar = x_ref_traj[k]
            u_bar = np.zeros(nu) # Nominal control usually 0 or gravity comp (simplification: 0)
            
            # Eval CasADi functions
            f_val = np.array(self.f_fun(x_bar, u_bar)).flatten()
            A_c = np.array(self.A_fun(x_bar, u_bar))
            B_c = np.array(self.B_fun(x_bar, u_bar))
            
            # Discretize
            A_k = np.eye(nx) + dt * A_c
            B_k = dt * B_c
            c_k = dt * (f_val - A_c @ x_bar - B_c @ u_bar)

            # Constraint: x_{k+1} - A_k x_k - B_k u_k = c_k
            row = np.zeros((nx, n_z))
            row[:, idx_x(k):idx_x(k)+nx] = -A_k
            row[:, idx_u(k):idx_u(k)+nu] = -B_k
            row[:, idx_x(k+1):idx_x(k+1)+nx] = np.eye(nx)
            
            A_eq_rows.append(row)
            b_eq_rows.append(c_k)

        A_eq = np.vstack(A_eq_rows)
        b_eq = np.concatenate(b_eq_rows)

        # --- Inequality Constraints (Bounds) ---
        # A_ineq z <= k_ineq
        # Upper bounds: I * z <= max
        # Lower bounds: -I * z <= -min  =>  z >= min
        
        A_ineq_rows = []
        k_ineq_rows = []

        for k in range(N):
            # Control bounds
            # u_k <= tau_max
            row = np.zeros((nu, n_z))
            row[:, idx_u(k):idx_u(k)+nu] = np.eye(nu)
            A_ineq_rows.append(row)
            k_ineq_rows.append(self.tau_max)

            # -u_k <= -tau_min
            row = np.zeros((nu, n_z))
            row[:, idx_u(k):idx_u(k)+nu] = -np.eye(nu)
            A_ineq_rows.append(row)
            k_ineq_rows.append(-self.tau_min)

            # State bounds (Position only for now, can add velocity)
            # theta <= theta_max
            row = np.zeros((self.arm.nq, n_z))
            row[:, idx_x(k):idx_x(k)+self.arm.nq] = np.eye(self.arm.nq)
            A_ineq_rows.append(row)
            k_ineq_rows.append(self.theta_max)

            # -theta <= -theta_min
            row = np.zeros((self.arm.nq, n_z))
            row[:, idx_x(k):idx_x(k)+self.arm.nq] = -np.eye(self.arm.nq)
            A_ineq_rows.append(row)
            k_ineq_rows.append(-self.theta_min)
        
        # NOTE: Usually terminal state bounds are also included. 
        # For simplicity we skip explicit terminal bounds loop or just assume N-1 covers enough
        # But strictly we should bound x_N too. Let's add x_N bounds.
        
        start = N * (nx + nu) # x_N index
        # theta_N <= max
        row = np.zeros((self.arm.nq, n_z))
        row[:, start:start+self.arm.nq] = np.eye(self.arm.nq)
        A_ineq_rows.append(row)
        k_ineq_rows.append(self.theta_max)
        
        # -theta_N <= -min
        row = np.zeros((self.arm.nq, n_z))
        row[:, start:start+self.arm.nq] = -np.eye(self.arm.nq)
        A_ineq_rows.append(row)
        k_ineq_rows.append(-self.theta_min)

        A_ineq = np.vstack(A_ineq_rows)
        k_ineq = np.concatenate(k_ineq_rows)

        return Q, p, A_eq, b_eq, A_ineq, k_ineq

    def build_reference_trajectory(self, x_current, x_goal):
        """Simple linear interpolation in joint space for N steps."""
        theta_curr = x_current[:2]
        theta_goal = x_goal[:2]
        
        x_refs = []
        for k in range(self.N + 1):
            alpha = k / self.N
            theta_k = theta_curr + (theta_goal - theta_curr) * alpha
            dtheta_k = np.zeros(2) # Target zero velocity
            x_refs.append(np.concatenate([theta_k, dtheta_k]))
        return np.array(x_refs)

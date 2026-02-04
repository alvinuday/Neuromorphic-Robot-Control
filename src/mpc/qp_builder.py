
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
        self.Qx = np.diag([2000, 2000, 100, 100]) if Qx is None else Qx
        self.Qf = np.diag([5000, 5000, 200, 200]) if Qf is None else Qf
        self.R  = np.diag([0.001, 0.001])      if R is None else R
        self.Qs = 1e6 # Soft constraint penalty

        # Default bounds
        if bounds is None:
            self.theta_min = np.array([-np.pi, -np.pi])
            self.theta_max = np.array([ np.pi,  np.pi])
            self.tau_min = np.array([-50.0, -50.0])
            self.tau_max = np.array([ 50.0,  50.0])
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
        n_z = N * (nx + nu) + nx # Decision variables without slacks
        # Total decision variables: z = [x_0, u_0, x_1, u_1, ..., x_N, slack_0, ..., slack_N]
        # We add slack variables for position constraints
        n_slack = (N + 1) * self.arm.nq
        n_z_total = n_z + n_slack

        # Helper indices
        def idx_x(k): return k * (nx + nu)
        def idx_u(k): return k * (nx + nu) + nx

        # --- Cost Function (Q, p) ---
        Q = np.zeros((n_z_total, n_z_total))
        p = np.zeros(n_z_total)

        for k in range(N):
            # Block H_k = 2 * diag(Qx, R)
            H = 2.0 * np.block([
                [self.Qx, np.zeros((nx, nu))],
                [np.zeros((nu, nx)), self.R]
            ])
            start = idx_x(k)
            Q[start:start+nx+nu, start:start+nx+nu] += H

            # Linear term p_k: -2 * Qx * x_ref
            xr = x_ref_traj[k]
            # Structure: [x_k, u_k]. Linear term for x_k is -x_ref^T Qx - x_ref^T Qx^T (symmetric) -> -2 Qx x_ref
            lin = np.concatenate([-2 * self.Qx @ xr, np.zeros(nu)])
            p[start:start+nx+nu] += lin

        # Terminal cost
        start_xN = N * (nx + nu)
        Q[start_xN:start_xN+nx, start_xN:start_xN+nx] += 2.0 * self.Qf
        xrN = x_ref_traj[N]
        p[start_xN:start_xN+nx] += -2 * self.Qf @ xrN

        # Soft constraint cost: sum(Qs * s_k^2)
        slack_start = n_z
        for k in range(N + 1):
            s_idx = slack_start + k * self.arm.nq
            Q[s_idx:s_idx+self.arm.nq, s_idx:s_idx+self.arm.nq] += 2.0 * self.Qs * np.eye(self.arm.nq)

        # --- Equality Constraints (Dynamics) ---
        A_eq_rows = []
        b_eq_rows = []

        # Initial condition constraint: I * x0 = x_init
        row0 = np.zeros((nx, n_z_total))
        row0[:, :nx] = np.eye(nx)
        A_eq_rows.append(row0)
        b_eq_rows.append(x0)

        for k in range(N):
            # Linearize around reference (or nominal)
            x_bar = x_ref_traj[k]
            theta_bar = x_bar[:2]
            
            # Gravity compensation for u_bar
            m1, m2, l1, l2, g = self.arm.m1, self.arm.m2, self.arm.l1, self.arm.l2, self.arm.g
            th1, th2 = theta_bar[0], theta_bar[1]
            G1 = (m1*l1/2 + m2*l1)*g*np.sin(th1) + m2*l2/2*g*np.sin(th1+th2)
            G2 = m2*l2/2*g*np.sin(th1+th2)
            u_bar = np.array([G1, G2])
            
            # Eval CasADi functions
            f_val = np.array(self.f_fun(x_bar, u_bar)).flatten()
            A_c = np.array(self.A_fun(x_bar, u_bar))
            B_c = np.array(self.B_fun(x_bar, u_bar))
            
            # Discretize
            A_k = np.eye(nx) + dt * A_c
            B_k = dt * B_c
            c_k = dt * (f_val - A_c @ x_bar - B_c @ u_bar)

            # Constraint: x_{k+1} - A_k x_k - B_k u_k = c_k
            row = np.zeros((nx, n_z_total))
            row[:, idx_x(k):idx_x(k)+nx] = -A_k
            row[:, idx_u(k):idx_u(k)+nu] = -B_k
            row[:, idx_x(k+1):idx_x(k+1)+nx] = np.eye(nx)
            
            A_eq_rows.append(row)
            b_eq_rows.append(c_k)

        A_eq = np.vstack(A_eq_rows)
        b_eq = np.concatenate(b_eq_rows)

        # --- Inequality Constraints (Bounds) ---
        A_ineq_rows = []
        k_ineq_rows = []

        for k in range(N):
            # Control bounds (hard)
            row = np.zeros((nu, n_z_total))
            row[:, idx_u(k):idx_u(k)+nu] = np.eye(nu)
            A_ineq_rows.append(row)
            k_ineq_rows.append(self.tau_max)

            row = np.zeros((nu, n_z_total))
            row[:, idx_u(k):idx_u(k)+nu] = -np.eye(nu)
            A_ineq_rows.append(row)
            k_ineq_rows.append(-self.tau_min)

            # State bounds (soft)
            # theta_k - slack_k <= theta_max
            # -theta_k - slack_k <= -theta_min
            s_idx = slack_start + k * self.arm.nq
            
            row = np.zeros((self.arm.nq, n_z_total))
            row[:, idx_x(k):idx_x(k)+self.arm.nq] = np.eye(self.arm.nq)
            row[:, s_idx:s_idx+self.arm.nq] = -np.eye(self.arm.nq)
            A_ineq_rows.append(row)
            k_ineq_rows.append(self.theta_max)

            row = np.zeros((self.arm.nq, n_z_total))
            row[:, idx_x(k):idx_x(k)+self.arm.nq] = -np.eye(self.arm.nq)
            row[:, s_idx:s_idx+self.arm.nq] = -np.eye(self.arm.nq)
            A_ineq_rows.append(row)
            k_ineq_rows.append(-self.theta_min)
            
            # slack >= 0
            row = np.zeros((self.arm.nq, n_z_total))
            row[:, s_idx:s_idx+self.arm.nq] = -np.eye(self.arm.nq)
            A_ineq_rows.append(row)
            k_ineq_rows.append(np.zeros(self.arm.nq))

        # Terminal state bounds (soft)
        k = N
        s_idx = slack_start + k * self.arm.nq
        row = np.zeros((self.arm.nq, n_z_total))
        row[:, start_xN:start_xN+self.arm.nq] = np.eye(self.arm.nq)
        row[:, s_idx:s_idx+self.arm.nq] = -np.eye(self.arm.nq)
        A_ineq_rows.append(row)
        k_ineq_rows.append(self.theta_max)

        row = np.zeros((self.arm.nq, n_z_total))
        row[:, start_xN:start_xN+self.arm.nq] = -np.eye(self.arm.nq)
        row[:, s_idx:s_idx+self.arm.nq] = -np.eye(self.arm.nq)
        A_ineq_rows.append(row)
        k_ineq_rows.append(-self.theta_min)
        
        row = np.zeros((self.arm.nq, n_z_total))
        row[:, s_idx:s_idx+self.arm.nq] = -np.eye(self.arm.nq)
        A_ineq_rows.append(row)
        k_ineq_rows.append(np.zeros(self.arm.nq))

        A_ineq = np.vstack(A_ineq_rows)
        k_ineq = np.concatenate(k_ineq_rows)

        return Q, p, A_eq, b_eq, A_ineq, k_ineq

    def build_reference_trajectory(self, x_current, x_goal):
        """Shortest angular path interpolation in joint space for N steps."""
        theta_curr = x_current[:2]
        theta_goal = x_goal[:2]
        
        # Calculate shortest path for angles
        def shortest_angular_dist(target, current):
            diff = (target - current + np.pi) % (2 * np.pi) - np.pi
            return diff

        theta_diff = shortest_angular_dist(theta_goal, theta_curr)
        
        x_refs = []
        for k in range(self.N + 1):
            alpha = k / self.N
            theta_k = theta_curr + theta_diff * alpha
            dtheta_k = np.zeros(2) # Target zero velocity
            x_refs.append(np.concatenate([theta_k, dtheta_k]))
        return np.array(x_refs)

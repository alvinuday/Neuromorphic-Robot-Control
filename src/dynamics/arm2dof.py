
import casadi as ca
import numpy as np

class Arm2DOF:
    def __init__(self, m1=1.0, m2=1.0, l1=0.5, l2=0.5, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

        self._setup_dynamics()

    def _setup_dynamics(self):
        # State and control dimensions
        self.nq = 2
        self.nv = 2
        self.nx = self.nq + self.nv
        self.nu = 2

        # Symbolic variables
        theta = ca.SX.sym('theta', self.nq)
        dtheta = ca.SX.sym('dtheta', self.nv)
        tau = ca.SX.sym('tau', self.nu)
        x = ca.vertcat(theta, dtheta)

        # 2-link parameters
        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g

        # Inertia Matrix M(theta)
        th2 = theta[1]
        I11 = m1*l1**2/3 + m2*(l1**2 + l2**2/3 + l1*l2*ca.cos(th2))
        I12 = m2*(l2**2/3 + 0.5*l1*l2*ca.cos(th2))
        I21 = I12
        I22 = m2*l2**2/3
        M = ca.vertcat(
            ca.hcat([I11, I12]),
            ca.hcat([I21, I22])
        )

        # Coriolis Matrix C(theta, dtheta)
        dth1, dth2 = dtheta[0], dtheta[1]
        h = -m2*l1*l2*ca.sin(th2)
        C11 = h*dth2
        C12 = h*(dth1 + dth2)
        C21 = -h*dth1
        C22 = 0.0
        C = ca.vertcat(
            ca.hcat([C11, C12]),
            ca.hcat([C21, C22])
        )

        # Gravity Matrix G(theta)
        th1, th2 = theta[0], theta[1]
        G1 = (m1*l1/2 + m2*l1)*g*ca.sin(th1) + m2*l2/2*g*ca.sin(th1+th2)
        G2 = m2*l2/2*g*ca.sin(th1+th2)
        G_mat = ca.vertcat(G1, G2)

        # Dynamics: M*ddtheta + C*dtheta + G = tau
        # ddtheta = M_inv * (tau - C*dtheta - G)
        inv_M = ca.inv(M) # Or solve
        ddtheta = inv_M @ (tau - C @ dtheta - G_mat)

        f = ca.vertcat(dtheta, ddtheta)  # xdot

        # CasADi functions
        self.f_fun = ca.Function('f_fun', [x, tau], [f])
        self.A_fun = ca.Function('A_fun', [x, tau], [ca.jacobian(f, x)])
        self.B_fun = ca.Function('B_fun', [x, tau], [ca.jacobian(f, tau)])
        
        # Exposure for testing
        self.M_fun = ca.Function('M_fun', [theta], [M])

    def get_dynamics_functions(self):
        """Returns the CasADi functions for f, A, and B."""
        return self.f_fun, self.A_fun, self.B_fun

    def step_dynamics(self, x, u, dt):
        """Forward integration of dynamics (Euler)."""
        xdot = np.array(self.f_fun(x, u)).flatten()
        return x + dt * xdot

    def forward_kinematics(self, theta):
        """Returns ((0, x1, x2), (0, y1, y2)) for plotting."""
        th1, th2 = theta[0], theta[1]
        # Angle from downward vertical
        x1 = self.l1 * np.sin(th1)
        y1 = -self.l1 * np.cos(th1)
        x2 = x1 + self.l2 * np.sin(th1 + th2)
        y2 = y1 - self.l2 * np.cos(th1 + th2)
        return np.array([[0, x1, x2],
                         [0, y1, y2]])

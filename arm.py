import casadi as ca
import numpy as np

# State and control dimensions
nq = 2  # 2 DOF
nv = 2
nx = nq + nv
nu = 2

# Symbolic variables
theta = ca.SX.sym('theta', nq)
dtheta = ca.SX.sym('dtheta', nv)
tau = ca.SX.sym('tau', nu)
x = ca.vertcat(theta, dtheta)

# 2-link parameters (example)
m1, m2 = 1.0, 1.0
l1, l2 = 0.5, 0.5
g = 9.81

def inertia_matrix(theta):
    th2 = theta[1]
    I11 = m1*l1**2/3 + m2*(l1**2 + l2**2/3 + l1*l2*ca.cos(th2))
    I12 = m2*(l2**2/3 + 0.5*l1*l2*ca.cos(th2))
    I21 = I12
    I22 = m2*l2**2/3
    return ca.vertcat(
        ca.hcat([I11, I12]),
        ca.hcat([I21, I22])
    )

def coriolis(theta, dtheta):
    th2 = theta[1]
    dth1, dth2 = dtheta[0], dtheta[1]
    h = -m2*l1*l2*ca.sin(th2)
    C11 = h*dth2
    C12 = h*(dth1 + dth2)
    C21 = -h*dth1
    C22 = 0
    return ca.vertcat(
        ca.hcat([C11, C12]),
        ca.hcat([C21, C22])
    )

def gravity(theta):
    th1, th2 = theta[0], theta[1]
    G1 = (m1*l1/2 + m2*l1)*g*ca.cos(th1) + m2*l2/2*g*ca.cos(th1+th2)
    G2 = m2*l2/2*g*ca.cos(th1+th2)
    return ca.vertcat(G1, G2)

# Continuous dynamics
M = inertia_matrix(theta)
C = coriolis(theta, dtheta)
G = gravity(theta)
ddtheta = ca.solve(M, tau - C @ dtheta - G)

f = ca.vertcat(dtheta, ddtheta)  # xdot

# CasADi functions
f_fun = ca.Function('f_fun', [x, tau], [f])
A_fun = ca.Function('A_fun', [x, tau], [ca.jacobian(f, x)])
B_fun = ca.Function('B_fun', [x, tau], [ca.jacobian(f, tau)])

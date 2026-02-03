
import osqp
import numpy as np
from scipy import sparse

class OSQPSolver:
    def __init__(self):
        pass

    def solve(self, qp_matrices):
        """
        Solves the QP using OSQP.
        qp_matrices: tuple (Q, p, A_eq, b_eq, A_ineq, k_ineq)
        Returns: full decision vector z
        """
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
        
        n_z = Q.shape[0]

        # Stack constraints:
        # A_eq z = b_eq
        # A_ineq z <= k_ineq  =>  -inf <= A_ineq z <= k_ineq
        
        # OSQP format: l <= A z <= u
        
        # Equality part
        # l_eq = b_eq, u_eq = b_eq
        
        # Inequality part
        # l_ineq = -inf, u_ineq = k_ineq
        
        A = np.vstack([A_eq, A_ineq])
        
        l_eq = b_eq
        u_eq = b_eq
        
        l_ineq = -np.inf * np.ones(len(k_ineq))
        u_ineq = k_ineq
        
        l = np.concatenate([l_eq, l_ineq])
        u = np.concatenate([u_eq, u_ineq])
        
        # Create sparse matrices
        P_sp = sparse.csc_matrix(Q)
        A_sp = sparse.csc_matrix(A)
        
        # Solve
        prob = osqp.OSQP()
        prob.setup(P=P_sp, q=p, A=A_sp, l=l, u=u, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
        res = prob.solve()
        
        if res.info.status != 'solved':
            # print("OSQP Warning:", res.info.status)
            pass
            
        return res.x

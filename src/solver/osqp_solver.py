
import osqp
import numpy as np
from scipy import sparse

class OSQPSolver:
    def __init__(self):
        # We'll create a new OSQP instance or re-setup every time for reliability 
        # because the linearized A matrix structure/values change significantly.
        pass

    def solve(self, qp_matrices):
        """
        Solves the QP using OSQP.
        qp_matrices: tuple (Q, p, A_eq, b_eq, A_ineq, k_ineq)
        Returns: full decision vector z
        """
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
        
        # OSQP format: l <= A z <= u
        A = np.vstack([A_eq, A_ineq])
        l = np.concatenate([b_eq, -np.inf * np.ones(len(k_ineq))])
        u = np.concatenate([b_eq, k_ineq])
        
        # Create sparse matrices
        P_sp = sparse.csc_matrix(Q)
        A_sp = sparse.csc_matrix(A)
        
        # Re-initialize solver every step to ensure time-varying A matrix is correct.
        # This is fast for small MPC problems and avoids indexing/sparsity issues with 'update'.
        prob = osqp.OSQP()
        prob.setup(P=P_sp, q=p, A=A_sp, l=l, u=u, verbose=False, 
                   eps_abs=1e-4, eps_rel=1e-4, warm_start=True)

        res = prob.solve()
        
        if res.info.status != 'solved':
            return None # Return None to let caller handle failure
            
        return res.x

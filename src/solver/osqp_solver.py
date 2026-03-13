
"""OSQP-based Quadratic Programming Solver.

This module provides a wrapper around the OSQP (Operator Splitting Quadratic Program)
solver for solving constrained QP problems arising in Model Predictive Control.
"""

import osqp
import numpy as np
from scipy import sparse
from typing import Optional, Tuple
from numpy.typing import NDArray


class OSQPSolver:
    """Quadratic Programming solver using OSQP algorithm.
    
    Solves problems of the form:
        min 0.5 z' Q z + p' z
        s.t. A_eq z = b_eq
             A_ineq z <= k_ineq
    
    Attributes:
        None (stateless solver)
    """
    
    def __init__(self) -> None:
        """Initialize the OSQP solver.
        
        Note: A new OSQP instance is created for each solve() call to handle
        time-varying constraint matrices in MPC applications.
        """
        pass

    def solve(
        self, 
        qp_matrices: Tuple[
            NDArray[np.float64], 
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]
        ]
    ) -> Optional[NDArray[np.float64]]:
        """Solve a quadratic programming problem using OSQP.
        
        Converts the problem to OSQP standard form and solves it.
        
        Args:
            qp_matrices: Tuple of QP matrices:
                - Q: Hessian matrix of shape (n, n), positive semidefinite
                - p: Linear cost vector of shape (n,)
                - A_eq: Equality constraint matrix of shape (m_eq, n)
                - b_eq: Equality constraint RHS of shape (m_eq,)
                - A_ineq: Inequality constraint matrix of shape (m_ineq, n)
                - k_ineq: Inequality constraint RHS of shape (m_ineq,)
                
        Returns:
            Solution vector z of shape (n,) if solver converged, else None
            
        Raises:
            None (returns None on failure instead of raising)
        """
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
        
        # Convert to OSQP standard form: l <= A z <= u
        A = np.vstack([A_eq, A_ineq])
        l = np.concatenate([b_eq, -np.inf * np.ones(len(k_ineq))])
        u = np.concatenate([b_eq, k_ineq])
        
        # Convert to sparse format for efficiency
        P_sp = sparse.csc_matrix(Q)
        A_sp = sparse.csc_matrix(A)
        
        # Create new solver instance each step for time-varying A
        # (faster and avoids sparsity index issues vs. in-place updates)
        prob = osqp.OSQP()
        prob.setup(
            P=P_sp, q=p, A=A_sp, l=l, u=u, 
            verbose=False, 
            eps_abs=1e-4, eps_rel=1e-4, 
            warm_start=True
        )

        res = prob.solve()
        
        if res.info.status != 'solved':
            # Return None to let caller handle failure
            return None
            
        return res.x

"""xArm 6-DOF simplified dynamics using diagonal inertia approximation."""
import numpy as np
from typing import Optional


class XArmDynamics:
    """
    Simplified 6-DOF xArm dynamics using diagonal-dominant approximation.
    
    Full M(q) computation from URDF is not needed here — the diagonal
    approximation keeps M(q) positive-definite everywhere, avoiding
    singularities in the MPC linearization.
    
    Gravity vector G(q) uses the full geometric chain.
    """
    
    def __init__(self, config: dict):
        cfg = config['robot']
        self.masses   = np.array(cfg['dynamics']['link_masses'])     # [6]
        self.lengths  = np.array(cfg['dynamics']['link_lengths'])    # [6]
        self.g        = cfg['dynamics']['gravity']
        self.n        = 6

    def inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """M(q) — [6,6] positive definite.  Diagonal rod-approximation."""
        assert q.shape == (6,), f"Expected q shape (6,), got {q.shape}"
        # Diagonal: I_i = m_i * L_i^2 / 3 + damping_floor
        diag = self.masses * self.lengths**2 / 3.0 + 0.05
        
        # Add small off-diagonal coupling via sin(q) scaling
        M = np.diag(diag)
        for i in range(1, self.n):
            coupling = 0.02 * self.masses[i] * abs(np.sin(q[i]))
            M[i-1, i] = M[i, i-1] = coupling
        
        return M

    def coriolis_vector(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """C(q,qdot)·qdot — [6] simplified centrifugal/Coriolis."""
        assert q.shape == (6,), f"Expected q shape (6,), got {q.shape}"
        assert qdot.shape == (6,), f"Expected qdot shape (6,), got {qdot.shape}"
        
        # Simplified: C·qdot ≈ (m * l^2 / 2) * sin(q) * qdot^2
        c = self.masses * self.lengths**2 / 2.0 * np.sin(q) * qdot**2
        
        return c

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """G(q) — [6] gravity torques using geometric chain."""
        assert q.shape == (6,), f"Expected q shape (6,), got {q.shape}"
        
        G = np.zeros(6)
        # Cumulative effect: each joint must support all links above it
        for i in range(self.n - 1, -1, -1):
            # Gravity contribution from link i+1...n to joint i
            mass_above = self.masses[i:].sum()
            length_i   = self.lengths[i]
            
            # Only joints with vertical component contribute
            # (joints 1, 2, 4: shoulder pitch, elbow, wrist pitch)
            vertical_gain = abs(np.cos(q[i])) if i in (1, 2, 4) else 0.0
            G[i] = mass_above * self.g * length_i * vertical_gain / 2.0
        
        return G

    def forward_dynamics(self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """qddot = M(q)^-1 * (tau - C*qdot - G(q))"""
        assert q.shape == (6,)
        assert qdot.shape == (6,)
        assert tau.shape == (6,)
        
        M = self.inertia_matrix(q)
        C = self.coriolis_vector(q, qdot)
        G = self.gravity_vector(q)
        
        return np.linalg.solve(M, tau - C - G)

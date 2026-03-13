"""
3-DOF Spatial Arm Lagrangian Dynamics.

Complete equations of motion:
    M(q)q̈ + C(q,q̇)q̇ + G(q) = τ

Where:
    M(q) ∈ ℝ³ˣ³ = mass/inertia matrix (symmetric, positive definite)
    C(q,q̇) ∈ ℝ³ˣ³ = Coriolis/centrifugal matrix
    G(q) ∈ ℝ³ = gravity vector
    τ ∈ ℝ³ = joint torques (control input)

Key property: G[0] = 0 always (azimuth joint rotates about vertical z-axis)

Full derivation: Section 5 of 3darm_smolvla_sl_mpc_techspec.md
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Physical constants
GRAVITY = 9.81  # m/s²


class Arm3DOFDynamics:
    """3-DOF spatial arm Lagrangian dynamics."""
    
    def __init__(self,
                 L0: float = 0.10,    # base height (m)
                 L1: float = 0.25,    # upper arm length
                 L2: float = 0.20,    # forearm length
                 m1: float = 0.50,    # base+motor mass (kg)
                 m2: float = 0.40,    # link 2 mass
                 m3: float = 0.20,    # link 3 + gripper mass
                 b1: float = 0.1,     # viscous damping (Nm·s/rad)
                 b2: float = 0.1,
                 b3: float = 0.05,
                 mu1: float = 0.05,   # Coulomb friction (Nm)
                 mu2: float = 0.05,
                 mu3: float = 0.02):
        """
        Initialize arm dynamics.
        
        Args:
            L0, L1, L2: Link lengths
            m1, m2, m3: Link masses
            bᵢ: Viscous damping coefficients
            μᵢ: Coulomb friction coefficients
        """
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        
        self.b = np.array([b1, b2, b3])          # viscous damping
        self.mu = np.array([mu1, mu2, mu3])      # Coulomb friction
        
        # Precomputed inertia tensors (thin rod approximations)
        # For uniform rod: Ixx = Iyy = mL²/12, Izz ≈ 0
        self.I2_xx = m2 * L1**2 / 12
        self.I2_yy = m2 * L1**2 / 12
        
        self.I3_xx = m3 * L2**2 / 12
        self.I3_yy = m3 * L2**2 / 12
    
    # ═════════════════════════════════════════════════════════════════════════
    # Mass Matrix M(q)
    # ═════════════════════════════════════════════════════════════════════════
    
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute mass/inertia matrix M(q) ∈ ℝ³ˣ³.
        
        Block diagonal structure (Section 5.3):
            M(q) = [M₁₁(q)    0          0      ]
                   [  0      M₂₂(q)   M₂₃(q)  ]
                   [  0      M₂₃(q)   M₃₃     ]
        
        Where:
            M₁₁ = (m₂+m₃)L₁²cos²(q₂) + m₃L₂²cos²(q₂+q₃) + 2m₃L₁L₂cos(q₂)cos(q₂+q₃)
            M₂₂ = m₂L₁²/3 + m₃(L₁² + L₂²/3 + L₁L₂cos(q₃))
            M₃₃ = m₃L₂²/3
            M₂₃ = m₃(L₂²/3 + L₁L₂cos(q₃)/2)
        
        Args:
            q: Joint angles [q₁, q₂, q₃]
        
        Returns:
            M: Mass matrix ∈ ℝ³ˣ³, symmetric positive definite
        """
        assert q.shape == (3,), f"Expected shape (3,), got {q.shape}"
        
        q2, q3 = q[1], q[2]
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        c3 = np.cos(q3)
        s3 = np.sin(q3)
        c23 = np.cos(q2 + q3)
        
        # Block (1,1): azimuth inertia
        M11 = ((self.m2 + self.m3) * self.L1**2 * c2**2 + 
               self.m3 * self.L2**2 * c23**2 + 
               2 * self.m3 * self.L1 * self.L2 * c2 * c23)
        
        # Block (2,2): shoulder + elbow elevation
        M22 = (self.m2 * self.L1**2 / 3 + 
               self.m3 * (self.L1**2 + self.L2**2 / 3 + self.L1 * self.L2 * c3))
        
        # Block (3,3): elbow only
        M33 = self.m3 * self.L2**2 / 3
        
        # Cross term (2,3) = (3,2)
        M23 = self.m3 * (self.L2**2 / 3 + self.L1 * self.L2 * c3 / 2)
        
        # Build matrix
        M = np.zeros((3, 3))
        M[0, 0] = M11
        M[1, 1] = M22
        M[1, 2] = M23
        M[2, 1] = M23
        M[2, 2] = M33
        
        return M
    
    # ═════════════════════════════════════════════════════════════════════════
    # Coriolis & Centrifugal Matrix C(q,q̇)
    # ═════════════════════════════════════════════════════════════════════════
    
    def coriolis_matrix(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal matrix C(q,q̇) via Christoffel symbols.
        
        C is defined such that the dynamics are:
            M·q̈ + C·q̇ + G = τ
        
        Computed via Christoffel symbols (Section 5.4):
            Cᵢⱼ = Σₖ Γᵢⱼₖ q̇ₖ  where Γᵢⱼₖ = ½(∂Mᵢₖ/∂qⱼ + ∂Mᵢⱼ/∂qₖ - ∂Mⱼₖ/∂qᵢ)
        
        Key property: xᵀ(Ṁ - 2C)x = 0 for all x (skew-symmetry)
        
        Args:
            q: Joint angles [q₁, q₂, q₃]
            q_dot: Joint velocities [q̇₁, q̇₂, q̇₃]
        
        Returns:
            C: Coriolis matrix ∈ ℝ³ˣ³
        """
        assert q.shape == (3,) and q_dot.shape == (3,), f"Invalid shapes"
        
        q2, q3 = q[1], q[2]
        q2_dot, q3_dot = q_dot[1], q_dot[2]
        
        c2 = np.cos(q2)
        s2 = np.sin(q2)
        c3 = np.cos(q3)
        s3 = np.sin(q3)
        c23 = np.cos(q2 + q3)
        s23 = np.sin(q2 + q3)
        
        # Precomputed terms
        h = self.m3 * self.L1 * self.L2 * s3  # ∂M₁₁/∂q₃, etc.
        
        # Christoffel symbol Γ₁₁₂
        gamma_112 = -(self.m2 + self.m3) * self.L1**2 * c2 * s2 - h * c2 * s23
        
        # Build Coriolis matrix (many entries are zero by structure)
        C = np.zeros((3, 3))
        
        # C[0,0] row
        C[0, 1] = gamma_112 * q2_dot
        C[0, 2] = -h * c2 * s23 * q3_dot
        
        # C[1,0], C[2,0] rows (transpose symmetry)
        C[1, 0] = gamma_112 * q2_dot
        C[2, 0] = -h * c2 * s23 * q3_dot
        
        # C[1,2] and C[2,1] (coupling between joints 2 and 3)
        beta = h * s3 / 2
        C[1, 2] = -beta * q3_dot
        C[2, 1] = beta * q2_dot
        
        return C
    
    # ═════════════════════════════════════════════════════════════════════════
    # Gravity Vector G(q)
    # ═════════════════════════════════════════════════════════════════════════
    
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation vector G(q) ∈ ℝ³.
        
        Computed as: G(q) = -∑ᵢ mᵢ g ∂pᵢ/∂q
        
        Key properties:
            - G[0] = 0 always (azimuth joint decoupling)
            - G[2] < G[1] in magnitude (elbow less affected)
        
        Formulas (Section 5.5):
            G₁ = 0
            G₂ = -g(m₂L₁cos(q₂)/2 + m₃L₁cos(q₂) + m₃L₂cos(q₂+q₃)/2)
            G₃ = -g·m₃·L₂·cos(q₂+q₃)/2
        
        Args:
            q: Joint angles [q₁, q₂, q₃]
        
        Returns:
            G: Gravity vector ∈ ℝ³
        """
        assert q.shape == (3,), f"Expected shape (3,), got {q.shape}"
        
        q2, q3 = q[1], q[2]
        c2 = np.cos(q2)
        c23 = np.cos(q2 + q3)
        
        G = np.zeros(3)
        
        # Joint 1: zero (azimuth decoupling)
        G[0] = 0.0
        
        # Joint 2: shoulder elevation
        G[1] = -GRAVITY * (self.m2 * self.L1 * c2 / 2 + 
                          self.m3 * self.L1 * c2 + 
                          self.m3 * self.L2 * c23 / 2)
        
        # Joint 3: elbow elevation
        G[2] = -GRAVITY * self.m3 * self.L2 * c23 / 2
        
        return G
    
    # ═════════════════════════════════════════════════════════════════════════
    # Friction & Damping Terms
    # ═════════════════════════════════════════════════════════════════════════
    
    def friction_torques(self, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute friction torques (viscous + Coulomb).
        
        τ_friction = B·q̇ + μ·sign(q̇)
        
        Args:
            q_dot: Joint velocities
        
        Returns:
            tau_fric: Friction torque vector ∈ ℝ³
        """
        tau_fric = self.b * q_dot + self.mu * np.sign(q_dot)
        return tau_fric
    
    # ═════════════════════════════════════════════════════════════════════════
    # Complete Dynamics (Forward Simulation)
    # ═════════════════════════════════════════════════════════════════════════
    
    def state_derivative(self, 
                        q: np.ndarray, 
                        q_dot: np.ndarray, 
                        tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute state time-derivatives: [q̈, q̇].
        
        From M(q)q̈ + C(q,q̇)q̇ + G(q) = τ:
            q̈ = M(q)⁻¹[τ - C(q,q̇)q̇ - G(q) - B·q̇ - μ·sign(q̇)]
        
        Args:
            q: Joint positions [q₁, q₂, q₃]
            q_dot: Joint velocities
            tau: Applied torques
        
        Returns:
            q_dot: Velocities (q̇)
            q_double_dot: Accelerations (q̈)
        """
        # Get dynamics matrices
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)
        tau_fric = self.friction_torques(q_dot)
        
        # RHS of dynamics equation
        rhs = tau - C @ q_dot - G - tau_fric
        
        # Solve for accelerations
        try:
            q_double_dot = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            logger.error(f"Singular mass matrix at q={q}")
            q_double_dot = np.zeros(3)
        
        return q_dot, q_double_dot
    
    # ═════════════════════════════════════════════════════════════════════════
    # Energy Functions (for validation)
    # ═════════════════════════════════════════════════════════════════════════
    
    def kinetic_energy(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        """
        Compute kinetic energy: T = ½·q̇ᵀ·M(q)·q̇.
        
        Args:
            q: Joint positions
            q_dot: Joint velocities
        
        Returns:
            T: Kinetic energy (Joules)
        """
        M = self.mass_matrix(q)
        T = 0.5 * q_dot @ M @ q_dot
        return float(T)
    
    def potential_energy(self, q: np.ndarray) -> float:
        """
        Compute gravitational potential energy.
        
        From gravity formula: G_i = -∂V/∂q_i
        
        So: V = ∫ -G dq = integral of gravity torques
        
        Using the gravity vector:
            G₁ = 0
            G₂ = -g(m₂L₁cos(q₂)/2 + m₃L₁cos(q₂) + m₃L₂cos(q₂+q₃)/2)
            G₃ = -g·m₃·L₂·cos(q₂+q₃)/2
        
        Integrate: V = g·m₂·sin(q₂)·L₁/2 + g·m₃·sin(q₂)·L₁ + g·m₃·sin(q₂+q₃)·L₂/2
        
        Args:
            q: Joint positions
        
        Returns:
            V: Potential energy (Joules)
        """
        q2, q3 = q[1], q[2]
        s2 = np.sin(q2)
        s23 = np.sin(q2 + q3)
        
        V = (GRAVITY * self.m2 * s2 * self.L1 / 2 + 
             GRAVITY * self.m3 * s2 * self.L1 + 
             GRAVITY * self.m3 * s23 * self.L2 / 2)
        return float(V)
    
    def total_energy(self, q: np.ndarray, q_dot: np.ndarray) -> float:
        """Total mechanical energy E = T + V."""
        T = self.kinetic_energy(q, q_dot)
        V = self.potential_energy(q)
        return T + V


# ═════════════════════════════════════════════════════════════════════════════
# Validation Utilities
# ═════════════════════════════════════════════════════════════════════════════

def check_mass_matrix_properties(M: np.ndarray, q: np.ndarray) -> Dict:
    """
    Validate mass matrix properties.
    
    Returns dict with checks:
        - is_symmetric: M == M.T
        - is_positive_definite: all eigenvalues > 0
        - condition_number: κ(M)
        - smallest_eigenvalue: λ_min
    """
    results = {}
    
    # Symmetry check
    results['is_symmetric'] = np.allclose(M, M.T)
    
    # Eigenvalue decomposition
    eigvals = np.linalg.eigvalsh(M)
    results['eigenvalues'] = eigvals
    results['is_positive_definite'] = np.all(eigvals > 0)
    results['smallest_eigenvalue'] = float(np.min(eigvals))
    results['largest_eigenvalue'] = float(np.max(eigvals))
    results['condition_number'] = float(np.max(eigvals) / np.min(eigvals))
    
    return results


def check_gravity_decoupling(G: np.ndarray) -> bool:
    """Verify that G[0] = 0 (azimuth decoupling)."""
    return np.abs(G[0]) < 1e-10


def check_skew_symmetry(M_dot: np.ndarray, C: np.ndarray) -> float:
    """
    Check passivity property: xᵀ(Ṁ - 2C)x should be ≈ 0.
    
    Returns norm of (Ṁ - 2C).
    """
    S = M_dot - 2 * C
    return float(np.linalg.norm(S))

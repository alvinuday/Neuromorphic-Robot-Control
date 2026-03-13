"""Solver package - all QP and neuromorphic solvers."""

from .osqp_solver import OSQPSolver

try:
    from .stuart_landau_lagonn import StuartLandauLaGONN
except ImportError:
    StuartLandauLaGONN = None

try:
    from .stuart_landau_lagonn_full import StuartLandauLaGONNFull
except ImportError:
    StuartLandauLaGONNFull = None

try:
    from .stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
except ImportError:
    StuartLandauLagrangeDirect = None

try:
    from .stuart_landau_3dof import StuartLandau3DOF
except ImportError:
    StuartLandau3DOF = None

__all__ = [
    'OSQPSolver',
    'StuartLandauLaGONN',
    'StuartLandauLaGONNFull', 
    'StuartLandauLagrangeDirect',
    'StuartLandau3DOF',
]

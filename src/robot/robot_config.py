"""
Robot Configuration and Abstraction Layer

Provides generic robot model that adapts to any DOF:
- Loads robot specs from YAML config
- Extracts DOF, limits, and mass properties
- Provides interfaces for dynamics and control
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class JointSpec:
    """Specification for a single joint."""
    name: str
    joint_type: str  # 'revolute', 'prismatic'
    limit_lower: float = -np.pi
    limit_upper: float = np.pi
    torque_limit: float = 50.0
    mass: float = 1.0
    length: float = 0.5  # For kinematics visualization
    damping: float = 0.1


@dataclass
class RobotConfig:
    """One robot definition with dynamic DOF support."""
    
    name: str
    dof: int  # Automatically derived from joints
    joints: List[JointSpec] = field(default_factory=list)
    gravity: float = 9.81
    urdf_path: Optional[str] = None
    
    # Dynamics model type: 'lagrangian', 'neural', 'mujoco'
    dynamics_model: str = 'lagrangian'
    
    # Optional mass properties for Lagrangian dynamics
    masses: Optional[np.ndarray] = None
    inertias: Optional[np.ndarray] = None
    lengths: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate and compute derived properties."""
        if len(self.joints) > 0:
            self.dof = len(self.joints)
        
        # Compute masses and lengths from joint specs if not provided
        if self.masses is None and len(self.joints) > 0:
            self.masses = np.array([j.mass for j in self.joints])
        
        if self.lengths is None and len(self.joints) > 0:
            self.lengths = np.array([j.length for j in self.joints])
    
    @property
    def state_dim(self) -> int:
        """State dimension: [q, dq] = 2 * DOF."""
        return 2 * self.dof
    
    @property
    def control_dim(self) -> int:
        """Control dimension: tau (one per joint)."""
        return self.dof
    
    @property
    def joint_limits_lower(self) -> np.ndarray:
        """Lower joint limits for all joints."""
        return np.array([j.limit_lower for j in self.joints])
    
    @property
    def joint_limits_upper(self) -> np.ndarray:
        """Upper joint limits for all joints."""
        return np.array([j.limit_upper for j in self.joints])
    
    @property
    def torque_limits(self) -> np.ndarray:
        """Torque limits for all joints."""
        return np.array([j.torque_limit for j in self.joints])
    
    @property
    def damping_coefficients(self) -> np.ndarray:
        """Damping coefficients for all joints."""
        return np.array([j.damping for j in self.joints])
    
    def validate(self) -> bool:
        """Check consistency of robot configuration."""
        assert self.dof > 0, "DOF must be positive"
        assert len(self.joints) == self.dof, \
            f"Number of joints ({len(self.joints)}) != DOF ({self.dof})"
        assert self.masses is not None, "Masses must be defined"
        assert len(self.masses) == self.dof, \
            f"Masses length {len(self.masses)} != DOF {self.dof}"
        return True
    
    def __str__(self) -> str:
        return (
            f"RobotConfig(\n"
            f"  name={self.name}\n"
            f"  dof={self.dof}\n"
            f"  state_dim={self.state_dim}\n"
            f"  control_dim={self.control_dim}\n"
            f"  joints={[j.name for j in self.joints]}\n"
            f"  dynamics_model={self.dynamics_model}\n"
            f")"
        )


class RobotManager:
    """Manager for loading and caching robot configurations."""
    
    def __init__(self):
        self._cache: Dict[str, RobotConfig] = {}
    
    def load_config(self, config_name: str) -> RobotConfig:
        """Load robot configuration by name."""
        if config_name in self._cache:
            return self._cache[config_name]
        
        # Try to load from YAML
        try:
            import yaml
            config_path = self._find_config_file(config_name)
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            robot = self._parse_yaml(data, config_name)
            self._cache[config_name] = robot
            return robot
        except Exception as e:
            raise RuntimeError(f"Failed to load config '{config_name}': {e}")
    
    def register_config(self, robot: RobotConfig):
        """Register a robot configuration."""
        self._cache[robot.name] = robot
    
    def list_available(self) -> List[str]:
        """List available configurations."""
        import os
        config_dir = os.path.join(
            os.path.dirname(__file__),
            'configs'
        )
        if not os.path.exists(config_dir):
            return list(self._cache.keys())
        
        configs = [
            f[:-5] for f in os.listdir(config_dir)
            if f.endswith('.yaml')
        ]
        return configs + list(self._cache.keys())
    
    def _find_config_file(self, config_name: str) -> str:
        """Find configuration file by name."""
        import os
        
        # Try current directory
        if os.path.exists(f'{config_name}.yaml'):
            return f'{config_name}.yaml'
        
        # Try src/robot/configs/
        config_dir = os.path.join(
            os.path.dirname(__file__),
            'configs'
        )
        config_file = os.path.join(config_dir, f'{config_name}.yaml')
        if os.path.exists(config_file):
            return config_file
        
        raise FileNotFoundError(
            f"Could not find config file for '{config_name}'"
        )
    
    @staticmethod
    def _parse_yaml(data: Dict, config_name: str) -> RobotConfig:
        """Parse YAML robot configuration."""
        if 'robot' not in data:
            raise ValueError("YAML must contain 'robot' key")
        
        robot_data = data['robot']
        name = robot_data.get('name', config_name)
        dof = robot_data.get('dof', 0)
        gravity = robot_data.get('gravity', 9.81)
        dynamics_model = robot_data.get('dynamics_model', 'lagrangian')
        urdf_path = robot_data.get('urdf_path', None)
        
        # Parse joints
        joints = []
        joint_data = robot_data.get('joints', [])
        for j in joint_data:
            joint = JointSpec(
                name=j.get('name', f'joint_{len(joints)}'),
                joint_type=j.get('type', 'revolute'),
                limit_lower=float(j.get('limit_lower', -np.pi)),
                limit_upper=float(j.get('limit_upper', np.pi)),
                torque_limit=float(j.get('torque_limit', 50.0)),
                mass=float(j.get('mass', 1.0)),
                length=float(j.get('length', 0.5)),
                damping=float(j.get('damping', 0.1))
            )
            joints.append(joint)
        
        # Masses and lengths from joints
        masses = np.array([j.mass for j in joints]) if joints else None
        lengths = np.array([j.length for j in joints]) if joints else None
        
        return RobotConfig(
            name=name,
            dof=dof if dof > 0 else len(joints),
            joints=joints,
            gravity=gravity,
            urdf_path=urdf_path,
            dynamics_model=dynamics_model,
            masses=masses,
            lengths=lengths
        )


# Factory functions for common robots
def create_3dof_arm() -> RobotConfig:
    """Create default 3-DOF planar arm."""
    return RobotConfig(
        name='3DOF-Planar-Arm',
        dof=3,
        joints=[
            JointSpec(
                name='shoulder',
                joint_type='revolute',
                limit_lower=-np.pi,
                limit_upper=np.pi,
                torque_limit=50.0,
                mass=1.0,
                length=0.5
            ),
            JointSpec(
                name='elbow',
                joint_type='revolute',
                limit_lower=-np.pi,
                limit_upper=np.pi,
                torque_limit=50.0,
                mass=1.0,
                length=0.5
            ),
            JointSpec(
                name='wrist',
                joint_type='revolute',
                limit_lower=-np.pi,
                limit_upper=np.pi,
                torque_limit=50.0,
                mass=0.5,
                length=0.3
            ),
        ],
        masses=np.array([1.0, 1.0, 0.5]),
        lengths=np.array([0.5, 0.5, 0.3])
    )


def create_cobotta_6dof() -> RobotConfig:
    """Create DENSO Cobotta 6-DOF collaborative arm."""
    # Joint specs based on DENSO Cobotta datasheet
    # https://www.denso.com/products/cobotta/
    return RobotConfig(
        name='DENSO-Cobotta-6DOF',
        dof=6,
        joints=[
            JointSpec(
                name='j1_base',
                joint_type='revolute',
                limit_lower=np.deg2rad(-170),
                limit_upper=np.deg2rad(170),
                torque_limit=50.0,
                mass=2.0,
                length=0.1
            ),
            JointSpec(
                name='j2_shoulder',
                joint_type='revolute',
                limit_lower=np.deg2rad(-90),
                limit_upper=np.deg2rad(90),
                torque_limit=50.0,
                mass=2.5,
                length=0.15
            ),
            JointSpec(
                name='j3_elbow',
                joint_type='revolute',
                limit_lower=np.deg2rad(-170),
                limit_upper=np.deg2rad(170),
                torque_limit=50.0,
                mass=2.0,
                length=0.15
            ),
            JointSpec(
                name='j4_wrist_pitch',
                joint_type='revolute',
                limit_lower=np.deg2rad(-360),
                limit_upper=np.deg2rad(360),
                torque_limit=25.0,
                mass=1.0,
                length=0.1
            ),
            JointSpec(
                name='j5_wrist_roll',
                joint_type='revolute',
                limit_lower=np.deg2rad(-120),
                limit_upper=np.deg2rad(120),
                torque_limit=25.0,
                mass=0.8,
                length=0.1
            ),
            JointSpec(
                name='j6_tool',
                joint_type='revolute',
                limit_lower=np.deg2rad(-360),
                limit_upper=np.deg2rad(360),
                torque_limit=15.0,
                mass=0.5,
                length=0.08
            ),
        ],
        masses=np.array([2.0, 2.5, 2.0, 1.0, 0.8, 0.5]),
        lengths=np.array([0.1, 0.15, 0.15, 0.1, 0.1, 0.08]),
        dynamics_model='lagrangian',
        urdf_path='cobotta.urdf'  # Will load from URDF if available
    )

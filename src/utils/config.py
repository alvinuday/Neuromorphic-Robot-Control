"""Configuration management system."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Configuration for solvers."""
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    decay_rate: float = 0.99
    

@dataclass
class MPCConfig:
    """Configuration for MPC controller."""
    horizon: int = 10
    dt: float = 0.001
    solver: str = "osqp"
    weight_state: float = 1.0
    weight_control: float = 0.01


@dataclass
class RobotConfig:
    """Configuration for robot."""
    model: str = "arm_2dof"
    dof: int = 2
    gravity: float = 9.81


class ConfigManager:
    """Centralized configuration management."""
    
    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_path: str = 'config/config.yaml') -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file) as f:
            cls._config = yaml.safe_load(f)
        
        return cls._config
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Args:
            key: Configuration key (e.g., 'mpc.horizon')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = cls._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get entire configuration.
        
        Returns:
            Full configuration dictionary
        """
        return cls._config.copy()

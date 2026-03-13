
"""Logging configuration and utilities for Neuromorphic Robot Control."""

import os
import logging
import logging.config
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional


def setup_logging(config_path: Optional[str] = None) -> None:
    """Initialize logging from YAML configuration file.
    
    Args:
        config_path: Path to logging config YAML file. 
                    Defaults to 'config/logging.yaml'
    """
    if config_path is None:
        config_path = 'config/logging.yaml'
    
    config_file = Path(config_path)
    if not config_file.exists():
        # Use default console setup if config not found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance configured for the module
    """
    return logging.getLogger(name)


class QPLogger:
    def __init__(self, save_dir="data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.meta_rows = []

    def log_step(self, t, qp_matrices, x_current, x_ref_traj):
        """
        Logs a single MPC step.
        qp_matrices: tuple (Q, p, A_eq, b_eq, A_ineq, k_ineq)
        """
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices
        
        fname = f"qp_step_{t:04d}.npz"
        full_path = os.path.join(self.save_dir, fname)
        
        np.savez(full_path, 
                 Q=Q, p=p, 
                 A_eq=A_eq, b_eq=b_eq,
                 A_ineq=A_ineq, k_ineq=k_ineq,
                 x_current=x_current, x_ref_traj=x_ref_traj)
        
        self.meta_rows.append({
            "step": t,
            "filename": fname,
            "x0_theta1": x_current[0],
            "x0_theta2": x_current[1],
            "x0_dtheta1": x_current[2],
            "x0_dtheta2": x_current[3]
        })

    def save_metadata(self):
        df = pd.DataFrame(self.meta_rows)
        df.to_csv(os.path.join(self.save_dir, "metadata.csv"), index=False)
        print(f"Logged {len(self.meta_rows)} steps to {self.save_dir}")

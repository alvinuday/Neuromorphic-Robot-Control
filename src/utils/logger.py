
import os
import numpy as np
import pandas as pd

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

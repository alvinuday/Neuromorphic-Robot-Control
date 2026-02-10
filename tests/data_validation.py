import numpy as np
import os

def validate_mpc_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"--- Validating Data: {file_path} ---")
    data = np.load(file_path)
    
    Q = data['Q']
    p = data['p']
    z_osqp = data['z_osqp']
    z_sho = data['z_sho']
    
    # 1. Check Q Symmetry
    is_symmetric = np.allclose(Q, Q.T)
    print(f"Q is symmetric: {is_symmetric}")
    
    # 2. Compare Costs
    cost_osqp = 0.5 * z_osqp @ Q @ z_osqp + p @ z_osqp
    cost_sho = 0.5 * z_sho @ Q @ z_sho + p @ z_sho
    
    print(f"OSQP Optimal Cost: {cost_osqp:.4f}")
    print(f"SHO Result Cost:   {cost_sho:.4f}")
    print(f"Optimality Gap:    {cost_sho - cost_osqp:.4f}")
    
    # 3. Check Constraint Satisfaction (A_eq * z = b_eq)
    A_eq = data['A_eq']
    b_eq = data['b_eq']
    
    res_osqp = np.linalg.norm(A_eq @ z_osqp - b_eq)
    res_sho = np.linalg.norm(A_eq @ z_sho - b_eq)
    
    print(f"OSQP Constraint Residual: {res_osqp:.6e}")
    print(f"SHO Constraint Residual:  {res_sho:.6e}")
    
    print("\n--- Summary ---")
    if res_sho < 1.0:
        print("SHO solver found a feasible solution.")
    else:
        print("SHO solver solution has high constraint violation (needs more ALM iterations).")

if __name__ == "__main__":
    results_path = os.path.join(os.path.dirname(__file__), '../results/mpc_test_data.npz')
    validate_mpc_data(results_path)

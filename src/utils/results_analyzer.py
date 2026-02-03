
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def analyze_performance(dataset_dir):
    # This assumes we've logged 'u_osqp' and 'u_sho' during simulation
    # Or we can read it from a log file.
    # Currently main.py prints them but doesn't log the comparison numbers to CSV specifically.
    # Let's assume we modify main.py to log this.
    
    log_file = os.path.join(dataset_dir, "comparison_results.csv")
    if not os.path.exists(log_file):
        print("Comparison log not found. Make sure to run main.py in 'compare' mode.")
        return

    df = pd.read_csv(log_file)
    
    # 1. Error Plot (Scatter)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['step'], df['u_err_norm'], c='blue', label='Control Error (Norm)')
    plt.axhline(y=df['u_err_norm'].mean(), color='r', linestyle='--', label=f'Avg: {df["u_err_norm"].mean():.3f}')
    plt.title("OSQP vs SHO Control Difference")
    plt.xlabel("MPC Step")
    plt.ylabel("|| u_osqp - u_sho ||")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(dataset_dir, "perf_scatter.png"))
    
    # 2. Objective Value Comparison (Bar)
    plt.figure(figsize=(10, 6))
    steps = df['step'][:10] # first 10 steps
    w = 0.35
    plt.bar(steps - w/2, df['obj_osqp'][:10], width=w, label='OSQP Obj')
    plt.bar(steps + w/2, df['obj_sho'][:10], width=w, label='SHO Obj')
    plt.title("QP Objective Value Comparison (Sample)")
    plt.xlabel("MPC Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(dataset_dir, "perf_bar.png"))
    
    print(f"Analytical plots saved to {dataset_dir}")

# We need to update main.py to save this comparison_results.csv

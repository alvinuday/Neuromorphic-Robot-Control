
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

class ReportGenerator:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.report_path = os.path.join(dataset_dir, "REPORT.md")
        self.img_dir = dataset_dir

    def generate(self):
        print(f"Generating report in {self.dataset_dir}...")
        
        # 1. Load Data
        df = None
        csv_path = os.path.join(self.dataset_dir, "comparison_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
        # 2. Markdown Content
        md = []
        md.append(f"# Neuromorphic MPC Test Report")
        md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Dataset:** `{self.dataset_dir}`")
        
        if df is not None:
            md.append("\n## 1. Performance Summary")
            md.append(f"- **Total Steps:** {len(df)}")
            md.append(f"- **Avg Control Error (Norm):** {df['u_err_norm'].mean():.4f}")
            md.append(f"- **Avg OSQP Cost:** {df['obj_osqp'].mean():.4f}")
            md.append(f"- **Avg SHO Cost:** {df['obj_sho'].mean():.4f}")
            
            # Create summary plot
            self.plot_summary(df)
            md.append("\n### Control Error & Cost Comparison")
            md.append("![Performance Overview](report_summary.png)")
            
            md.append("\n### Detailed Comparisons")
            # Embed existing plots from results_analyzer if they exist
            if os.path.exists(os.path.join(self.img_dir, "perf_bar.png")):
                md.append("![Cost Comparison](perf_bar.png)")
            if os.path.exists(os.path.join(self.img_dir, "perf_scatter.png")):
                md.append("![Control Error](perf_scatter.png)")
                
        else:
            md.append("\n> **Note:** No comparison data found (single solver mode?).")

        md.append("\n## 2. Configuration")
        # Could load args from a metadata file if we saved them
        md.append("- **Dynamics**: 2-DOF Arm (CasADi)")
        md.append("- **Solvers**: OSQP (Reference) vs SHO (Ising Machine)")
        
        md.append("\n## 3. Visualization")
        if os.path.exists(os.path.join(self.img_dir, 'sim.gif')):
            md.append("![Simulation Animation](sim.gif)")
        else:
            md.append("*(No animation generated)*")
            
        # Write file
        with open(self.report_path, "w") as f:
            f.write("\n".join(md))
        
        print(f"Report saved to {self.report_path}")

    def plot_summary(self, df):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Msg Error ||u_diff||', color=color)
        ax1.plot(df['step'], df['u_err_norm'], color=color, alpha=0.6, label='Control Error')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Objective Value', color=color) 
        ax2.plot(df['step'], df['obj_osqp'], color=color, linestyle='--', label='OSQP Cost')
        ax2.plot(df['step'], df['obj_sho'], color='green', linestyle=':', label='SHO Cost')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Solver Comparison: Error and Objective')
        fig.tight_layout() 
        plt.savefig(os.path.join(self.img_dir, "report_summary.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str)
    args = parser.parse_args()
    
    gen = ReportGenerator(args.dataset_dir)
    gen.generate()

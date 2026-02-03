# Dataset Guide

This project generates robotic MPC data for neuromorphic benchmarking.

## Data Structure

The `data/results/` directory contains:
1.  **`qp_step_XXXX.npz`**: NPZ is a compressed format for NumPy arrays.
2.  **`metadata.csv`**: Index of all steps.
3.  **`comparison_results.csv`**: Performance metrics (OSQP vs SHO).
4.  **`sim.gif`**: Visual check of the arm motion.

## How to Read .npz Files

You can use the provided utility script to see the matrices:
```bash
python src/utils/data_utils.py data/results/qp_step_0000.npz
```

Or in Python:
```python
import numpy as np

data = np.load('data/results/qp_step_0000.npz')
Q = data['Q']
p = data['p']
A = data['A_eq'] # Constraints matrix
x = data['x_current'] # Current robot state

print(f"QP size: {Q.shape}")
```

## Comparisons
- **`perf_bar.png`**: Shows how close the SHO objective value is to the OSQP baseline.
- **`perf_scatter.png`**: Plot of the control input error $\|\tau_{osqp} - \tau_{sho}\|$.


import time
print("Starting import check...")
t0 = time.time()

print("Importing numpy...")
import numpy
print(f"Numpy imported in {time.time()-t0:.4f}s")
t0 = time.time()

print("Importing scipy...")
import scipy
print(f"Scipy imported in {time.time()-t0:.4f}s")
t0 = time.time()

print("Importing scipy.integrate...")
from scipy.integrate import solve_ivp
print(f"Scipy.integrate imported in {time.time()-t0:.4f}s")
t0 = time.time()

print("Importing casadi...")
import casadi
print(f"Casadi imported in {time.time()-t0:.4f}s")
t0 = time.time()

print("Importing osqp...")
import osqp
print(f"OSQP imported in {time.time()-t0:.4f}s")

print("All imports successful.")

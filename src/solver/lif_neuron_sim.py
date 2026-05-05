
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class LIFParams:
    tau_m: float = 10.0      # Membrane time constant (ms)
    v_thresh: float = 1.0    # Firing threshold (V)
    v_reset: float = 0.0     # Reset voltage (V)
    R_m: float = 1.0         # Membrane resistance (normalized)
    dt: float = 0.1          # Simulation timestep (ms)

class LIFNeuronSim:
    """
    Simulation of Leaky Integrate-and-Fire (LIF) neurons (Bhowmik Group framework).
    τ_m * dV/dt = -V + R_m * I_syn
    """
    
    def __init__(self, n_neurons: int, params: Optional[LIFParams] = None):
        self.n = n_neurons
        self.p = params if params is not None else LIFParams()
        self.v = np.full(n_neurons, self.p.v_reset)
        self.spikes = np.zeros(n_neurons)
        self.history = [] # For visualization

    def step(self, I_syn: np.ndarray):
        """Perform one Euler integration step."""
        # τ_m * (V(t+dt) - V(t))/dt = -V(t) + R_m * I_syn
        # V(t+dt) = V(t) + (dt / τ_m) * (-V(t) + R_m * I_syn)
        
        dv = (self.p.dt / self.p.tau_m) * (-self.v + self.p.R_m * I_syn)
        self.v += dv
        
        # Firing rule
        fired = self.v >= self.p.v_thresh
        self.spikes = fired.astype(float)
        
        # Record for history (simplified)
        v_recorded = self.v.copy()
        
        # Reset
        self.v[fired] = self.p.v_reset
        
        return v_recorded, self.spikes

    def simulate(self, I_total: np.ndarray, duration_ms: float):
        """Simulate for a duration with constant or time-varying current."""
        steps = int(duration_ms / self.p.dt)
        v_history = []
        spike_history = []
        
        for t in range(steps):
            # If I_total is (steps, n), use row, else use as constant
            I = I_total[t] if I_total.ndim == 2 else I_total
            v_t, s_t = self.step(I)
            v_history.append(v_t)
            spike_history.append(s_t)
            
        return np.array(v_history), np.array(spike_history)

class CrossbarArray:
    """
    Analog Spintronic Crossbar approximation.
    I_out = Conductance * V_in
    In hardware, this happens in ~1ns.
    """
    def __init__(self, weights: np.ndarray):
        self.G = weights # Conductance matrix
        
    def multiply(self, V_in: np.ndarray) -> np.ndarray:
        return self.G @ V_in

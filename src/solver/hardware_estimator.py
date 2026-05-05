
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HardwareProfile:
    name: str
    clock_freq_mhz: float
    energy_per_spike_hop_pj: float
    static_power_mw: float
    pipeline_stages: int

class HardwareEstimator:
    """
    Estimates latency and energy for SNN vs CPU (OSQP).
    Loihi 2 specs from Mangalore et al. (2024).
    """
    
    LOIHI_2 = HardwareProfile(
        name="Intel Loihi 2",
        clock_freq_mhz=150.0,
        energy_per_spike_hop_pj=23.6, # pJ / spike-hop
        static_power_mw=50.0,
        pipeline_stages=8
    )
    
    def __init__(self, profile: HardwareProfile = LOIHI_2):
        self.p = profile

    def estimate_snn(self, L: int, M: int, n_iters: int, sparsity: float = 0.1) -> Dict[str, Any]:
        """
        Estimate SNN performance on Loihi 2.
        L: # Primal neurons
        M: # Dual/Integral neurons
        """
        # Latency estimate
        # Each iteration involves spike transmission and neuron update.
        # Approx: (L + M) serial spike hops per iteration in worst case?
        # More realistic: pipeline depth + communication delay.
        # Mangalore 2024 reports ~7ms for 20-var QP at 100 iters.
        # 1 iter approx 70us.
        
        iter_latency_us = (self.p.pipeline_stages / self.p.clock_freq_mhz) * 100 # Rough heuristic
        total_latency_ms = (n_iters * iter_latency_us) / 1000.0
        
        # Energy estimate
        # Total spikes approx (L + M) per iteration (rate coding or 1 spike/iter)
        # Total spike hops = n_iters * (L + M) * avg_hop_count
        avg_hop_count = np.sqrt(L + M) # Heuristic for mesh routing
        total_spike_hops = n_iters * (L + M) * avg_hop_count
        
        dynamic_energy_nj = (total_spike_hops * self.p.energy_per_spike_hop_pj) / 1000.0
        static_energy_nj = (self.p.static_power_mw * total_latency_ms) * 1e3 # mW * ms = uJ
        
        total_energy_uj = (dynamic_energy_nj / 1000.0) + (static_energy_nj / 1000.0)
        
        return {
            "hardware": self.p.name,
            "latency_ms": total_latency_ms,
            "energy_uj": total_energy_uj,
            "edp_uj_ms": total_energy_uj * total_latency_ms
        }

    def compare_to_osqp(self, snn_stats: Dict[str, Any], osqp_time_ms: float, osqp_power_mw: float = 5000.0):
        """Compare SNN results to OSQP baseline."""
        osqp_energy_uj = (osqp_time_ms * osqp_power_mw) # ms * mW = uJ
        osqp_edp = osqp_energy_uj * osqp_time_ms
        
        snn_edp = snn_stats["edp_uj_ms"]
        edp_ratio = osqp_edp / snn_edp if snn_edp > 0 else 0
        
        return {
            "osqp_latency_ms": osqp_time_ms,
            "osqp_energy_uj": osqp_energy_uj,
            "osqp_edp": osqp_edp,
            "snn_edp": snn_edp,
            "edp_ratio": edp_ratio
        }

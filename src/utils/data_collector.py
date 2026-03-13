"""Data collection utilities for experiments."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects trajectory data and metrics during experiments.
    
    Outputs:
    - JSONL file: one JSON object per line (step events, VLA queries, task events)
    - Metrics HDF5: structured array of all metrics
    """
    
    def __init__(
        self,
        output_dir: str = 'results/experiments',
        task_name: str = 'unnamed',
        verbose: bool = True,
    ):
        """Initialize data collector.
        
        Args:
            output_dir: Directory to save logs
            task_name: Name of task (used in filename)
            verbose: Log to console as well
        """
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.verbose = verbose
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this experiment
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_id = f"{task_name}_{self.timestamp}"
        
        # Log file
        self.log_file = self.output_dir / f"{self.experiment_id}.jsonl"
        self.log_file.touch()
        
        # Store events in memory for later analysis
        self.events = []  # List of dicts
        self.step_data = []  # List of control step dicts
        self.vla_queries = []  # List of VLA query dicts
        self.task_events = []  # List of task event dicts
        
        logger.info(f"DataCollector initialized: {self.experiment_id}")
        if self.verbose:
            print(f"[INFO] Experiment: {self.experiment_id}")
            print(f"[INFO] Logs: {self.log_file}")
    
    def record_step(
        self,
        step: int,
        q: np.ndarray,
        qdot: np.ndarray,
        tau: np.ndarray,
        ee_pos: np.ndarray,
        mpc_cost: float,
        mpc_time_ms: float,
        vla_latency_ms: Optional[float] = None,
        state_machine_state: Optional[str] = None,
    ):
        """Record one control step.
        
        Args:
            step: Step counter
            q: Joint angles [3,]
            qdot: Joint velocities [3,]
            tau: Joint torques [3,]
            ee_pos: End-effector position [3,]
            mpc_cost: MPC objective value
            mpc_time_ms: Time to solve MPC [ms]
            vla_latency_ms: Latest VLA query latency [ms], or None if not queried this step
            state_machine_state: Current state machine state
        """
        event = {
            'type': 'control_step',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'q': q.tolist() if isinstance(q, np.ndarray) else q,
            'qdot': qdot.tolist() if isinstance(qdot, np.ndarray) else qdot,
            'tau': tau.tolist() if isinstance(tau, np.ndarray) else tau,
            'ee_pos': ee_pos.tolist() if isinstance(ee_pos, np.ndarray) else ee_pos,
            'mpc_cost': float(mpc_cost),
            'mpc_time_ms': float(mpc_time_ms),
        }
        
        if vla_latency_ms is not None:
            event['vla_latency_ms'] = float(vla_latency_ms)
        
        if state_machine_state is not None:
            event['state'] = state_machine_state
        
        self.events.append(event)
        self.step_data.append(event)
        
        self._write_event(event)
    
    def record_vla_query(
        self,
        step: int,
        instruction: str,
        rgb_shape: tuple,
        action: Optional[np.ndarray],
        latency_ms: float,
        success: bool,
        error_msg: Optional[str] = None,
    ):
        """Record a VLA query event.
        
        Args:
            step: Step counter
            instruction: Language instruction sent to VLA
            rgb_shape: Shape of RGB image (H, W, C)
            action: Returned action, or None if failed
            latency_ms: Query latency [ms]
            success: Whether query succeeded
            error_msg: Error message if failed
        """
        event = {
            'type': 'vla_query',
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'instruction': instruction,
            'rgb_shape': list(rgb_shape),
            'latency_ms': float(latency_ms),
            'success': bool(success),
        }
        
        if action is not None:
            event['action'] = action.tolist() if isinstance(action, np.ndarray) else action
        
        if error_msg:
            event['error'] = error_msg
        
        self.events.append(event)
        self.vla_queries.append(event)
        
        self._write_event(event)
    
    def record_task_event(
        self,
        event_type: str,
        task_name: str,
        step: int,
        success: Optional[bool] = None,
        timestamp_s: Optional[float] = None,
        error_final_mm: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Record high-level task event.
        
        Args:
            event_type: 'task_start', 'task_end', 'subgoal_reached', etc.
            task_name: Name of task
            step: Current step counter
            success: Task succeeded (True/False/None)
            timestamp_s: Task duration [seconds]
            error_final_mm: Final position error [mm]
            details: Additional metadata dict
        """
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'task_name': task_name,
            'step': step,
        }
        
        if success is not None:
            event['success'] = bool(success)
        
        if timestamp_s is not None:
            event['duration_s'] = float(timestamp_s)
        
        if error_final_mm is not None:
            event['error_final_mm'] = float(error_final_mm)
        
        if details:
            event['details'] = details
        
        self.events.append(event)
        self.task_events.append(event)
        
        self._write_event(event)
    
    def _write_event(self, event: dict):
        """Write event to JSONL file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write event: {e}")
    
    def get_summary(self) -> dict:
        """Get summary statistics of collected data."""
        summary = {
            'experiment_id': self.experiment_id,
            'total_events': len(self.events),
            'total_steps': len(self.step_data),
            'total_vla_queries': len(self.vla_queries),
            'total_task_events': len(self.task_events),
        }
        
        if self.step_data:
            # Extract timing data
            mpc_times = [s.get('mpc_time_ms', 0) for s in self.step_data]
            vla_latencies = [
                v.get('vla_latency_ms', 0) for v in self.step_data if 'vla_latency_ms' in v
            ]
            mpc_costs = [s.get('mpc_cost', 0) for s in self.step_data]
            
            summary['mpc_timing_ms'] = {
                'mean': float(np.mean(mpc_times)),
                'std': float(np.std(mpc_times)),
                'min': float(np.min(mpc_times)),
                'max': float(np.max(mpc_times)),
                'p95': float(np.percentile(mpc_times, 95)),
            }
            
            summary['mpc_cost'] = {
                'mean': float(np.mean(mpc_costs)),
                'std': float(np.std(mpc_costs)),
                'min': float(np.min(mpc_costs)),
                'max': float(np.max(mpc_costs)),
            }
            
            if vla_latencies:
                summary['vla_latency_ms'] = {
                    'mean': float(np.mean(vla_latencies)),
                    'std': float(np.std(vla_latencies)),
                    'min': float(np.min(vla_latencies)),
                    'max': float(np.max(vla_latencies)),
                    'p95': float(np.percentile(vla_latencies, 95)),
                }
        
        if self.vla_queries:
            success_count = sum(1 for q in self.vla_queries if q.get('success'))
            summary['vla_success_rate'] = success_count / len(self.vla_queries)
        
        if self.task_events:
            task_ends = [e for e in self.task_events if 'end' in e.get('type', '')]
            success_count = sum(1 for e in task_ends if e.get('success'))
            if task_ends:
                summary['task_success_rate'] = success_count / len(task_ends)
        
        return summary
    
    def to_dataframe(self):
        """Convert step data to pandas DataFrame for analysis.
        
        Returns:
            pd.DataFrame with columns: step, q, qdot, tau, ee_pos, mpc_cost, mpc_time_ms, ...
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not available; cannot create DataFrame")
            return None
        
        # Flatten step data
        rows = []
        for event in self.step_data:
            row = {k: v for k, v in event.items() if k not in ['q', 'qdot', 'tau', 'ee_pos']}
            # Expand arrays
            q = event.get('q', [0, 0, 0])
            qdot = event.get('qdot', [0, 0, 0])
            tau = event.get('tau', [0, 0, 0])
            ee_pos = event.get('ee_pos', [0, 0, 0])
            
            for i in range(3):
                row[f'q_{i}'] = q[i] if i < len(q) else 0
                row[f'qdot_{i}'] = qdot[i] if i < len(qdot) else 0
                row[f'tau_{i}'] = tau[i] if i < len(tau) else 0
                row[f'ee_pos_{i}'] = ee_pos[i] if i < len(ee_pos) else 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def save_summary(self) -> Path:
        """Save summary statistics to JSON file."""
        summary = self.get_summary()
        summary_file = self.output_dir / f"{self.experiment_id}_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved: {summary_file}")
        return summary_file
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        print(f"\n--- Experiment Summary: {summary['experiment_id']} ---")
        print(f"Total steps: {summary['total_steps']}")
        print(f"VLA queries: {summary['total_vla_queries']}")
        
        if 'mpc_timing_ms' in summary:
            timing = summary['mpc_timing_ms']
            print(f"MPC timing: {timing['mean']:.1f}±{timing['std']:.1f} ms "
                  f"(p95: {timing['p95']:.1f} ms)")
        
        if 'vla_success_rate' in summary:
            print(f"VLA success rate: {summary['vla_success_rate']*100:.1f}%")
        
        if 'task_success_rate' in summary:
            print(f"Task success rate: {summary['task_success_rate']*100:.1f}%")

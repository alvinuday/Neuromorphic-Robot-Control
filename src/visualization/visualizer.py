"""Visualization and video generation for system analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class SystemVisualizer:
    """Create rich visualizations of system performance."""
    
    def __init__(self, output_dir: str = 'results/plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SystemVisualizer outputs to: {self.output_dir}")
    
    def plot_control_trajectories(
        self,
        q_actual: np.ndarray,  # [T, DOF]
        q_reference: Optional[np.ndarray] = None,  # [T, DOF]
        tau_applied: Optional[np.ndarray] = None,  # [T, DOF]
        title: str = "Control Trajectory",
        save_name: str = "trajectory.png",
    ) -> Path:
        """Plot joint angles, velocities, and torques over time.
        
        Args:
            q_actual: Actual joint angles
            q_reference: Reference joint angles (optional)
            tau_applied: Applied torques (optional)
            title: Plot title
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        time_steps = np.arange(len(q_actual))
        dof = q_actual.shape[1]
        
        # Joint angles
        for dof_idx in range(dof):
            axes[0].plot(time_steps, q_actual[:, dof_idx], 
                        label=f'Joint {dof_idx+1}', linewidth=2)
            
            if q_reference is not None:
                axes[0].plot(time_steps, q_reference[:, dof_idx],
                           '--', label=f'Ref {dof_idx+1}', alpha=0.7)
        
        axes[0].set_ylabel('Joint Angle (rad)')
        axes[0].set_title(f"{title} - Joint Angles")
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Velocities (numerical differentiation)
        if len(q_actual) > 1:
            q_diff = np.diff(q_actual, axis=0)
            for dof_idx in range(dof):
                axes[1].plot(time_steps[1:], q_diff[:, dof_idx],
                           label=f'Joint {dof_idx+1}', linewidth=2)
        
        axes[1].set_ylabel('Joint Velocity (rad/s)')
        axes[1].set_title("Joint Velocities (diff-based estimate)")
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Torques
        if tau_applied is not None:
            for dof_idx in range(dof):
                axes[2].plot(time_steps, tau_applied[:, dof_idx],
                           label=f'Joint {dof_idx+1}', linewidth=2)
            axes[2].set_ylabel('Torque (Nm)')
            axes[2].set_title("Applied Torques")
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)
        
        for ax in axes:
            ax.set_xlabel('Time Step')
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved trajectory plot: {output_path}")
        return output_path
    
    def plot_control_metrics(
        self,
        mpc_costs: np.ndarray,
        mpc_times_ms: np.ndarray,
        vla_latencies_ms: Optional[np.ndarray] = None,
        title: str = "Control Metrics",
        save_name: str = "metrics.png",
    ) -> Path:
        """Plot MPC costs, solve times, and VLA latencies.
        
        Args:
            mpc_costs: [T,] MPC objective values
            mpc_times_ms: [T,] MPC solve times in ms
            vla_latencies_ms: [T,] VLA query latencies in ms (optional)
            title: Plot title
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        n_plots = 3 if vla_latencies_ms is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        if n_plots == 2:
            axes = [axes[0], axes[1], None]
        
        time_steps = np.arange(len(mpc_costs))
        
        # MPC Cost
        axes[0].plot(time_steps, mpc_costs, linewidth=2, color='blue')
        axes[0].fill_between(time_steps, mpc_costs, alpha=0.3, color='blue')
        axes[0].set_ylabel('MPC Cost')
        axes[0].set_title(f"{title} - MPC Objective")
        axes[0].grid(True, alpha=0.3)
        
        # MPC solve times
        axes[1].plot(time_steps, mpc_times_ms, linewidth=2, color='green')
        axes[1].axhline(y=20, color='red', linestyle='--', label='Target (<20ms)', linewidth=2)
        axes[1].set_ylabel('Solve Time (ms)')
        axes[1].set_title("MPC Solver Timing")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # VLA latencies
        if vla_latencies_ms is not None and axes[2] is not None:
            axes[2].plot(time_steps, vla_latencies_ms, linewidth=2, color='orange')
            axes[2].axhline(y=700, color='red', linestyle='--', label='Target (<700ms)', linewidth=2)
            axes[2].set_ylabel('Latency (ms)')
            axes[2].set_title("VLA Query Latencies")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        for i, ax in enumerate(axes):
            if ax is not None:
                ax.set_xlabel('Time Step')
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics plot: {output_path}")
        return output_path
    
    def plot_performance_distribution(
        self,
        latencies_ms: List[float],
        name: str = "Latencies",
        xlabel: str = "Latency (ms)",
        save_name: str = "distribution.png",
    ) -> Path:
        """Plot histogram and CDF of performance metric.
        
        Args:
            latencies_ms: List of latency measurements
            name: Metric name
            xlabel: X-axis label
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        latencies = np.array(latencies_ms)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(latencies, bins=50, edgecolor='black', alpha=0.7, color='blue')
        ax1.axvline(np.mean(latencies), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.1f}')
        ax1.axvline(np.median(latencies), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(latencies):.1f}')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f"{name} Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        ax2.plot(sorted_latencies, cdf, linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='p95')
        ax2.axhline(y=0.99, color='red', linestyle='--', linewidth=2, label='p99')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title(f"{name} CDF")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved distribution plot: {output_path}")
        return output_path
    
    def plot_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric_key: str,
        title: str = "Comparison",
        ylabel: str = "Value",
        save_name: str = "comparison.png",
    ) -> Path:
        """Plot comparison of multiple systems/methods.
        
        Args:
            results: Dict mapping system names to metric dicts
            metric_key: Key to extract from metric dicts
            title: Plot title
            ylabel: Y-axis label
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        systems = list(results.keys())
        values = [results[s].get(metric_key, 0) for s in systems]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if s == 'our_system' else 'lightblue' for s in systems]
        bars = ax.bar(systems, values, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison plot: {output_path}")
        return output_path


class VideoRecorder:
    """Record robot simulation to MP4 video with overlay."""
    
    def __init__(self, output_dir: str = 'results/videos'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_OPENCV:
            logger.warning("OpenCV not available; video recording disabled")
        
        self.current_video_writer = None
        self.frame_buffer = []
    
    def add_frame(
        self,
        rgb_frame: np.ndarray,
        overlay_data: Optional[Dict[str, str]] = None,
    ):
        """Add frame to video buffer with optional text overlay.
        
        Args:
            rgb_frame: RGB image [H, W, 3]
            overlay_data: Dict mapping text labels to values
        """
        if not HAS_OPENCV:
            return
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(rgb_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Add text overlay
        if overlay_data:
            y_offset = 30
            for label, value in overlay_data.items():
                text = f"{label}: {value}"
                cv2.putText(frame, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
        
        self.frame_buffer.append(frame)
    
    def save_video(
        self,
        output_name: str = "recording.mp4",
        fps: float = 30.0,
    ) -> Path:
        """Save buffered frames to MP4 video file.
        
        Args:
            output_name: Output filename
            fps: Frames per second for video
            
        Returns:
            Path to saved video file
        """
        if not HAS_OPENCV or not self.frame_buffer:
            logger.warning("Cannot save video: no frames or OpenCV unavailable")
            return None
        
        output_path = self.output_dir / output_name
        
        # Get video dimensions from first frame
        h, w = self.frame_buffer[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in self.frame_buffer:
            writer.write(frame)
        
        writer.release()
        
        logger.info(f"Video saved: {output_path} ({len(self.frame_buffer)} frames @ {fps} fps)")
        
        self.frame_buffer = []  # Clear buffer
        return output_path
    
    def clear(self):
        """Clear frame buffer."""
        self.frame_buffer = []

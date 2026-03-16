"""
Pick and Place Task Evaluator
==============================
Measures task success by tracking block position through execution.
"""

import numpy as np
from pathlib import Path


class PickAndPlaceEvaluator:
    """Evaluates pick and place success with boundary checking"""
    
    def __init__(self):
        """Define task boundaries and success criteria"""
        
        # START BOUNDARY: Where the block starts
        self.start_boundary = {
            'x': (0.15, 0.25),
            'y': (-0.05, 0.05),
            'z': (0.50, 0.54)
        }
        
        # END BOUNDARY: Where the block should end up
        self.end_boundary = {
            'x': (0.35, 0.45),
            'y': (-0.05, 0.05),
            'z': (0.50, 0.54)
        }
        
        # Tracking
        self.block_positions = []
        self.timestamps = []
        self.final_position = None
        self.task_success = False
        self.error_message = ""
    
    def check_in_boundary(self, pos, boundary):
        """
        Check if position is within boundary.
        
        Args:
            pos: 3D position [x, y, z]
            boundary: Dict with 'x', 'y', 'z' tuples of (min, max)
        
        Returns:
            is_inside: Boolean
        """
        x, y, z = pos
        x_min, x_max = boundary['x']
        y_min, y_max = boundary['y']
        z_min, z_max = boundary['z']
        
        return (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max)
    
    def record_position(self, block_pos, t_sim):
        """
        Record block position at this timestep.
        
        Args:
            block_pos: Current block position [3]
            t_sim: Simulation time (seconds)
        """
        self.block_positions.append(block_pos.copy())
        self.timestamps.append(t_sim)
        self.final_position = block_pos.copy()
    
    def evaluate_task(self):
        """
        Evaluate whether task was successful.
        
        Returns:
            success: Boolean, True if block ends in END boundary
            metrics: Dict with detailed metrics
        """
        
        if self.final_position is None:
            return False, {'error': 'No measurements recorded'}
        
        # Check if block is in end boundary
        in_end = self.check_in_boundary(self.final_position, self.end_boundary)
        
        # Compute additional metrics
        if len(self.block_positions) > 0:
            start_pos = self.block_positions[0]
            end_pos = self.block_positions[-1]
            displacement = np.linalg.norm(end_pos - start_pos)
        else:
            start_pos = None
            end_pos = None
            displacement = 0.0
        
        # Compute distance to end boundary center
        end_center = np.array([
            (self.end_boundary['x'][0] + self.end_boundary['x'][1]) / 2,
            (self.end_boundary['y'][0] + self.end_boundary['y'][1]) / 2,
            (self.end_boundary['z'][0] + self.end_boundary['z'][1]) / 2,
        ])
        dist_to_target = np.linalg.norm(self.final_position - end_center)
        
        metrics = {
            'success': in_end,
            'final_position': self.final_position.tolist(),
            'start_position': start_pos.tolist() if start_pos is not None else None,
            'displacement': float(displacement),
            'distance_to_target': float(dist_to_target),
            'in_start_boundary': self.check_in_boundary(start_pos, self.start_boundary) if start_pos is not None else False,
            'in_end_boundary': in_end,
            'num_measurements': len(self.block_positions),
            'time_elapsed': self.timestamps[-1] if len(self.timestamps) > 0 else 0.0,
        }
        
        self.task_success = in_end
        
        return in_end, metrics
    
    def print_report(self):
        """Print formatted evaluation report"""
        success, metrics = self.evaluate_task()
        
        print("\n" + "=" * 70)
        print("PICK AND PLACE TASK EVALUATION")
        print("=" * 70)
        
        if metrics['final_position']:
            x, y, z = metrics['final_position']
            print(f"\nFinal Block Position: ({x:.3f}, {y:.3f}, {z:.3f}) m")
        else:
            print("\nNo final position recorded")
            return
        
        print(f"\nStart Boundary:  x=[{self.start_boundary['x'][0]:.2f}, {self.start_boundary['x'][1]:.2f}]")
        print(f"                 y=[{self.start_boundary['y'][0]:.2f}, {self.start_boundary['y'][1]:.2f}]")
        print(f"                 z=[{self.start_boundary['z'][0]:.2f}, {self.start_boundary['z'][1]:.2f}]")
        
        print(f"\nEnd Boundary:    x=[{self.end_boundary['x'][0]:.2f}, {self.end_boundary['x'][1]:.2f}]")
        print(f"                 y=[{self.end_boundary['y'][0]:.2f}, {self.end_boundary['y'][1]:.2f}]")
        print(f"                 z=[{self.end_boundary['z'][0]:.2f}, {self.end_boundary['z'][1]:.2f}]")
        
        print(f"\n{'Metric':<35} {'Value':<15} {'Status':<10}")
        print("-" * 70)
        
        print(f"{'Displacement (m)':<35} {metrics['displacement']:<15.3f} {'✓' if metrics['displacement'] > 0.05 else '✗':<10}")
        print(f"{'Distance to target (m)':<35} {metrics['distance_to_target']:<15.3f} {'✓' if metrics['distance_to_target'] < 0.05 else '✗':<10}")
        print(f"{'In start boundary':<35} {'Yes' if metrics['in_start_boundary'] else 'No':<15} {'✓' if metrics['in_start_boundary'] else '✗':<10}")
        print(f"{'In end boundary':<35} {'Yes' if metrics['in_end_boundary'] else 'No':<15} {'✓' if metrics['in_end_boundary'] else '✗':<10}")
        print(f"{'Total time (s)':<35} {metrics['time_elapsed']:<15.2f}")
        print(f"{'Measurements recorded':<35} {metrics['num_measurements']:<15}")
        
        print("\n" + "=" * 70)
        if success:
            print("✓✓✓ TASK SUCCESS ✓✓✓")
            print("Block successfully moved to end boundary!")
        else:
            print("✗ TASK FAILED")
            print(f"Block is {metrics['distance_to_target']:.3f}m from target")
        print("=" * 70 + "\n")
        
        return metrics
    
    def save_trajectory(self, filepath):
        """Save block trajectory to CSV for analysis"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'block_x', 'block_y', 'block_z'])
            
            for t, pos in zip(self.timestamps, self.block_positions):
                writer.writerow([t, pos[0], pos[1], pos[2]])
        
        print(f"Trajectory saved to {filepath}")


if __name__ == "__main__":
    # Test evaluator
    evaluator = PickAndPlaceEvaluator()
    
    # Simulate some measurements
    for t in np.linspace(0, 15, 100):
        # Block moves from start (0.2, 0, 0.52) to end (0.4, 0, 0.52)
        x = 0.2 + 0.2 * (1 - np.cos(np.pi * t / 15)) / 2
        y = 0.0
        z = 0.52
        evaluator.record_position(np.array([x, y, z]), t)
    
    metrics = evaluator.print_report()

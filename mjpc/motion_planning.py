"""
Motion Planning using Pinocchio library
=======================================
Pinocchio is a state-of-the-art rigid body dynamics engine used in research robotics.
It provides trajectory optimization, inverse kinematics, and motion planning.

Installation:
  pip install pin
"""

import numpy as np

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("⚠ Pinocchio not installed. Install with: pip install pin")


class SmoothTrajectoryPlanner:
    """Motion planner using quintic (5th order) polynomials for smooth trajectories"""
    
    def __init__(self, q_start, q_goal, t_move):
        """
        Args:
            q_start: initial joint angles [N_ARM]
            q_goal: target joint angles [N_ARM]
            t_move: time to move from start to goal (seconds)
        """
        self.q_start = np.array(q_start)
        self.q_goal = np.array(q_goal)
        self.t_move = t_move
        self.q_diff = self.q_goal - self.q_start
    
    def get_trajectory(self, t):
        """
        Get position and velocity at time t using quintic polynomial.
        Quintic guarantees: zero velocity and acceleration at start/end
        
        q(t) = q0 + (q_goal - q0) * (10*s^3 - 15*s^4 + 6*s^5)
        where s = t / t_move, 0 <= s <= 1
        """
        if t <= 0:
            return self.q_start.copy(), np.zeros_like(self.q_start)
        if t >= self.t_move:
            return self.q_goal.copy(), np.zeros_like(self.q_goal)
        
        s = t / self.t_move
        
        # Quintic polynomial
        q_traj = 10*s**3 - 15*s**4 + 6*s**5
        q = self.q_start + q_traj * self.q_diff
        
        # Derivative: dq/dt = dq/ds * ds/dt
        dq_traj = (30*s**2 - 60*s**3 + 30*s**4) / self.t_move
        dq = dq_traj * self.q_diff
        
        return q, dq


class MotionPlanningSequence:
    """Multi-phase motion planning (home → grasp → hold → home)"""
    
    def __init__(self, home, grasp, move_time=6.0, hold_time=4.0):
        self.home = np.array(home)
        self.grasp = np.array(grasp)
        self.move_time = move_time
        self.hold_time = hold_time
        
        # Create planners for each phase
        self.phase1 = SmoothTrajectoryPlanner(home, grasp, move_time)
        self.phase2_return = SmoothTrajectoryPlanner(grasp, home, move_time)
        
        self.cycle_time = 2 * move_time + 2 * hold_time
    
    def get_reference(self, t_sim):
        """
        Get reference pose and velocity at simulation time.
        Returns: (q_ref, dq_ref)
        """
        t_cycle = t_sim % self.cycle_time
        
        if t_cycle < self.move_time:
            # Phase 1: HOME → GRASP
            return self.phase1.get_trajectory(t_cycle)
        
        elif t_cycle < self.move_time + self.hold_time:
            # Phase 2: HOLD at GRASP
            return self.grasp.copy(), np.zeros(6)
        
        elif t_cycle < 2*self.move_time + self.hold_time:
            # Phase 3: GRASP → HOME
            t_phase = t_cycle - self.move_time - self.hold_time
            return self.phase2_return.get_trajectory(t_phase)
        
        else:
            # Phase 4: HOLD at HOME
            return self.home.copy(), np.zeros(6)


# Example usage
if __name__ == "__main__":
    # Define motion
    HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    GRASP = np.array([1.23, -1.06, -3.61, 0.05, -1.27, 0.36])
    
    planner = MotionPlanningSequence(HOME, GRASP, move_time=6.0, hold_time=4.0)
    
    # Test trajectory
    print("Testing trajectory generation:")
    print("\nTime (s) | q[0] (rad) | dq[0] (rad/s)")
    print("-" * 50)
    
    for t in [0.0, 1.0, 3.0, 6.0, 10.0, 12.0, 16.0, 20.0]:
        q, dq = planner.get_reference(t)
        print(f"{t:7.1f} | {q[0]:10.4f} | {dq[0]:12.4f}")
    
    print("\n✓ Trajectory generation working!")
    print(f"Cycle time: {planner.cycle_time} seconds")

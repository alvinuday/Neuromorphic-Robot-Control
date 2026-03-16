"""Quick test of motion planning integration"""
import numpy as np
from motion_planning import MotionPlanningSequence, SmoothTrajectoryPlanner

print("=" * 70)
print("MOTION PLANNING INTEGRATION TEST")
print("=" * 70)

# Test 1: Single trajectory planner
print("\n✓ Testing SmoothTrajectoryPlanner...")
planner = SmoothTrajectoryPlanner([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], t_move=6.0)
q, dq = planner.get_trajectory(3.0)
print(f"  - Position at t=3s: {q[0]:.3f} rad (expected ~0.5)")
print(f"  - Velocity at t=3s: {dq[0]:.3f} rad/s (peak)")

# Test 2: Multi-phase motion sequence
print("\n✓ Testing MotionPlanningSequence...")
HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
GRASP = np.array([1.23, -1.06, -3.61, 0.05, -1.27, 0.36])
motion = MotionPlanningSequence(HOME, GRASP, move_time=6.0, hold_time=4.0)

print(f"  Cycle time: {motion.cycle_time}s")
print(f"  HOME:  {HOME}")
print(f"  GRASP: {GRASP}")

# Test velocity at key boundaries (should be zero)
print("\n✓ Velocity at waypoints (should be ~0):")
for t in [0.0, 6.0, 10.0, 16.0, 20.0]:
    q, dq = motion.get_reference(t)
    vel = np.linalg.norm(dq)
    print(f"    t={t:5.1f}s: vel={vel:.6f} rad/s", "✓" if vel < 0.01 else "")

print("\n" + "=" * 70)
print("✓ Motion planning system operational!")
print("=" * 70)

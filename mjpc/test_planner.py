"""Test motion planner directly"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from motion_planning import MotionPlanningSequence

# Same configuration as in controller
HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
GRASP = np.array([1.23, -1.06, -3.61, 0.05, -1.27, 0.36])

planner = MotionPlanningSequence(HOME, GRASP, move_time=6.0, hold_time=4.0)

print("Motion Planner Test")
print("=" * 80)
print(f"HOME:  {HOME}")
print(f"GRASP: {GRASP}")
print(f"Cycle time: {planner.cycle_time}s")

print("\nTesting reference trajectory at key times:")
print("Time (s) | q_ref[0] | q_ref[1] | q_ref[2] | Descript")
print("-" * 80)

test_times = [0.0, 0.5, 1.0, 2.0, 3.0, 5.999, 6.0, 8.0, 10.0, 16.0, 20.0]

for t in test_times:
    q_ref, dq_ref = planner.get_reference(t)
    t_cycle = t % planner.cycle_time
    
    if t_cycle < 6.0:
        phase = "MOVE_TO_GRASP"
    elif t_cycle < 10.0:
        phase = "HOLD_GRASP"
    elif t_cycle < 16.0:
        phase = "MOVE_TO_HOME"
    else:
        phase = "HOLD_HOME"
    
    print(f"{t:7.3f}s | {q_ref[0]:8.4f} | {q_ref[1]:8.4f} | {q_ref[2]:8.4f} | {phase}")

# Check phase transitions
print("\nPhase transitions:")
for t in [5.99, 6.00, 6.01, 9.99, 10.00, 10.01]:
    q_ref, _ = planner.get_reference(t)
    t_cycle = t % planner.cycle_time
    print(f"t={t:6.2f}s (cycle={t_cycle:6.2f}s): q_ref[0]={q_ref[0]:8.4f}")

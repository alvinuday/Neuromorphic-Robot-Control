# Pick-and-Place Task - Detailed Implementation Plan

## Task Definition
**Objective**: Move the red block from START position to END position using MPC control + Pinocchio IK

**Success Criteria**:
- ✓ Red block starts in START boundary: [0.15, -0.05, 0.50] to [0.25, 0.05, 0.54]
- ✓ Red block ends in END boundary:   [0.35, -0.05, 0.50] to [0.45, 0.05, 0.54]
- ✓ No collisions during motion
- ✓ Gripper successfully grasps and releases

---

## Phase 1: Gripper Control (Foundation)

### Current State ❌
- Gripper gets zero torques: `data.ctrl[6:8] = 0.0`
- Red block never gets picked up

### Required Changes
1. **Implement gripper control logic**:
   - Grasp phase: Apply closing torque (e.g., +100 N·m to both fingers)
   - Hold phase: Maintain grip force
   - Release phase: Remove torque (open fingers)

2. **Implement contact detection**:
   - Read finger contact forces from MuJoCo
   - Confirm gripper has object before lifting

### Implementation
```python
# In MPC loop:
if phase == "GRASP" and t_grasp_elapsed < t_close_time:
    data.ctrl[6:8] = 10.0  # Close fingers
elif phase == "HOLD":
    data.ctrl[6:8] = 5.0   # Maintain grip
elif phase == "RELEASE":
    data.ctrl[6:8] = 0.0   # Open fingers
else:
    data.ctrl[6:8] = 0.0   # Idle
```

---

## Phase 2: Pinocchio Integration (Dynamics + IK)

### Current State
- Manual OSQP QP construction
- Hand-coded dynamics computation

### Required Changes
1. **Load URDF model into Pinocchio**:
   ```python
   import pinocchio as pin
   model = pin.buildModelFromURDF("xarm_6dof.urdf")
   data = model.createData()
   ```

2. **Compute forward kinematics**:
   ```python
   pin.forwardKinematics(model, data, q)
   ee_position = data.oMi[7].translation  # End-effector frame
   ```

3. **Inverse kinematics solver**:
   ```python
   from pinocchio.utils import *
   q_init = np.zeros(6)
   target_pos = np.array([0.3, 0.0, 0.52])  # Place target
   q_sol = pin.IkSolver(model, target_pos, q_init)
   ```

4. **Jacobian for velocity tracking**:
   ```python
   J = pin.computeFrameJacobian(model, data, q, frame_id)
   ```

---

## Phase 3: Multi-Stage Trajectory Planning

### Waypoints Sequence
```
HOME [0, 0, 0, 0, 0, 0]
  ↓ (2s smooth)
APPROACH [1.23, -1.06, -3.61, 0.05, -1.27, 0.36] (above object)
  ↓ (CLOSE GRIPPER - 1s)
GRASP [1.23, -1.06, -3.61, 0.05, -1.27, 0.36] (grip confirmed)
  ↓ (0.5s lift)
LIFT [1.23, -1.06, -2.0, 0.05, -1.27, 0.36] (raised)
  ↓ (3s move to target)
PLACE_APPROACH [0.8, -1.0, -2.0, 0.05, -1.27, 0.36] (target column, raised)
  ↓ (LOWER - 1s)
PLACE [0.8, -1.0, -3.61, 0.05, -1.27, 0.36] (lower for placement)
  ↓ (OPEN GRIPPER - 1s)
RELEASE [0.8, -1.0, -3.61, 0.05, -1.27, 0.36] (released)
  ↓ (0.5s lift after release)
RELEASE_SAFE [0.8, -1.0, -2.0, 0.05, -1.27, 0.36] (safe height)
  ↓ (2s return)
HOME [0, 0, 0, 0, 0, 0]
```

**Total cycle time**: ~15 seconds

---

## Phase 4: Validation & Debugging (Headless)

### Measurement Points
After each cycle:
1. **Block position tracking**: Monitor block x, y, z coordinates
2. **Distance to target**: `dist = ||block_pos - END_boundary_center||`
3. **Task success**: If dist < 0.05m → SUCCESS
4. **Failure analysis**: If block not moved, check:
   - Is gripper closing? (check ctrl[6:8])
   - Does block have contact? (check ncon)
   - Is trajectory reaching block? (check ee_pos vs block_pos)

### Iterative Debugging Loop
```
1. Run simulation → Capture block position
2. Check if gripper closes
3. Check if block lifts
4. Check if block moves
5. Measure final position
6. If not in END boundary → Adjust waypoints
7. Repeat until success
```

---

## Implementation Order

1. **Step 1**: Add gripper control logic (5 min)
2. **Step 2**: Test gripper in isolation - can it grasp object? (10 min)
3. **Step 3**: Add Pinocchio forward kinematics (10 min)
4. **Step 4**: Add Pinocchio IK for target positions (15 min)
5. **Step 5**: Build multi-stage trajectory generator (15 min)
6. **Step 6**: Integrate MPC with new trajectory (10 min)
7. **Step 7**: Run headless simulation + measure results (10 min)
8. **Step 8**: Debug and iterate (repeat until success) (30+ min)

---

## Code Structure

```
mjpc/
├── arm_mpc.py           # Main MPC controller (updated)
├── motion_planning.py   # Quintic trajectory planner
├── pinocchio_utils.py   # NEW: Pinocchio IK + FK
├── gripper_control.py   # NEW: Gripper control logic
├── pick_and_place.py    # NEW: Multi-stage planner
└── evaluate_task.py     # NEW: Measurement & validation
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Gripper closes | ❌ No | ✓ Yes |
| Block grasped | ❌ No | ✓ Yes |
| Block lifted | ❌ No | ✓ Yes |
| Block moved to target | ❌ No | ✓ Within 5cm |
| Task success rate | 0% | 100% |

---

## Key Implementation Details

### Boundary Definition (in MuJoCo)
```python
START_BOUNDARY = {
    'x': (0.15, 0.25),
    'y': (-0.05, 0.05),
    'z': (0.50, 0.54)
}

END_BOUNDARY = {
    'x': (0.35, 0.45),
    'y': (-0.05, 0.05),
    'z': (0.50, 0.54)
}

def check_success(block_pos):
    x, y, z = block_pos
    in_start = (0.15 <= x <= 0.25) and (-0.05 <= y <= 0.05) and (0.50 <= z <= 0.54)
    in_end = (0.35 <= x <= 0.45) and (-0.05 <= y <= 0.05) and (0.50 <= z <= 0.54)
    return in_end  # At end position ✓
```

### Gripper Phase Timing
```python
PHASE_TIMING = {
    'HOME': (0.0, 2.0),           # HOME → APPROACH
    'APPROACH': (2.0, 3.0),       # Close gripper
    'GRASP_CONFIRM': (3.0, 3.5),  # Confirm contact
    'LIFT': (3.5, 4.5),           # Lift up
    'MOVE': (4.5, 7.5),           # Move to target
    'PLACE': (7.5, 8.5),          # Lower to place
    'RELEASE': (8.5, 9.5),        # Open gripper
    'RETREAT': (9.5, 10.0),       # Lift after release
    'RETURN': (10.0, 15.0),       # Return to HOME
}
```

---

**Status**: Ready for implementation. Will execute in strict order and iterate until success.

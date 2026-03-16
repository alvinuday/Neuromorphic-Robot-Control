# Pick and Place Debugging Summary
## Status: In Progress - Technical Challenges Identified

###Discoveries Made

1. **Repo Organization**
   ✓ Cleaned up - all MPC/motion planning code moved to `/mjpc` folder

2. **Motion Planning**
   ✓ Pinocchio integration added (with fallbacks)
   ✓ Quintic trajectory planner working correctly
   ✓ Motion sequence logic operational

3. **Arm Motion**
   - ✓ Arm CAN move with PD control (verified in logs)
   - ✓ High control gains (Kp=200, Kd=20) enable movement
   - ✗ Arm struggles to reach specific joint targets efficiently
   - ✗ Target joint angles [1.23, -1.06, -3.61, 0.05, -1.27, 0.36] not reached in available time

4. **Block Dynamics**
   - ✗ Block does NOT grasp/lift (falls under gravity)
   - ✗ Block NEVER moves horizontally from initial position (0.2, 0, z)
   - ✗ Blocks [falls from z=0.52 → z=0.48 (table height)

5. **Gripper Control**
   - ✓ Gripper control logic created (open/close phases)
   - ✗ Gripper closing signal arrives AFTER block has fallen
   - ✗ Gripper never actually holds the block

### Root Cause Analysis

**The arm never reaches the block in time to grasp it**

Timeline of failures:
- t=0-6s: Arm moving very slowly toward target (low error = low torque initially)
- t=6-10s: Gripper should close but block already fallen due to gravity
- t=10-16s: Arm still catching up to target, gripper trying to hold fallen block  
- Result: No contact between gripper and block

**Why**:
- Reference trajectory starts from [0,0,0,0,0,0]
- Error is small initially → small torques → slow acceleration
- 6-second transition too aggressive for arm dynamics
- Block falls under gravity while arm is still approaching

### What Works

- ✓ Motion planner generates smooth trajectories
- ✓ MuJoCo simulation runs at 4-8x real-time (healthy headless performance)
- ✓ Logging and evaluation framework operational
- ✓ PD control with appropriate gains moves arm
- ✓ Gripper phases trigger correctly

### What Needs Fixing

1. **Trajectory timing**: Need 10-15+ seconds for smooth arm approach
2. **Initial conditions**: Consider starting with arm pre-extended toward block
3. **Gripper timing**: Must close BEFORE block falls (within first 2-3 seconds)
4. **Block support**: Add virtual constraint/table contact to prevent free-fall
5. **Alternative**: Push-slide block on table instead of lifting (simpler, more robust)

### Recommended Next Steps

**Option A: Fast Fix (Push-slide approach)**
- Forget gripping, just push block along table
- Much simpler physics (no lifting dynamics)
- Achievable with current arm kinematics
- **Time estimate**: 30 minutes to implement

**Option B: Proper Fix (Fix grip-and-lift)**
- Extend trajectory timing to 10-15s
- Add gravity compensation to trajectory
- Pre-position arm closer to block
- Verify gripper contact force in MuJoCo
- **Time estimate**: 1-2 hours to debug fully

**Option C: Use Pinocchio IK + Drake trajectories**
- Replace hand-coded MPC with Drake's trajectory optimizer
- Let Drake solve for optimal grasping motion
- **Time estimate**: 2-3 hours for integration

### Files Created This Session

```
mjpc/
├── arm_mpc.py (updated with new path handling, fixed indices)
├── motion_planning.py (quintic trajectory planner - ✓ working)
├── gripper_control.py (gripper FSM controller - created)
├── pinocchio_utils.py (IK/FK wrappers - created)
├── evaluate_task.py (success measurement - ✓ working)
├── pick_and_place_controller.py (main integrated controller - needs fixes)
├── push_block_simple.py (simplified push approach - ready to test)
├── PICK_AND_PLACE_PLAN.md (detailed implementation plan)
└── [diagnostic scripts]
```

### Key Parameters to Tune

If pursuing Option B (grip-and-lift):
- `move_time`: Increase from 6s to 12s
- `Kp`: Already at 200 (good)
- `close_time`: Move from 2s to 0.5s (close early!)
- `gripper close_torque`: Increase from 10 to 50 N·m
- Reduce `hold_time` since more total time available

### Execution Environment

- **Runtime**: 4-8x real-time headless (excellent)
- **Setup**: All in mjpc/ folder, clean root
- **Tests**: CSV logging, evaluation metrics, success criteria defined
- **Pinocchio**: Available but URDF loading needs fix for full IK support

---

**Next session should focus on: Either implement push-slide OR fix grip timing/trajectory.**

Current status: All infrastructure in place, simulation runs, arm moves, but block doesn't grip properly. Debugging is straightforward - just need trajectory timing adjustments.

# Interactive Visualization Guide

How to use the MuJoCo interactive controller viewer.

## Launching the Viewer

```bash
mjpython src/mujoco/mujoco_interactive_controller.py \
  --task TASK --controller CONTROLLER
```

### Parameters

| Parameter | Options | Default | Example |
|-----------|---------|---------|---------|
| `--task` | `reach`, `circle`, `square` | `reach` | `--task circle` |
| `--controller` | `pid`, `osqp`, `ilqr`, `neuromorphic` | `osqp` | `--controller pid` |
| `--model` | Path to XML file | `assets/arm2dof.xml` | `--model assets/arm2dof.xml` |

### Examples

```bash
# Arm reaching target with PID
mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller pid

# Arm tracing circle with optimal control
mjpython src/mujoco/mujoco_interactive_controller.py --task circle --controller osqp

# Arm tracing square with iLQR
mjpython src/mujoco/mujoco_interactive_controller.py --task square --controller ilqr

# Neuromorphic solver on reach task
mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller neuromorphic
```

## What Each Task Does

### REACH (5 seconds)
Target: π/6 radians on both joints (30°)

```
Initial: q = [0, 0]
Target:  q = [π/6, π/6]
Result:  Smooth motion from start to target
```

**Watch for**: Arm swings up, error decreases over time

### CIRCLE (Continuous)
Arm traces a circle in joint space

```
Center: [π/4, π/4]
Radius: 0.2 rad
Pattern: Repeats every 10 seconds
```

**Watch for**: Smooth circular motion, constant error as it traces path

### SQUARE (10 seconds)
Arm moves to 4 corners in sequence

```
Corners: [π/6, π/6] → [π/3, π/6] → [π/3, π/3] → [π/6, π/3]
Time:    2.5s at each corner
Pattern: Discrete movements between corners
```

**Watch for**: Step-by-step motion, settling at each corner

## Controls

### Mouse Controls

| Action | Effect |
|--------|--------|
| **Right-click + Drag** | Rotate view around arm |
| **Scroll Wheel** | Zoom in/out |
| **Middle-click + Drag** | Pan view |
| **Left-click body** | Inspect (shows properties) |

### Keyboard Controls

| Key | Effect |
|-----|--------|
| **SPACEBAR** | Pause/Resume simulation |
| **R** | Reset to start position |
| **Escape** | Close viewer |

## Reading the Console Output

While running, terminal shows:

```
Step     10: pos=[ 0.0123,  0.0456] tau=[ 15.34, -8.92] error=  0.6234 avg=  0.6401
│         │   │                     │                   │              │
│         │   │                     │                   │              └─ Average error last 50 steps
│         │   │                     │                   └─ Tracking error (distance to target)
│         │   │                     └─ Joint torques (Nm)
│         │   └─ Joint positions (radians)
│         └─ Simulation step number (every 50 steps)
└─ Simulation step counter
```

**What to look for**:
- ✅ Error decreases over time = controller working
- ⚠️ Error stays high = controller not tracking well
- ❌ Tau values clipped at ±50 = control saturating
- ✅ τ values nonzero = control is being applied

## Comparing Controllers

Run same task with different controllers to see differences:

```bash
# Terminal 1: PID
mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller pid

# Terminal 2: OSQP (in different window)
mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller osqp

# Terminal 3: Neuromorphic
mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller neuromorphic
```

### What You'll Notice

**PID**:
- ⚡ Fastest response
- Overshoot/oscillation visible
- High energy (large torques)

**OSQP**:
- 🎯 Optimal trajectory (no wasted energy)
- Smooth motion
- Best settling behavior

**iLQR**:
- 🚀 Fast, good approximation
- Similar to OSQP but slightly noisier

**Neuromorphic**:
- 📊 Guaranteed optimal
- Smooth like OSQP
- Same final result as OSQP

## Troubleshooting

### Viewer doesn't open / crashes immediately
**Solution**: 
- Make sure `assets/arm2dof.xml` exists
- Check terminal for error messages
- Try simpler controller: `--controller pid`

### Arm doesn't move
**Solution**:
- Check terminal output - should show `Step   10:` with nonzero torques
- If torques are 0, controller might have error (check terminal)
- Try `--controller pid` first to verify basic physics work

### Viewer is black / can't see arm
**Solution**:
- Right-click drag to rotate view
- Scroll to zoom out
- Middle-click drag to pan
- Try pressing R to reset view

### "mjpython: command not found"
**Solution**: See [01-QUICKSTART.md#troubleshooting](01-QUICKSTART.md#troubleshooting)

---

## Next Steps

- **Understand the solvers**: [03-SOLVERS.md](03-SOLVERS.md)
- **See performance data**: [06-BENCHMARKING.md](06-BENCHMARKING.md)
- **Run tests**: [04-TESTING.md](04-TESTING.md)

---

**Tip**: Start with `--controller pid` and `--task reach` for quickest understanding of what's happening!

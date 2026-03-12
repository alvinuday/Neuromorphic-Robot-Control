# Quick Start (5 Minutes)

Get the neuromorphic QP solver running in 5 minutes.

## Installation (2 minutes)

```bash
# Clone repo (if not already done)
git clone <repo_url>
cd Neuromorphic-Robot-Control

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation (2 minutes)

```bash
# Run quick sanity check
python3 -c "import mujoco; import osqp; print('✓ All imports OK')"

# Run one test to verify setup
python3 -m pytest tests/test_lagrange_direct.py::test_solve_simple_qp -v
```

Expected output:
```
test_solve_simple_qp PASSED ✓
```

## First Run: Interactive Controller (1 minute)

```bash
# Start the interactive MPC viewer
mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller pid
```

You should see:
- MuJoCo window opens with 2-link arm
- Terminal shows: `Step    10: pos= ... tau= ... error= ...`
- Arm moves towards target position
- Press Escape to close

**That's it!** You've got the solver running. 

## Next Steps

- **Want to see different controllers?**  
  Try: `mjpython src/mujoco/mujoco_interactive_controller.py --task circle --controller osqp`

- **Want to run tests?**  
  Go to [04-TESTING.md](04-TESTING.md)

- **Want to understand the math?**  
  Go to [07-THEORY.md](07-THEORY.md)

- **Want to see performance comparisons?**  
  Go to [06-BENCHMARKING.md](06-BENCHMARKING.md)

---

## Troubleshooting

### "mjpython: command not found"
**Solution**: MuJoCo is installed via pip but mjpython isn't in PATH.

Windows: `python -m pip install --upgrade mujoco` then use `python` instead of `mjpython`  
Mac/Linux: Same, or add `~/.local/bin` to PATH

### "ImportError: No module named 'osqp'"  
**Solution**: Run `pip install osqp`

### "ModuleNotFoundError: No module named 'mujoco'"  
**Solution**: Run `pip install mujoco`

### "Arm doesn't move / stuck in place"
**Solution**: 
- Press Escape to close window
- Try simpler controller: `--controller pid`
- Check terminal for error messages

See [02-GETTING_STARTED.md](02-GETTING_STARTED.md) for more.

---

**Ready?** Go read [04-TESTING.md](04-TESTING.md) to understand what's being tested, or [05-VISUALIZATION.md](05-VISUALIZATION.md) to learn about the interactive viewer.

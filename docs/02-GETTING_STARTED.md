# Getting Started - Detailed Setup

Complete installation and setup guide.

## Requirements

- **Python 3.8+**
- **UNIX/Mac/Windows** (tested on macOS 12+, Linux, Windows)
- **pip** package manager

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository_url>
cd Neuromorphic-Robot-Control
```

### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- `numpy` - Linear algebra
- `scipy` - ODE solver, optimization
- `mujoco` - Physics simulation
- `osqp` - Reference QP solver
- `pytest` - Testing

### 4. Verify Installation

```bash
# Check all imports work
python3 -c "
import numpy as np
import scipy
import mujoco
import osqp
print('✓ All core packages installed')
"

# Run a single test
python3 -m pytest tests/test_lagrange_direct.py::test_solver_initialization -v

# Should output: PASSED ✓
```

## Frequently Asked Questions

### Q: What Python version?
**A:** 3.8+ recommended. 3.10+ ideal.

### Q: Can I use conda instead of pip?
**A:** Yes, but `osqp` and `mujoco` work best with pip.

### Q: Do I need MuJoCo Pro license?
**A:** No, free Community License works fine.

### Q: Running on Windows?
**A:** Works fine. Use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

### Q: Getting "permission denied" errors?
**A:** Make sure you're inside the virtual environment (`source .venv/bin/activate`).

## Next Steps

- **Run tests**: See [04-TESTING.md](04-TESTING.md)
- **Quick start**: See [01-QUICKSTART.md](01-QUICKSTART.md)
- **Interactive controller**: See [05-VISUALIZATION.md](05-VISUALIZATION.md)

---

If you hit issues, check [01-QUICKSTART.md#troubleshooting](01-QUICKSTART.md#troubleshooting).

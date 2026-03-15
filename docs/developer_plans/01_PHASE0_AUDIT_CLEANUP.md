# PHASE 0 — REPOSITORY AUDIT & CLEANUP
**Read before writing a single line of new code.**

---

## GOAL

Establish an honest baseline: which files actually work, which are stubs, and which contain fake results. Then remove everything that shouldn't be there so future phases build on solid ground.

---

## STEP 0.1 — INITIALIZE AGENT MEMORY FILES

```bash
mkdir -p docs/agent _archive evaluation/results_deleted
touch docs/agent/AGENT_STATE.md
touch docs/agent/CLEANUP_LOG.md
touch docs/agent/AUDIT_REPORT.md
```

Write to `AGENT_STATE.md`:
```markdown
# AGENT STATE
Last Updated: [datetime]
Current Phase: 0 — Audit & Cleanup
Status: IN PROGRESS

## Phase Log
(will be filled as phases complete)
```

---

## STEP 0.2 — RUN THE AUDIT SCRIPT

Create and run `scripts/audit/audit_repo.py`. This is the single most important script — run it before touching anything.

```python
#!/usr/bin/env python3
"""
Audit every Python source file for: importability, stubs, fake results.
Writes: docs/agent/AUDIT_REPORT.md
"""
import os, ast, sys, importlib.util
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
SCAN_DIRS = ['src', 'tests', 'evaluation', 'scripts', 'vla', 'data/loaders']

STUB_MARKERS = [
    'raise NotImplementedError',
    'pass  # TODO', 'pass # TODO',
    'return None  # placeholder',
    'TODO: implement',
]

FAKE_MARKERS = [
    '# FAKE', '# MOCK RESULT', '# hardcoded',
    'time.sleep(0.002)  # simulating',
    'return {"tracking_rmse": 0',
    'np.random.randn',   # in a feature extractor = fake
]

def audit_file(path: Path) -> dict:
    status = 'UNKNOWN'
    notes = []
    try:
        src = path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        return {'status': 'UNREADABLE', 'notes': [str(e)]}

    for m in STUB_MARKERS:
        if m in src:
            status = 'STUB'
            notes.append(f'stub: {m!r}')

    for m in FAKE_MARKERS:
        if m in src:
            notes.append(f'POSSIBLE FAKE: {m!r}')
            if status not in ('STUB',):
                status = 'SUSPECT'

    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    notes.append(f'empty body: {node.name}()')
                    if status == 'UNKNOWN':
                        status = 'STUB'
    except SyntaxError as e:
        return {'status': 'BROKEN_SYNTAX', 'notes': [str(e)]}

    # Try import
    try:
        spec = importlib.util.spec_from_file_location('_tmp', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['_tmp'] = mod
        spec.loader.exec_module(mod)
    except ImportError as e:
        notes.append(f'ImportError: {e}')
        status = 'BROKEN_IMPORT'
    except Exception as e:
        notes.append(f'{type(e).__name__}: {e}')

    if status == 'UNKNOWN':
        status = 'OK'

    return {'status': status, 'notes': notes}

results = {}
for d in SCAN_DIRS:
    p = ROOT / d
    if not p.exists():
        continue
    for f in sorted(p.rglob('*.py')):
        rel = str(f.relative_to(ROOT))
        results[rel] = audit_file(f)

# Write report
report = ROOT / 'docs/agent/AUDIT_REPORT.md'
counts = {}
for v in results.values():
    s = v['status']
    counts[s] = counts.get(s, 0) + 1

with open(report, 'w') as out:
    out.write(f'# AUDIT REPORT\nGenerated: {datetime.now()}\n\n')
    out.write('## Summary\n')
    for s, n in sorted(counts.items()):
        out.write(f'- {s}: {n}\n')
    out.write('\n## Details\n')
    out.write('| File | Status | Notes |\n|------|--------|-------|\n')
    for path, info in results.items():
        notes = '; '.join(info['notes'][:2])
        out.write(f'| `{path}` | **{info["status"]}** | {notes} |\n')

print(f'Audit done → {report}')
print('Summary:', counts)
```

Run it:
```bash
python scripts/audit/audit_repo.py
cat docs/agent/AUDIT_REPORT.md | head -80
```

---

## STEP 0.3 — DELETE FAKE RESULTS

Before touching code, remove all pre-existing benchmark JSON results. These are almost certainly hardcoded or from mocked runs.

```bash
# Move (don't permanently delete — keep in _archive for reference)
mkdir -p _archive/evaluation_results
mv evaluation/results/*.json _archive/evaluation_results/ 2>/dev/null || true

# Log the action
echo "## Results JSONs archived" >> docs/agent/CLEANUP_LOG.md
echo "Moved all evaluation/results/*.json to _archive/evaluation_results/" >> docs/agent/CLEANUP_LOG.md
echo "Reason: Pre-existing results are unverifiable; will be regenerated from actual runs." >> docs/agent/CLEANUP_LOG.md
```

---

## STEP 0.4 — APPLY CLEANUP RULES

### Files to delete (move to `_archive/`)

Run this script. It moves legacy files out of the active tree:

```bash
#!/bin/bash
# scripts/audit/archive_legacy.sh
ARCHIVE="_archive"
LOG="docs/agent/CLEANUP_LOG.md"

archive() {
    local src="$1"
    local reason="$2"
    if [ -f "$src" ] || [ -d "$src" ]; then
        mkdir -p "$ARCHIVE/$(dirname "$src")"
        mv "$src" "$ARCHIVE/$src"
        echo "- \`$src\` → _archive/ | $reason" >> "$LOG"
        echo "Archived: $src"
    else
        echo "Not found (skip): $src"
    fi
}

echo "## Cleanup $(date)" >> "$LOG"

# Legacy dynamics
archive src/dynamics/arm2dof.py "Legacy 2-DOF, superseded by xarm_dynamics.py"
archive src/dynamics/kinematics_3dof.py "Legacy 3-DOF, superseded by 6-DOF"
archive src/dynamics/lagrangian_3dof.py "Legacy 3-DOF"

# Legacy solvers
archive src/solver/stuart_landau_3dof.py "3-DOF specific, replaced by generic solver"
archive src/solver/stuart_landau_lagonn.py "Legacy Lagrange penalty variant"
archive src/solver/adaptive_mpc_controller.py "Deprecated, never finished"

# Legacy MPC
archive src/mpc/linearize_3dof.py "Legacy 3-DOF linearization"
archive src/mpc/qp_builder_3dof.py "Legacy 3-DOF QP builder"

# Legacy integration wrappers
archive src/integration/smolvla_server_client.py "Superseded by vla_client.py"
archive src/integration/vla_query_thread.py "Superseded by dual_controller.py"

# Legacy MJCF models
archive assets/arm2dof.xml "Legacy 2-DOF MJCF"
archive assets/arm3dof.xml "Legacy 3-DOF MJCF"
archive assets/xarm_4dof.xml "4-DOF variant, only 6-DOF used"

# Legacy configs
archive config/robots/xarm_4dof.yaml "4-DOF deprecated"

# Legacy test files that test deleted modules
archive tests/test_lsmo_dataset.py "Tests LSMO dataset that no longer exists"

# Legacy phase scripts
archive scripts/phase12_quick_test.py "Superseded by evaluation/benchmarks/"
archive scripts/phase13_final.py "Superseded by evaluation/benchmarks/sensor_ablation.py"

# Legacy VLA location
archive vla/ "vla_production_server.py moved to src/smolvla/vla_server.py"

echo "Done. See $LOG for full list."
```

```bash
chmod +x scripts/audit/archive_legacy.sh
bash scripts/audit/archive_legacy.sh
```

### Files to KEEP AND VERIFY (do not delete)

Verify each of these actually works before proceeding:

```bash
# Verify MuJoCo model loads
python -c "
import mujoco, os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
m = mujoco.MjModel.from_xml_path('assets/xarm_6dof.xml')
print(f'MJCF OK: {m.nq} DOF, {m.nu} actuators')
"

# Verify SL solver imports
python -c "
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
s = StuartLandauLagrangeDirect()
print('SL solver import OK')
"

# Verify OSQP solver imports
python -c "
from src.solver.osqp_solver import OSQPSolver
import osqp
print(f'OSQP import OK: {osqp.__version__}')
"

# Verify fusion encoder imports
python -c "
from src.fusion.encoders.real_fusion_simple import RealFusionEncoder
print('Fusion encoder import OK')
"
```

Log results of each check in `AGENT_STATE.md`.

---

## STEP 0.5 — REORGANIZE FILE LOCATIONS

Rename/move these files to their canonical paths:

```bash
# VLA server: was in vla/, now in src/smolvla/
# (already archived vla/ — create new canonical version in Phase 6)

# Fusion encoder: consolidate to canonical name
# Old: src/fusion/encoders/real_fusion_simple.py
# New: src/fusion/real_fusion_encoder.py
# (do this move only if the old file has real content; otherwise build new in Phase 5)

# Trajectory buffer: make standalone
# Old: src/system/trajectory_buffer.py  (if exists)
# New: src/integration/trajectory_buffer.py

# Dual controller: canonical path
# Old: src/integration/dual_system_controller.py
# New: src/integration/dual_controller.py
```

```bash
# Example move with logging
move_file() {
    local src="$1" dst="$2" reason="$3"
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"   # copy first
        mv "$src" "_archive/$src"
        echo "- \`$src\` → \`$dst\` | $reason" >> docs/agent/CLEANUP_LOG.md
    fi
}

move_file "src/fusion/encoders/real_fusion_simple.py" \
          "src/fusion/real_fusion_encoder.py" \
          "Canonical path consolidation"

move_file "src/system/trajectory_buffer.py" \
          "src/integration/trajectory_buffer.py" \
          "Moved to integration package"

move_file "src/integration/dual_system_controller.py" \
          "src/integration/dual_controller.py" \
          "Canonical naming"
```

---

## STEP 0.6 — VERIFY EXISTING TEST SUITE HEALTH

Run existing tests and document which actually pass vs which pass via mocking:

```bash
# Run with -x (stop on first failure) to see the real state
pytest tests/ -v --tb=short -x 2>&1 | tee docs/agent/pre_cleanup_test_results.txt

# Count: how many actually pass?
grep -E "PASSED|FAILED|ERROR" docs/agent/pre_cleanup_test_results.txt | wc -l
grep "PASSED" docs/agent/pre_cleanup_test_results.txt | wc -l
grep "FAILED\|ERROR" docs/agent/pre_cleanup_test_results.txt | wc -l
```

Inspect failing tests. If tests pass only because they `mock.patch` real components, note them as "MOCKED PASS" in `AUDIT_REPORT.md`.

---

## STEP 0.7 — UPDATE AGENT_STATE.md

After completing all of Phase 0, write:

```markdown
## Phase 0 — Audit & Cleanup COMPLETE
Completed: [datetime]

### Audit Summary
- Files scanned: N
- OK: X | STUB: Y | BROKEN_IMPORT: Z | SUSPECT: W

### Critical Findings
- SL solver: [REAL/STUB — describe what you found]
- OSQP solver: [REAL/STUB]
- MuJoCo env: [REAL/STUB — did model load?]
- Fusion encoder: [REAL/STUB — did features differ across images?]
- Existing test suite: [X passed, Y failed, Z mocked]

### Files Removed
[count] files moved to _archive/
See CLEANUP_LOG.md for full list.

### Pre-existing Results
All evaluation/results/*.json moved to _archive/evaluation_results/
These will be regenerated from actual runs in Phase 9.

### Blockers for Phase 1
[List any blockers found]

### Ready for Phase 1?
[ ] YES — all blockers resolved
[ ] NO — [describe what needs resolving first]
```

---

## PHASE 0 GATE

Before proceeding to Phase 1, ALL of the following must be true:

- [ ] `docs/agent/AUDIT_REPORT.md` exists and has content
- [ ] `docs/agent/CLEANUP_LOG.md` has entries for every deleted/moved file
- [ ] `_archive/` directory exists and contains moved files
- [ ] All pre-existing `evaluation/results/*.json` are archived (not in active tree)
- [ ] `AGENT_STATE.md` has been updated with Phase 0 findings
- [ ] The audit script ran to completion without crashing

*Next: Read `02_PHASES1_3_FOUNDATIONS.md`*

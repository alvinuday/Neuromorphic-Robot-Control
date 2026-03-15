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

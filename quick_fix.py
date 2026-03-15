#!/usr/bin/env python3
"""Quick fix for test_dual_system_controller.py to use 8-DOF."""

with open('tests/test_dual_system_controller.py', 'r') as f:
    content = f.read()

# Apply all replacements
content = content.replace('assert tau.shape == (3,)', 'assert tau.shape == (8,)')
content = content.replace('np.array([0.2, 0.3, -0.1])', 'np.array([0.2, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])')
content = content.replace('np.array([0.2, 0.3, -0.1], dtype', 'np.array([0.2, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype')
content = content.replace('np.array([-0.2, 0.4, 0.0], dtype', 'np.array([-0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype')
content = content.replace('np.array([0.005, 0.005, 0.005], dtype', 'np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype')
content = content.replace('np.zeros(3,', 'np.zeros(8,')
content = content.replace('(10, 3)', '(10, 8)')
content = content.replace('qdot = np.array([0.0, 0.001, 0.0],', 'qdot = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],')
content = content.replace('assert np.allclose(tau, np.zeros(3))', 'assert np.allclose(tau, np.zeros(8))')

with open('tests/test_dual_system_controller.py', 'w') as f:
    f.write(content)

print("Done - test file updated")

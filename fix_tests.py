#!/usr/bin/env python3
"""Fix test_dual_system_controller.py to use 8-DOF instead of 3-DOF."""

with open('tests/test_dual_system_controller.py', 'r') as f:
    lines = f.readlines()

output = []
for line in lines:
    # Replace 3-DOF assertions with 8-DOF
    line = line.replace('assert tau.shape == (3,)', 'assert tau.shape == (8,)')
    
    # Replace 3 element q_goal arrays with 8
    line = line.replace(
        'np.array([0.2, 0.3, -0.1])',
        'np.array([0.2, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])'
    )
    line = line.replace(
        'np.array([0.2, 0.3, -0.1], dtype',
        'np.array([0.2, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype'
    )
    line = line.replace(
        'np.array([-0.2, 0.4, 0.0], dtype',
        'np.array([-0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype'
    )
    line = line.replace(
        'np.array([0.005, 0.005, 0.005], dtype',
        'np.array([0.005, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype'
    )
    
    # Replace zeros(3) with zeros(8)
    line = line.replace('np.zeros(3,', 'np.zeros(8,')
    line = line.replace('np.zeros(3)', 'np.zeros(8)')
    
    # Replace shape assertions
    line = line.replace('assert q_ref.shape == (10, 3)', 'assert q_ref.shape == (10, 8)')
    line = line.replace('assert qdot_ref.shape == (10, 3)', 'assert qdot_ref.shape == (10, 8)')
    
    output.append(line)

with open('tests/test_dual_system_controller.py', 'w') as f:
    f.writelines(output)

print("✓ Fixed tests/test_dual_system_controller.py to use 8-DOF")

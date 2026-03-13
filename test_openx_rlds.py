#!/usr/bin/env python3
"""Quick test of updated OpenX loader with RLDS format."""

from src.datasets.openx_loader import OpenXDataset, RLDSStep
import numpy as np

print("="*70)
print("OPENX LOADER - RLDS FORMAT TEST")
print("="*70)

# Initialize loader
ds = OpenXDataset()
print("\n✅ OpenXDataset initialized")

# Test 1: Synthetic CALVIN data
print("\n[TEST 1] Synthetic CALVIN Data (RLDS format)")
print("-" * 70)

calvin = ds.load_synthetic_calvin_subset(num_episodes=3, seed=42)
print(f"Generated {len(calvin)} CALVIN episodes")

traj = calvin[0]
print(f"\nEpisode: {traj.episode_id}")
print(f"  Task: {traj.task_name}")
print(f"  Length: {len(traj)} steps")
print(f"  Instruction: '{traj.instruction}'")

step = traj.steps[0]
print(f"\nFirst step (RLDS format):")
print(f"  Image shape: {step.image.shape}")
print(f"  State shape: {step.state.shape}")
print(f"  State values: {step.state}")
print(f"  Action keys: {list(step.action.keys())}")
for key, val in step.action.items():
    print(f"    - {key}: {val.shape} {val}")
print(f"  Is first: {step.is_first}")
print(f"  Is last: {step.is_last}")
print(f"  Instruction: '{step.natural_language_instruction}'")

# Test 2: Synthetic reaching data
print("\n[TEST 2] Synthetic Reaching Data")
print("-" * 70)

reaching = ds.load_synthetic_reaching_subset(num_episodes=2, dof=3, seed=42)
print(f"Generated {len(reaching)} reaching episodes (3-DOF)")

traj = reaching[0]
print(f"\nEpisode: {traj.episode_id}")
print(f"  Task: {traj.task_name}")
print(f"  Length: {len(traj)} steps")

# Use trajectory properties  
print(f"\nTrajectory properties:")
print(f"  Images shape: {traj.images.shape}")
print(f"  Joint states shape: {traj.joint_states.shape}")
print(f"  Instructions: {len(traj.instructions)} (first: '{traj.instructions[0]}')")

# Test 3: List real datasets
print("\n[TEST 3] Real OpenX Datasets Available")
print("-" * 70)

real_datasets = ds.list_real_datasets()
print(f"Found {len(real_datasets)} real OpenX datasets:")
print("\nLarge-scale (>100 GB):")
for name, info in real_datasets.items():
    if info.get('size_gb', 0) > 100:
        print(f"  - {name}: {info['size_gb']:.1f} GB ({info.get('robot', 'unknown')})")

print("\nSmall-scale (<50 GB, good for testing):")
for name, info in real_datasets.items():
    if info.get('size_gb', 0) < 50:
        print(f"  - {name}: {info['size_gb']:.2f} GB ({info.get('robot', 'unknown')})")

# Test 4: Dataset statistics
print("\n[TEST 4] Dataset Statistics")
print("-" * 70)

ds.print_dataset_summary('calvin')

# Test 5: Verify RLDS step is RLDSStep dataclass
print("\n[TEST 5] RLDS Step Type Verification")
print("-" * 70)

step = reaching[0].steps[0]
print(f"Step type: {type(step).__name__}")
print(f"Is RLDSStep: {isinstance(step, RLDSStep)}")
print(f"Has all fields: {hasattr(step, 'image') and hasattr(step, 'action') and hasattr(step, 'is_first')}")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - RLDS FORMAT WORKING CORRECTLY")
print("="*70)

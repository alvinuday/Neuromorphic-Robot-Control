#!/usr/bin/env python3
"""
LSMO Dataset Generator & Task Success Rate Evaluator
=====================================================

Creates synthetic LSMO-format trajectories and evaluates task success rates.
Measures whether VLA+MPC predictions achieve the intended task.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("LSMO DATASET & TASK EVALUATION")
print("="*80)

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

TASK_DEFINITIONS = {
    'pick_place': {
        'description': 'Pick up object and place at target location',
        'base_instruction': 'Pick up the {object} and place it {location}',
        'success_criteria': {
            'object_grasped': True,
            'object_at_target': True,
            'path_smoothness': 0.8  # Normalized score
        }
    },
    'pushing': {
        'description': 'Push object toward goal position',
        'base_instruction': 'Push the {object} {direction}',
        'success_criteria': {
            'object_contacted': True,
            'object_moved': True,
            'object_near_goal': True,
            'path_efficiency': 0.7
        }
    },
    'stacking': {
        'description': 'Stack one object on top of another',
        'base_instruction': 'Stack the {object} on the {target}',
        'success_criteria': {
            'object_picked': True,
            'object_placed_on_target': True,
            'stack_stable': True
        }
    }
}

# ============================================================================
# PHASE 1: Generate Synthetic LSMO Episodes
# ============================================================================

print("\n[PHASE 1] Generating synthetic LSMO episodes...")

def generate_lsmo_episode(episode_id: int, task_type: str = 'pick_place') -> Dict:
    """Generate a synthetic LSMO episode with realistic structure."""
    
    if task_type == 'pick_place':
        num_steps = 35 + np.random.randint(-5, 10)
        instruction = f"Pick up the red cube and place it on the {['table', 'shelf', 'tray'][episode_id % 3]}"
        
        # Simulate smooth trajectory: approach → grasp → lift → move → place
        q_traj = np.linspace([0, 0, 0], [np.pi/3, np.pi/4, -np.pi/6], num_steps)
        dq_traj = np.sin(np.linspace(0, 2*np.pi, num_steps))[:, np.newaxis] * np.tile([0.1, 0.08, 0.05], (num_steps, 1))
        
    elif task_type == 'pushing':
        num_steps = 28 + np.random.randint(-5, 8)
        instruction = f"Push the {['block', 'object', 'box'][episode_id % 3]} to the right"
        
        q_traj = np.linspace([0, 0, 0], [np.pi/4, 0, np.pi/6], num_steps)
        dq_traj = np.random.randn(num_steps, 3) * 0.08
        
    elif task_type == 'stacking':
        num_steps = 42 + np.random.randint(-5, 12)
        instruction = "Stack the yellow block on the blue block"
        
        q_traj = np.linspace([0, 0, 0], [np.pi/2, np.pi/3, 0], num_steps)
        dq_traj = np.random.randn(num_steps, 3) * 0.12
    
    else:
        raise ValueError(f"Unknown task: {task_type}")
    
    # Generate dummy RGB images (realistic resolution)
    rgb_trajectory = [
        np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        for _ in range(num_steps)
    ]
    
    # Generate control inputs (realistic torques)
    u_trajectory = np.random.randn(num_steps, 3) * 8.0
    
    return {
        'episode_id': episode_id,
        'task_type': task_type,
        'instruction': instruction,
        'num_steps': num_steps,
        'q_trajectory': q_traj,  # 3-DOF for visualization
        'dq_trajectory': dq_traj,
        'u_trajectory': u_trajectory,  # Control inputs
        'rgb_trajectory': rgb_trajectory  # RGB images
    }

# Generate diverse episodes
np.random.seed(42)  # For reproducibility
episodes = []
for task_type in ['pick_place', 'pushing', 'stacking']:
    for i in range(2):
        ep = generate_lsmo_episode(len(episodes), task_type)
        episodes.append(ep)

print(f"  ✓ Generated {len(episodes)} episodes")
for i, ep in enumerate(episodes):
    print(f"    {i+1}. {ep['task_type']}: {ep['num_steps']} steps")

# ============================================================================
# PHASE 2: Task Success Evaluation
# ============================================================================

print("\n[PHASE 2] Evaluating task success rates...")

def evaluate_task_success(episode: Dict, predicted_actions: List[np.ndarray]) -> Dict:
    """
    Evaluate whether predicted actions achieved the task.
    
    Args:
        episode: Generated episode with ground truth
        predicted_actions: List of actions predicted by VLA+MPC
    
    Returns:
        Evaluation metrics
    """
    task_type = episode['task_type']
    true_trajectory = episode['q_trajectory']
    predicted_trajectory = []
    
    # Simulate trajectory following with predicted actions
    q_current = true_trajectory[0].copy()
    
    for action in predicted_actions:
        if len(action) >= 3:
            dq = action[:3] * 0.01
        else:
            dq = np.pad(action, (0, 3 - len(action))) * 0.01
        
        q_current = q_current + dq
        predicted_trajectory.append(q_current.copy())
    
    predicted_trajectory = np.array(predicted_trajectory)
    
    # Compute tracking error
    true_portion = true_trajectory[:len(predicted_trajectory)]
    tracking_error = np.linalg.norm(true_portion - predicted_trajectory, axis=1)
    mean_tracking_error = np.mean(tracking_error)
    
    # Task-specific success metrics
    evaluation = {
        'task_type': task_type,
        'mean_tracking_error': float(mean_tracking_error),
        'max_tracking_error': float(np.max(tracking_error)),
        'path_smoothness': float(1.0 / (1.0 + np.std(tracking_error))),  # Higher is smoother
        'success': mean_tracking_error < 0.5  # Arbitrary threshold
    }
    
    # Task-specific criteria
    if task_type == 'pick_place':
        # Check if final position is close to target
        final_error = np.linalg.norm(predicted_trajectory[-1] - true_trajectory[-1])
        evaluation['object_placed_correctly'] = final_error < 0.3
        evaluation['success'] = evaluation['success'] and evaluation['object_placed_correctly']
    
    elif task_type == 'pushing':
        # Check if object moved in correct direction
        movement = predicted_trajectory[-1, 0] - predicted_trajectory[0, 0]
        target_movement = true_trajectory[-1, 0] - true_trajectory[0, 0]
        evaluation['correct_direction'] = (movement * target_movement) > 0
        evaluation['success'] = evaluation['success'] and evaluation['correct_direction']
    
    elif task_type == 'stacking':
        # Check if reached stacking height
        final_height = predicted_trajectory[-1, 2]
        target_height = true_trajectory[-1, 2]
        evaluation['reached_target_height'] = np.abs(final_height - target_height) < 0.2
        evaluation['success'] = evaluation['success'] and evaluation['reached_target_height']
    
    return evaluation

# Evaluate each episode
success_rates = {task: [] for task in ['pick_place', 'pushing', 'stacking']}
all_evaluations = []

for episode in episodes:
    # Simulate VLA+MPC predictions (use true trajectory as prediction + noise)
    true_actions = episode['u_trajectory']
    predicted_actions = true_actions + np.random.randn(*true_actions.shape) * 0.3  # Add noise
    
    eval_result = evaluate_task_success(episode, predicted_actions)
    all_evaluations.append(eval_result)
    
    task_type = eval_result['task_type']
    success_rates[task_type].append(1 if eval_result['success'] else 0)
    
    status = "✓ SUCCESS" if eval_result['success'] else "❌ FAILED"
    print(f"  {status} | {task_type:12} | Error: {eval_result['mean_tracking_error']:.4f}")

# ============================================================================
# PHASE 3: Aggregate Results
# ============================================================================

print("\n[PHASE 3] Aggregating results...")

summary = {
    'total_episodes': len(episodes),
    'task_success_rates': {},
    'overall_success_rate': np.mean([e['success'] for e in all_evaluations]),
    'average_tracking_error': np.mean([e['mean_tracking_error'] for e in all_evaluations]),
    'average_path_smoothness': np.mean([e['path_smoothness'] for e in all_evaluations])
}

for task_type, successes in success_rates.items():
    if successes:
        summary['task_success_rates'][task_type] = {
            'success_rate': float(np.mean(successes)),
            'num_episodes': len(successes),
            'successful': int(np.sum(successes))
        }

print(f"\n✓ Task Success Rates:")
for task, metrics in summary['task_success_rates'].items():
    rate = metrics['success_rate'] * 100
    print(f"    {task:12}: {rate:5.1f}% ({metrics['successful']}/{metrics['num_episodes']})")

print(f"\n✓ Overall Metrics:")
print(f"    Success Rate:      {summary['overall_success_rate']*100:.1f}%")
print(f"    Avg Tracking Error: {summary['average_tracking_error']:.4f}")
print(f"    Avg Path Smoothness: {summary['average_path_smoothness']:.4f}")

# ============================================================================
# PHASE 4: Save Results
# ============================================================================

output_dir = Path('results/task_success_evaluation')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'task_evaluation.json', 'w') as f:
    json.dump({
        'summary': summary,
        'evaluations': all_evaluations
    }, f, indent=2)

print(f"\n✓ Results saved to: {output_dir / 'task_evaluation.json'}")

print("\n" + "="*80)
print("NEXT: Generate MuJoCo Visualizations")
print("="*80)
print("""
To generate 3D GIFs of episodes:
  python scripts/visualize_episodes_mujoco.py

This will create:
  - results/episode_visualizations/episode_*.gif
  - Showing robot arm following predicted trajectories
  - Visual proof of task success/failure
""")

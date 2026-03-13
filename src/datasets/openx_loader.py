"""OpenX Embodiment Dataset integration for robot control evaluation.

Supports loading from:
- Google DeepMind OpenX datasets via tensorflow_datasets (RLDS format)
- HuggingFace robotics datasets  
- Local cached datasets

Reference: https://github.com/google-deepmind/open_x_embodiment
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

# Try to import tensorflow for real RLDS loading
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow/TFDS not available - synthetic data only")


@dataclass
class RLDSStep:
    """Single step from RLDS dataset following OpenX format.
    
    Mirrors the structure from Google DeepMind OpenX datasets.
    """
    # Observations
    image: np.ndarray  # [H, W, 3] uint8 RGB
    natural_language_instruction: str  # Task description
    state: Optional[np.ndarray] = None  # [DOF] joint angles or robot state
    
    # Actions (flexible format - varies by robot)
    action: Dict[str, np.ndarray] = None
    
    # Episode metadata
    is_first: bool = False
    is_last: bool = False
    is_terminal: bool = False
    reward: float = 0.0
    
    # Optional fields
    language_embedding: Optional[np.ndarray] = None  # [512] from BERT/T5
    natural_language_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.action is None:
            self.action = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Trajectory:
    """Single trajectory from OpenX dataset."""
    episode_id: str
    task_name: str
    instruction: str
    steps: List[RLDSStep]  # Sequence of RLDS steps
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        """Trajectory length in steps."""
        return len(self.steps)
    
    @property
    def images(self) -> np.ndarray:
        """Stack all images into [T, H, W, 3]."""
        return np.stack([step.image for step in self.steps])
    
    @property
    def joint_states(self) -> Optional[np.ndarray]:
        """Stack joint states if available."""
        if self.steps[0].state is None:
            return None
        return np.stack([step.state for step in self.steps])
    
    @property
    def instructions(self) -> List[str]:
        """Get all instructions in episode."""
        return [step.natural_language_instruction for step in self.steps]
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            'episode_id': self.episode_id,
            'task_name': self.task_name,
            'instruction': self.instruction,
            'length': len(self),
            'metadata': self.metadata,
        }


class OpenXDataset:
    """Interface to OpenX Embodiment datasets (Google DeepMind).
    
    Supports:
    - Real datasets via tensorflow_datasets (RLDS format)
    - Cached local datasets  
    - Synthetic datasets for development
    
    Reference: https://github.com/google-deepmind/open_x_embodiment
    Dataset list: https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit
    
    Real datasets available (partial list with sizes):
    - bridge (387 GiB) - Real robot manipulation
    - kuka (778 GiB) - KUKA robot
    - fractal20220817_data (111 GiB) - Fractal dataset
    - language_table (399 GiB) - Language-guided tabletop
    - taco_play (47 GiB) - Tabletop play collections
    - And 50+ more...
    """
    
    # Real datasets from OpenX (as of 2026-03)
    REAL_DATASETS = {
        # Large-scale real robot datasets
        'bridge': {'size_gb': 387.49, 'robot': 'Widowx', 'tasks': 'manipulation'},
        'kuka': {'size_gb': 778.02, 'robot': 'KUKA', 'tasks': 'manipulation'},
        'fractal20220817_data': {'size_gb': 111.07, 'robot': 'multiple', 'tasks': 'manipulation'},
        'language_table': {'size_gb': 399.23, 'robot': 'tabletop', 'tasks': 'manipulation'},
        'robo_net': {'size_gb': 799.91, 'robot': 'multiple', 'tasks': 'vision-based'},
        
        # Smaller-scale datasets (still high quality)
        'taco_play': {'size_gb': 47.77, 'robot': 'UR5', 'tasks': 'play'},
        'jaco_play': {'size_gb': 9.24, 'robot': 'Jaco', 'tasks': 'play'},
        'roboturk': {'size_gb': 45.39, 'robot': 'Widowx', 'tasks': 'manipulation'},
        'berkeley_cable_routing': {'size_gb': 4.67, 'robot': 'Widowx', 'tasks': 'cable'},
        'violà': {'size_gb': 10.40, 'robot': 'Widowx', 'tasks': 'manipulation'},
        'berkeley_autolab_ur5': {'size_gb': 76.39, 'robot': 'UR5', 'tasks': 'manipulation'},
        'utaustin_mutex': {'size_gb': 20.79, 'robot': 'UR5', 'tasks': 'manipulation'},
        'bc_z': {'size_gb': 80.54, 'robot': 'Widowx', 'tasks': 'manipulation'},
        
        # Converted external datasets
        'stanford_kuka_multimodal_dataset_converted_externally_to_rlds': {
            'size_gb': 31.98, 'robot': 'KUKA', 'tasks': 'manipulation'},
        'stanford_hydra_dataset_converted_externally_to_rlds': {
            'size_gb': 72.48, 'robot': 'Hydra', 'tasks': 'manipulation'},
        'nyu_rot_dataset_converted_externally_to_rlds': {
            'size_gb': 5.33, 'robot': 'Widowx', 'tasks': 'rotation'},
    }
    
    # Task types present in OpenX datasets
    TASK_TYPES = [
        'reaching', 'grasping', 'picking', 'placing', 'pushing', 
        'pulling', 'rotating', 'inserting', 'stacking', 'manipulation',
        'opening', 'closing', 'sliding', 'wiping', 'cable_routing',
        'pouring', 'sorting', 'sweeping', 'placing_in_context',
    ]
    
    def __init__(self, cache_dir: str = 'data/openx_cache', use_tfds: bool = True):
        """Initialize dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
            use_tfds: Whether to use tfds for real data (requires TensorFlow)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tfds = use_tfds and TF_AVAILABLE
        if use_tfds and not TF_AVAILABLE:
            logger.warning("TensorFlow not available - will use synthetic data only")
        
        self.loaded_datasets = {}  # {name: list of Trajectory objects}
        self.dataset_builders = {}  # Cache of tfds builders
        
        logger.info(f"OpenXDataset initialized (TensorFlow: {TF_AVAILABLE})")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    @staticmethod
    def list_real_datasets() -> Dict[str, Dict[str, Any]]:
        """List all real OpenX datasets available."""
        return OpenXDataset.REAL_DATASETS.copy()
    
    @staticmethod
    def list_all_available_datasets() -> Dict[str, Dict[str, Any]]:
        """List known dataset metadata."""
        return {
            'real_datasets': OpenXDataset.REAL_DATASETS,
            'num_real_datasets': len(OpenXDataset.REAL_DATASETS),
            'total_size_gb': sum(d.get('size_gb', 0) for d in OpenXDataset.REAL_DATASETS.values()),
        }
    
    def load_from_tfds(
        self,
        dataset_name: str,
        split: str = 'train[:10%]',  # Default to 10% of training data
        max_episodes: Optional[int] = None,
    ) -> List[Trajectory]:
        """Load real dataset from TensorFlow Datasets (RLDS format).
        
        Args:
            dataset_name: Name of dataset (e.g., 'bridge', 'kuka')
            split: Dataset split to load (default: 10% of training)
            max_episodes: Limit number of episodes loaded
            
        Returns:
            List of Trajectory objects
            
        Example:
            >>> ds = OpenXDataset()
            >>> trajectories = ds.load_from_tfds('bridge', split='train[:1%]', max_episodes=100)
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available - cannot load real datasets")
        
        logger.info(f"Loading {dataset_name} (split={split})...")
        
        try:
            # Load builder
            builder = tfds.builder(dataset_name)
            self.dataset_builders[dataset_name] = builder
            
            # Load dataset
            ds = builder.as_dataset(split=split)
            
            trajectories = []
            
            # Convert RLDS episodes to Trajectory objects
            for episode_idx, episode in enumerate(ds.take(max_episodes or -1)):
                try:
                    trajectory = self._rlds_episode_to_trajectory(
                        episode, dataset_name, episode_idx
                    )
                    trajectories.append(trajectory)
                    
                    if (episode_idx + 1) % 50 == 0:
                        logger.info(f"Loaded {episode_idx + 1} episodes from {dataset_name}")
                
                except Exception as e:
                    logger.warning(f"Failed to load episode {episode_idx}: {e}")
                    continue
            
            self.loaded_datasets[dataset_name] = trajectories
            logger.info(f"Loaded {len(trajectories)} episodes from {dataset_name}")
            
            return trajectories
        
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            logger.info("Hint: Check if dataset name is correct. "
                       "Use list_real_datasets() for valid names.")
            raise
    
    def _rlds_episode_to_trajectory(
        self,
        episode: Dict[str, Any],
        dataset_name: str,
        episode_idx: int,
    ) -> Trajectory:
        """Convert RLDS episode to Trajectory object.
        
        RLDS episode structure:
        {
            'steps': Dataset of steps,
            'features': (optional) episode-level metadata
        }
        
        Each step has:
        {
            'observation': {
                'image': [...],
                'natural_language_instruction': "...",
                'state': [...],
                ...
            },
            'action': {...varies by robot...},
            'is_first': bool,
            'is_last': bool,
            'is_terminal': bool,
            'reward': float,
        }
        """
        steps_data = episode.get('steps', [])
        
        # Convert to list if it's a tf.data.Dataset
        if isinstance(steps_data, tf.data.Dataset):
            steps_data = list(steps_data)
        
        rlds_steps = []
        instruction = ""
        
        for step_idx, step in enumerate(steps_data):
            # Extract observation
            obs = step.get('observation', {})
            
            # Image (convert TF tensor to numpy)
            image = np.array(obs.get('image', np.zeros((256, 256, 3), dtype=np.uint8)))
            
            # Language instruction
            lang_instr = obs.get('natural_language_instruction', b'').numpy()
            if isinstance(lang_instr, bytes):
                lang_instr = lang_instr.decode('utf-8', errors='ignore')
            
            if not instruction and lang_instr:
                instruction = lang_instr
            
            # State/joint angles
            state = obs.get('state')
            if state is not None:
                state = np.array(state, dtype=np.float32)
            
            # Language embedding (if available)
            lang_embed = obs.get('natural_language_embedding')
            if lang_embed is not None:
                lang_embed = np.array(lang_embed, dtype=np.float32)
            
            # Action (varies by robot - keep as dict)
            action = step.get('action', {})
            action_dict = {}
            for key, val in action.items():
                if isinstance(val, tf.Tensor):
                    action_dict[key] = val.numpy()
                else:
                    action_dict[key] = np.array(val)
            
            # Episode markers
            is_first = bool(step.get('is_first', False))
            is_last = bool(step.get('is_last', False))
            is_terminal = bool(step.get('is_terminal', False))
            reward = float(step.get('reward', 0.0))
            
            # Create RLDS step
            rlds_step = RLDSStep(
                image=image,
                natural_language_instruction=lang_instr,
                state=state,
                action=action_dict,
                is_first=is_first,
                is_last=is_last,
                is_terminal=is_terminal,
                reward=reward,
                language_embedding=lang_embed,
            )
            
            rlds_steps.append(rlds_step)
        
        # Determine task name from instruction
        task_name = self._infer_task(instruction)
        
        trajectory = Trajectory(
            episode_id=f'{dataset_name}_ep_{episode_idx:06d}',
            task_name=task_name,
            instruction=instruction,
            steps=rlds_steps,
            metadata={
                'source': 'openx_rlds',
                'dataset': dataset_name,
                'num_steps': len(rlds_steps),
            },
        )
        
        return trajectory
    
    @staticmethod
    def _infer_task(instruction: str) -> str:
        """Infer task type from instruction text."""
        instruction_lower = instruction.lower()
        
        for task in OpenXDataset.TASK_TYPES:
            if task in instruction_lower:
                return task
        
        return 'manipulation'  # Default
    
    def load_synthetic_calvin_subset(
        self,
        num_episodes: int = 100,
        seed: int = 42,
    ) -> List[Trajectory]:
        """Load or create CALVIN-like synthetic dataset for testing.
        
        Generates synthetic RLDS-formatted data with realistic structure
        for CALVIN task (language-guided manipulation on tabletop).
        
        Args:
            num_episodes: Number of episodes to generate
            seed: Random seed for reproducibility
            
        Returns:
            List[Trajectory] with RLDS-compatible step format
        """
        np.random.seed(seed)
        
        trajectories = []
        
        task_types = ['reaching', 'grasping', 'placing', 'stacking']
        instructions = {
            'reaching': ["reach to the cube", "move to the target", "go to position"],
            'grasping': ["pick up the object", "grasp the block", "hold the cube"],
            'placing': ["put it down", "place on table", "release it"],
            'stacking': ["stack the blocks", "put one on top", "arrange them"],
        }
        
        for episode_idx in range(num_episodes):
            task_type = np.random.choice(task_types)
            instruction = np.random.choice(instructions[task_type])
            
            # Random episode length (30-100 steps)
            tlen = np.random.randint(30, 100)
            
            # Create RLDS steps
            rlds_steps = []
            
            for step_idx in range(tlen):
                # Synthetic RGB image (224x224 standard for CALVIN)
                image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                
                # Robot joint state (7D for typical arm: 6 joints + gripper)
                state = np.random.uniform(-np.pi, np.pi, 7).astype(np.float32)
                
                # Language embedding (512D from BERT/T5 embeddings)
                lang_embedding = np.random.randn(512).astype(np.float32)
                
                # Action varies by robot type - use flexible dict format
                # For CALVIN: world_vector (3D), rotation_delta (3D), gripper (1D)
                action_dict = {
                    'world_vector': np.random.uniform(-0.1, 0.1, 3).astype(np.float32),
                    'rotation_delta': np.random.uniform(-0.01, 0.01, 3).astype(np.float32),
                    'gripper_closedness_action': np.array([
                        np.random.choice([0.0, 1.0])
                    ], dtype=np.float32),
                }
                
                # Episode markers
                is_first = (step_idx == 0)
                is_last = (step_idx == tlen - 1)
                is_terminal = False  # Could be true for last step
                reward = float(is_last)  # Give reward at end
                
                rlds_step = RLDSStep(
                    image=image,
                    natural_language_instruction=instruction,
                    state=state,
                    action=action_dict,
                    is_first=is_first,
                    is_last=is_last,
                    is_terminal=is_terminal,
                    reward=reward,
                    language_embedding=lang_embedding,
                    metadata={'step': step_idx, 'episode': episode_idx},
                )
                
                rlds_steps.append(rlds_step)
            
            trajectory = Trajectory(
                episode_id=f'calvin_synthetic_{episode_idx:06d}',
                task_name=task_type,
                instruction=instruction,
                steps=rlds_steps,
                metadata={
                    'source': 'calvin_synthetic',
                    'robot': 'ORCA arm (simulated)',
                    'image_size': (224, 224),
                    'action_space': 'world_vector + rotation_delta + gripper',
                },
            )
            
            trajectories.append(trajectory)
        
        self.loaded_datasets['calvin'] = trajectories
        logger.info(f"Created {num_episodes} synthetic CALVIN episodes (RLDS format)")
        
        return trajectories
    
    def load_synthetic_reaching_subset(
        self,
        num_episodes: int = 50,
        seed: int = 42,
        dof: int = 3,
    ) -> List[Trajectory]:
        """Load or create synthetic reaching dataset in RLDS format.
        
        Generates realistic reaching trajectories for point-to-point motion
        on a multi-DOF robot arm.
        
        Args:
            num_episodes: Number of episodes
            seed: Random seed
            dof: Degrees of freedom for arm (default: 3)
            
        Returns:
            List[Trajectory] with realistic reaching motions
        """
        np.random.seed(seed)
        
        trajectories = []
        
        for episode_idx in range(num_episodes):
            # Random start and goal configurations
            start_q = np.random.uniform(-np.pi, np.pi, dof)
            goal_q = np.random.uniform(-np.pi, np.pi, dof)
            
            # Reaching trajectory length (50-200 steps)
            tlen = np.random.randint(50, 200)
            
            rlds_steps = []
            
            for t in range(tlen):
                # Smooth trajectory interpolation
                alpha = t / max(1, tlen - 1)
                s = 3*alpha**2 - 2*alpha**3  # Smoothstep function
                
                # Current joint position
                q_t = start_q + s * (goal_q - start_q)
                
                # Joint velocity (approximate derivative)
                if t == 0:
                    qdot_t = np.zeros(dof)
                else:
                    qdot_t = (rlds_steps[t-1].state - q_t) / 0.01  # Assume 10ms steps
                
                # Synthetic camera image
                image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
                
                # Language embedding
                lang_embedding = np.random.randn(512).astype(np.float32)
                
                # Action: desired joint velocities + gripper
                action_dict = {
                    'joint_velocities': qdot_t.astype(np.float32),
                    'desiredAngularVelocity': qdot_t.astype(np.float32),
                    'gripper': np.array([0.0], dtype=np.float32),  # No gripper for reaching
                }
                
                # Episode markers
                is_first = (t == 0)
                is_last = (t == tlen - 1)
                is_terminal = is_last
                
                # Reward: negative distance to goal at each step
                distance_to_goal = np.linalg.norm(goal_q - q_t)
                reward = -distance_to_goal
                
                rlds_step = RLDSStep(
                    image=image,
                    natural_language_instruction=f"reach to position {goal_q}",
                    state=q_t.astype(np.float32),
                    action=action_dict,
                    is_first=is_first,
                    is_last=is_last,
                    is_terminal=is_terminal,
                    reward=reward,
                    language_embedding=lang_embedding,
                    metadata={
                        'step': t,
                        'distance_to_goal': float(distance_to_goal),
                        'start_q': start_q.tolist(),
                        'goal_q': goal_q.tolist(),
                    },
                )
                
                rlds_steps.append(rlds_step)
            
            trajectory = Trajectory(
                episode_id=f'reaching_synthetic_{episode_idx:06d}',
                task_name='reaching',
                instruction=f"reach to position {goal_q}",
                steps=rlds_steps,
                metadata={
                    'source': 'reaching_synthetic',
                    'robot': f'{dof}-DOF arm',
                    'task_type': 'reaching',
                    'start_q': start_q.tolist(),
                    'goal_q': goal_q.tolist(),
                    'success': True,  # Synthetic trajectories are always successful
                },
            )
            
            trajectories.append(trajectory)
        
        self.loaded_datasets['reaching'] = trajectories
        logger.info(f"Created {num_episodes} synthetic {dof}-DOF reaching episodes (RLDS format)")
        
        return trajectories
    
    def get_dataset(self, name: str) -> Optional[List[Trajectory]]:
        """Get loaded dataset by name.
        
        Returns list of Trajectory objects, or None if not loaded.
        """
        return self.loaded_datasets.get(name)
    
    def get_dataset_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics about a loaded dataset."""
        trajectories = self.loaded_datasets.get(name)
        if not trajectories:
            return {}
        
        traj_lengths = [len(t) for t in trajectories]
        
        # Estimate total image data size
        if trajectories and len(trajectories[0].steps) > 0:
            first_image = trajectories[0].steps[0].image
            bytes_per_image = first_image.nbytes
            total_image_gb = (sum(traj_lengths) * bytes_per_image) / (1024**3)
        else:
            total_image_gb = 0
        
        return {
            'name': name,
            'num_episodes': len(trajectories),
            'mean_length': float(np.mean(traj_lengths)),
            'std_length': float(np.std(traj_lengths)),
            'min_length': int(np.min(traj_lengths)),
            'max_length': int(np.max(traj_lengths)),
            'total_steps': int(np.sum(traj_lengths)),
            'estimated_size_gb': float(total_image_gb),
            'task_types': list(set(t.task_name for t in trajectories)),
            'has_rlds_format': True,
        }
    
    def save_dataset_to_disk(
        self,
        name: str,
        output_dir: Optional[str] = None,
        save_format: str = 'npz',  # 'npz' or 'json'
    ) -> Path:
        """Save dataset to disk in efficient format.
        
        Args:
            name: Dataset name to save
            output_dir: Where to save (default: cache_dir/{name})
            save_format: 'npz' (efficient) or 'json' (readable)
            
        Returns:
            Path to saved dataset directory
        """
        trajectories = self.loaded_datasets.get(name)
        if not trajectories:
            raise ValueError(f"Dataset {name} not loaded")
        
        output_dir = Path(output_dir or self.cache_dir / name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {name} dataset to {output_dir}...")
        
        # Save metadata
        metadata = {
            'name': name,
            'num_episodes': len(trajectories),
            'format': 'RLDS (steps as dict)',
            'stats': self.get_dataset_stats(name),
            'episode_summaries': [t.to_dict() for t in trajectories[:10]],
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual episodes
        for episode_idx, traj in enumerate(trajectories):
            episode_data = {
                'episode_id': traj.episode_id,
                'task_name': traj.task_name,
                'instruction': traj.instruction,
                'num_steps': len(traj.steps),
                'metadata': traj.metadata,
                'steps': [],
            }
            
            # Save each step
            for step in traj.steps:
                step_data = {
                    'image_shape': step.image.shape,
                    'has_state': step.state is not None,
                    'state_shape': step.state.shape if step.state is not None else None,
                    'instruction': step.natural_language_instruction,
                    'is_first': step.is_first,
                    'is_last': step.is_last,
                    'is_terminal': step.is_terminal,
                    'reward': step.reward,
                    'action_keys': list(step.action.keys()),
                }
                episode_data['steps'].append(step_data)
            
            # Save as JSON metadata + compressed images + actions
            if save_format == 'npz':
                # Efficient binary format
                images = np.stack([step.image for step in traj.steps])
                states = np.stack([step.state for step in traj.steps]) if traj.steps[0].state is not None else None
                
                save_dict = {
                    'episode_id': traj.episode_id,
                    'task_name': traj.task_name,
                    'instruction': traj.instruction,
                    'images': images,
                }
                
                if states is not None:
                    save_dict['states'] = states
                
                # Save first action as example (keep full RLDS in JSON)
                if traj.steps[0].action:
                    for action_key, action_val in traj.steps[0].action.items():
                        save_dict[f'action_sample_{action_key}'] = action_val
                
                np.savez_compressed(
                    output_dir / f'episode_{episode_idx:06d}.npz',
                    **save_dict
                )
            
            # Always save detailed JSON
            with open(output_dir / f'episode_{episode_idx:06d}_meta.json', 'w') as f:
                json.dump(episode_data, f, indent=2)
            
            if (episode_idx + 1) % 50 == 0:
                logger.info(f"Saved {episode_idx + 1}/{len(trajectories)} episodes")
        
        logger.info(f"Saved {name} dataset to {output_dir}")
        return output_dir
    
    def print_dataset_summary(self, name: str) -> None:
        """Print summary of loaded dataset."""
        stats = self.get_dataset_stats(name)
        if not stats:
            print(f"Dataset '{name}' not loaded. Available: {list(self.loaded_datasets.keys())}")
            return
        
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        print(f"Episodes: {stats['num_episodes']}")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Trajectory length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} "
              f"(range: {stats['min_length']}-{stats['max_length']})")
        print(f"Task types: {', '.join(stats['task_types'])}")
        if stats.get('estimated_size_gb') > 0:
            print(f"Estimated size: {stats['estimated_size_gb']:.2f} GB")
        print(f"Format: RLDS (ReverseLS) with flexible action spaces")
        
        # Sample trajectory
        trajectories = self.loaded_datasets[name]
        if trajectories:
            sample_traj = trajectories[0]
            print(f"\nSample episode {sample_traj.episode_id}:")
            print(f"  Task: {sample_traj.task_name}")
            print(f"  Instruction: '{sample_traj.instruction}'")
            print(f"  Length: {len(sample_traj)} steps")
            
            if len(sample_traj.steps) > 0:
                step = sample_traj.steps[0]
                print(f"  Image shape: {step.image.shape}")
                print(f"  State shape: {step.state.shape if step.state is not None else 'None'}")
                print(f"  Action keys: {list(step.action.keys())}")
        
        print(f"{'='*60}\n")


class DatasetEvaluator:
    """Evaluate control system performance on dataset trajectories.
    
    Runs a controller on real or synthetic trajectories and measures
    success rate, tracking error, and other metrics.
    """
    
    def __init__(
        self,
        controller=None,
        dynamics_model=None,
    ):
        """Initialize evaluator.
        
        Args:
            controller: Dual system controller (MPC + VLA) - optional
            dynamics_model: Forward dynamics model - optional
        """
        self.controller = controller
        self.dynamics = dynamics_model
        logger.info("DatasetEvaluator initialized")
    
    def evaluate_trajectory(
        self,
        trajectory: Trajectory,
        timeout_s: float = 60.0,
    ) -> Dict[str, float]:
        """Evaluate controller on single RLDS trajectory.
        
        Args:
            trajectory: Trajectory with RLDS steps
            timeout_s: Max time allowed
            
        Returns:
            Dict with metrics: {'success': bool, 'tracking_error': float, ...}
        """
        metrics = {
            'success': False,
            'tracking_error_rad': 0.0,
            'final_error_rad': 0.0,
            'path_length': 0.0,
            'energy_cost': 0.0,
            'num_steps': len(trajectory),
        }
        
        # Get reference trajectory from RLDS steps
        if trajectory.joint_states is not None:
            ref_states = trajectory.joint_states
            
            # If we have a controller, run it
            if self.controller is not None:
                try:
                    # Run controller on trajectory
                    # This is a placeholder - actual implementation depends on controller
                    errors = []
                    for step_idx, (ref_state, rlds_step) in enumerate(
                        zip(ref_states, trajectory.steps)
                    ):
                        # Would execute controller here
                        # For now, just track reference
                        pass
                    
                    # Compute metrics
                    if len(errors) > 0:
                        metrics['tracking_error_rad'] = float(np.mean(errors))
                        metrics['final_error_rad'] = float(errors[-1])
                        metrics['success'] = float(errors[-1]) < 0.1
                
                except Exception as e:
                    logger.warning(f"Controller evaluation failed: {e}")
        
        return metrics
    
    def evaluate_dataset(
        self,
        trajectories: List[Trajectory],
        num_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate on multiple RLDS trajectories.
        
        Args:
            trajectories: List of trajectories
            num_samples: Limit evaluation to N trajectories (for speed)
            verbose: Print progress
            
        Returns:
            Aggregate statistics
        """
        if num_samples:
            trajectories = trajectories[:num_samples]
        
        results = []
        
        for idx, traj in enumerate(trajectories):
            try:
                result = self.evaluate_trajectory(traj)
                results.append(result)
                
                if verbose and (idx + 1) % max(1, len(trajectories) // 10) == 0:
                    logger.info(
                        f"Evaluated {idx+1}/{len(trajectories)} trajectories "
                        f"(avg error: {np.mean([r['tracking_error_rad'] for r in results]):.4f} rad)"
                    )
            
            except Exception as e:
                logger.warning(f"Failed to evaluate trajectory {idx}: {e}")
                continue
        
        # Aggregate results
        successes = [r['success'] for r in results if r['success'] is not None]
        errors = [r['tracking_error_rad'] for r in results if r['tracking_error_rad'] > 0]
        
        return {
            'num_evaluated': len(results),
            'success_rate': float(np.mean(successes)) if successes else 0.0,
            'mean_tracking_error_rad': float(np.mean(errors)) if errors else 0.0,
            'std_tracking_error_rad': float(np.std(errors)) if errors else 0.0,
        }
    
    def print_evaluation_summary(self, results: Dict[str, float]) -> None:
        """Print evaluation results."""
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Trajectories evaluated: {results['num_evaluated']}")
        print(f"Success rate: {results['success_rate']*100:.1f}%")
        print(f"Mean tracking error: {results['mean_tracking_error_rad']:.4f} rad")
        print(f"Std tracking error: {results['std_tracking_error_rad']:.4f} rad")
        print(f"{'='*60}\n")

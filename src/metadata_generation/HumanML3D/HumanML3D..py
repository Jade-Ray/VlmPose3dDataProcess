import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import torch

from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor
from src.utils.logging import get_logger
from src.metadata_generation.SMPL import SMPL21Joint
from src.metadata_generation.HumanML3D.motion_process import recover_from_ric


logger = get_logger(__name__, log_file="HumanML3D_metadata_generation.log")


@dataclass
class HumanML3DConfig(BaseProcessorConfig):
    """Configuration for HumanML3D metadata generation."""
    processed_dir: str = "data/HumanML3D"
    save_dir: str = "data/processed_data/HumanML3D"
    split: str = "all" # Options: 'train', 'val', 'train_val' 'test', 'all'
    max_motion_length: int = 200  # Max length of motion sequences to consider
    min_motion_length: int = 20   # Min length of motion sequences to consider
    unit_length: int = 4  # Length of unit alignment of motion sequences 

class HumanML3DProcessor(AbstractSceneProcessor[HumanML3DConfig]):
    """Processor for generating metadata for HumanML3D dataset."""
    
    def __init__(self, config: HumanML3DConfig):
        super().__init__(config)
        # Validate essential directories early
        if not Path(self.config.processed_dir).is_dir():
            logger.warning(f"Processed data directory not found: {self.config.processed_dir}. May cause errors.")
        # Mean and std of the dataset
        self.mean = np.load(Path(self.config.processed_dir) / "Mean.npy")
        self.std = np.load(Path(self.config.processed_dir) / "Std.npy")
        self.njoints = 22
        self.fps = 20

    def _load_scene_list(self) -> List[str]:
        """Loads the list of scene IDs from the specified file."""
        logger.info(f"Loading all scene files from: {self.config.processed_dir}")
        split_file = Path(self.config.processed_dir) / f"{self.config.split}.txt"
        with open(split_file, 'r') as f:
            scene_ids = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(scene_ids)} scene IDs.")
        return scene_ids
    
    def _process_single_scene(self, scene_id: str) -> Dict[str, Any] | None:
        """Processes metadata for sampled frames in a single scene."""
        # Define base paths using config
        processed_scene_base = self.config.processed_dir
        
        # Define paths for data
        motion_path = Path(processed_scene_base) / "new_joint_vecs" / f"{scene_id}.npy"
        text_path = Path(processed_scene_base) / "texts" / f"{scene_id}.txt"
        
        # --- 1. Read Motion Data ---
        try:
            motion = np.load(motion_path)  # (T, 263)
        except FileNotFoundError:
            logger.error(f"Motion file not found for scene {scene_id}: {motion_path}")
            return None
        
        # --- 2. Read Action Description ---
        try:
            descriptions = []
            start_times = []
            end_times = []
            with open(text_path, 'r') as f:
                for line in f.readlines():
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    descriptions.append(caption)
                    
                    start_time = float(line_split[2])
                    end_time = float(line_split[3])
                    start_time = 0.0 if np.isnan(start_time) else start_time
                    end_time = 0.0 if np.isnan(end_time) else end_time
                    
        except FileNotFoundError:
            logger.error(f"Text file not found for scene {scene_id}: {text_path}")
            return None
        
        # --- 3. Process Each Sample ---
        sample_data = []
        for desc, start_t, end_t in zip(descriptions, start_times, end_times):
            if start_t == 0.0 and end_t == 0.0:
                # Use full motion
                selected_motion = motion
                name = f"{scene_id}_full"
            else:
                start_frame = int(start_t * self.fps)
                end_frame = int(end_t * self.fps)
                selected_motion = motion[start_frame:end_frame]
                name = f"{scene_id}_{start_frame}_{end_frame}"
            
            motion_length = selected_motion.shape[0]
            if motion_length < self.config.min_motion_length or motion_length > self.config.max_motion_length:
                logger.info(f"Skipping sample in scene {scene_id} due to length {motion_length} outside [{self.config.min_motion_length}, {self.config.max_motion_length}]")
                continue
            
            # Align motion length to times of unit_length
            motion_length = (motion_length // self.config.unit_length) * self.config.unit_length
            selected_motion = selected_motion[:motion_length]
            
            sample_data.append({
                "name": name,
                "motion": self.normalize(selected_motion),
                "length": motion_length,
                "description": desc,
            })
        
        # --- 4. Final Scene Summary ---
        return sample_data if sample_data else None
    
    def feat2joints(self, features) -> torch.Tensor:
        return recover_from_ric(features, self.njoints)
    
    def normalize(self, motion: np.ndarray) -> np.ndarray:
        return (motion - self.mean) / self.std
    
    def _save_results(self, results: Dict[str, Any], save_max_len: int = 2000):
        results.update({
            "dataset": "HumanML3D",
            "num_joints": self.njoints,
            "joint_names": SMPL21Joint.JOINT_NAMES,
            "fps": self.fps,
            "mean": self.mean,
            "std": self.std,
        })
        super()._save_results(results, save_max_len)


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process HumanML3D metadata based on the original data.")
    
    # Arguments for input data locations
    parser.add_argument('--processed_dir', type=str, default=HumanML3DConfig.processed_dir, 
                        help='Directory containing sampled description/pose3d subdirs.')
    
    # Arguments for output and processing behavior (from BaseProcessorConfig)
    parser.add_argument('--save_dir', type=str, default=HumanML3DConfig.save_dir, # Use base default initially
                        help=f'Directory to save the output JSON metadata (defaults to processed_dir: {HumanML3DConfig.processed_dir}).')
    parser.add_argument('--output_filename', type=str, default="simple_pose_metadata_train", # Keep default name
                        help='Name of the output JSON file.')
    parser.add_argument('--num_workers', type=int, default=BaseProcessorConfig.num_workers,
                        help='Number of worker processes for parallel processing.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=BaseProcessorConfig.random_seed,
                        help='Random seed for operations.')

    args = parser.parse_args()
    
    # Create config object using parsed arguments
    config = HumanML3DConfig(
        processed_dir=args.processed_dir,
        save_dir=args.save_dir, # Pass user value or base default
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed
    )
    
    # Initialize and run the processor
    logger.info("Starting simple pose metadata processing...")
    processor = HumanML3DProcessor(config)
    processor.process_all_scenes()


if __name__ == "__main__":
    main()

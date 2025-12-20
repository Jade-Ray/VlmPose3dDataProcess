import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

import numpy as np

from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor
from src.utils.logging import get_logger
from src.metadata_generation.SMPL import SMPL21Joint


logger = get_logger(__name__, log_file="pose3dmotion_metadata_generation.log")


@dataclass
class SimplePoseProcessorConfig(BaseProcessorConfig):
    """Configuration for SimplePose metadata generation."""
    processed_dir: str = "data/Pos3DMotionSeq/simple_pose_1w"
    save_dir: str = "data/processed_data/SimplePose"


class SimplePoseProcessor(AbstractSceneProcessor[SimplePoseProcessorConfig]):
    """Processor for generating metadata for SimplePose dataset."""
    
    def __init__(self, config: SimplePoseProcessorConfig):
        super().__init__(config)
        # Validate essential directories early
        if not Path(self.config.processed_dir).is_dir():
            logger.warning(f"Processed data directory not found: {self.config.processed_dir}. May cause errors.")

    def _load_scene_list(self) -> List[str]:
        """Loads the list of scene IDs from the specified file."""
        logger.info(f"Loading all scene files from: {self.config.processed_dir}")
        scene_ids = []
        for scene_file in Path(self.config.processed_dir).glob("*.json"):
            scene_ids.append(scene_file.stem)
        logger.info(f"Loaded {len(scene_ids)} scene IDs.")
        return scene_ids
    
    def _process_single_scene(self, scene_id: str) -> Dict[str, Any] | None:
        """Processes metadata for sampled frames in a single scene."""
        # Define base paths using config
        processed_scene_base = self.config.processed_dir
        
        # Define paths for data
        pose_path = Path(processed_scene_base) / f"{scene_id}.json"
        
        # --- 1. Read Pose3D Data ---
        try:
            with open(pose_path, 'r', encoding='utf-8') as f:
                pose3d_data = json.load(f)
            body_motion_pose = np.array(pose3d_data) # (T, 22, 3) 坐标表示
            body_motion_pose = body_motion_pose - body_motion_pose[0, 0, :] # Normalize first root joint to origin
        except FileNotFoundError:
            logger.error(f"Pose3D data file not found for scene {scene_id}: {pose_path}")
            return None
        
        # --- 2. Read Action Description ---
        description = str(scene_id).split(".")[1][1:]
        
        # --- 3. Process Each Sample Frame ---
        frame_data = []
        for i, body_pose in enumerate(body_motion_pose):
            frame_data.append({
                "frame_index": i,
                "body_poses": [body_pose], # (1, 22, 3) for single object
            })
        
        # --- 4. Final Scene Summary ---
        return {
            "joint_names": SMPL21Joint.JOINT_NAMES,
            "description": description,
            "frames": frame_data,
        }


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process Simple Pose metadata based on the original data.")
    
    # Arguments for input data locations
    parser.add_argument('--processed_dir', type=str, default=SimplePoseProcessorConfig.processed_dir, 
                        help='Directory containing sampled description/pose3d subdirs.')
    
    # Arguments for output and processing behavior (from BaseProcessorConfig)
    parser.add_argument('--save_dir', type=str, default=SimplePoseProcessorConfig.save_dir, # Use base default initially
                        help=f'Directory to save the output JSON metadata (defaults to processed_dir: {SimplePoseProcessorConfig.processed_dir}).')
    parser.add_argument('--output_filename', type=str, default="simple_pose_metadata.json", # Keep default name
                        help='Name of the output JSON file.')
    parser.add_argument('--num_workers', type=int, default=BaseProcessorConfig.num_workers,
                        help='Number of worker processes for parallel processing.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=BaseProcessorConfig.random_seed,
                        help='Random seed for operations.')

    args = parser.parse_args()
    
    # Create config object using parsed arguments
    config = SimplePoseProcessorConfig(
        processed_dir=args.processed_dir,
        save_dir=args.save_dir, # Pass user value or base default
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed
    )
    
    # Initialize and run the processor
    logger.info("Starting simple pose metadata processing...")
    processor = SimplePoseProcessor(config)
    processor.process_all_scenes()


if __name__ == "__main__":
    main()

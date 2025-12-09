import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

import numpy as np

from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor
from src.utils.logging import get_logger
from src.metadata_generation.Pos3DMotionSeq.SMPL import SMPL21Joint


logger = get_logger(__name__, log_file="pose3dmotion_metadata_generation.log")


@dataclass
class Pose3DSingleProcessorConfig(BaseProcessorConfig):
    """Configuration for Pose3DSingle metadata generation."""
    processed_dir: str = "data/Pos3DMotionSeq/SinglePose"
    save_dir: str = "data/processed_data/SinglePose"


class Pose3DSingleProcessor(AbstractSceneProcessor[Pose3DSingleProcessorConfig]):
    """Processor for generating metadata for Pose3DSingle dataset."""
    
    def __init__(self, config: Pose3DSingleProcessorConfig):
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
            body_pose = np.array(pose3d_data['joints']) # (22, 3) 坐标表示
            body_pose = body_pose - body_pose[0] # Normalize root joint to origin
        except FileNotFoundError:
            logger.error(f"Pose3D data file not found for scene {scene_id}: {pose_path}")
            return None
        
        # --- 2. Read Action Description ---
        description = str(scene_id).replace('_', ' ')
        
        # --- 3. Process Single Sample Frame ---
        frame_data = [{
            "frame_index": 0,
            "body_poses": [body_pose], # (1, 22, 3) for single object
        }]
        
        # --- 4. Final Scene Summary ---
        return {
            "joint_names": SMPL21Joint.JOINT_NAMES,
            "description": description,
            "frames": frame_data,
        }


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process Pose3DSingle metadata based on the original data.")
    
    # Arguments for input data locations
    parser.add_argument('--processed_dir', type=str, default=Pose3DSingleProcessorConfig.processed_dir, 
                        help='Directory containing sampled description/pose3d subdirs.')
    
    # Arguments for output and processing behavior (from BaseProcessorConfig)
    parser.add_argument('--save_dir', type=str, default=Pose3DSingleProcessorConfig.save_dir, # Use base default initially
                        help=f'Directory to save the output JSON metadata (defaults to processed_dir: {Pose3DSingleProcessorConfig.processed_dir}).')
    parser.add_argument('--output_filename', type=str, default="Pose3DSingle_metadata.json", # Keep default name
                        help='Name of the output JSON file.')
    parser.add_argument('--num_workers', type=int, default=BaseProcessorConfig.num_workers,
                        help='Number of worker processes for parallel processing.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=BaseProcessorConfig.random_seed,
                        help='Random seed for operations.')

    args = parser.parse_args()
    
    # Create config object using parsed arguments
    config = Pose3DSingleProcessorConfig(
        processed_dir=args.processed_dir,
        save_dir=args.save_dir, # Pass user value or base default
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed
    )
    
    # Initialize and run the processor
    logger.info("Starting Pose3DSingle metadata processing...")
    processor = Pose3DSingleProcessor(config)
    processor.process_all_scenes()


if __name__ == "__main__":
    main()

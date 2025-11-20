import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

import numpy as np

from src.base_processor import BaseProcessorConfig, AbstractSceneProcessor
from src.utils.logging import get_logger
from src.metadata_generation.Pos3DMotionSeq.SMPL import SMPL21Joint


logger = get_logger(__name__, log_file="pose3dmotion_metadata_generation.log")


@dataclass
class Pose3DMotionProcessorConfig(BaseProcessorConfig):
    """Configuration for Pose3DMotionSeq metadata generation."""
    processed_dir: str = "data/Pose3DMotionSeq"
    scene_list_file: str = "data/Pose3DMotionSeq/train.txt" # Default scene list
    split: str = field(init=False) # Will be inferred from scene_list_file
    
    def __post_init__(self):
        # Infer split from scene_list_file
        try:
            name = Path(self.scene_list_file).stem
            if name in ['train', 'val', 'test']:
                self.split = name
                logger.info(f"Inferred split '{self.split}' from scene_list_file: {self.scene_list_file}")
            else:
                # Fallback or error if split cannot be determined
                logger.warning(f"Could not infer split from scene_list_file name: {self.scene_list_file}. Defaulting to 'unknown'.")
                self.split = 'unknown' # Or raise an error?
        except Exception as e:
            logger.error(f"Error inferring split from scene_list_file: {e}. Defaulting to 'unknown'.")
            self.split = 'unknown'
        
        # Ensure save_dir defaults relative to processed_dir if not explicitly set different
        if self.save_dir == BaseProcessorConfig.save_dir: # Check if using base default
            self.save_dir = self.processed_dir
            logger.info(f"Defaulting save_dir to processed_dir: {self.save_dir}")

        # Update output filename to include split if not already custom
        if self.output_filename == "pos3dmotion_metadata.json" and self.split != 'unknown':
            self.output_filename = f"pos3dmotion_metadata_{self.split}.json"
            logger.info(f"Updated output filename to include split: {self.output_filename}")
            

class Pose3DMotionProcessor(AbstractSceneProcessor[Pose3DMotionProcessorConfig]):
    """Processor for generating metadata for Pose3DMotionSeq dataset."""
    
    def __init__(self, config: Pose3DMotionProcessorConfig):
        super().__init__(config)
        # Validate essential directories early
        if not Path(self.config.processed_dir).is_dir():
            logger.warning(f"Processed data directory not found: {self.config.processed_dir}. May cause errors.")

    def _load_scene_list(self) -> List[str]:
        """Loads the list of scene IDs from the specified file."""
        logger.info(f"Loading scene list from: {self.config.scene_list_file}")
        try:
            with open(self.config.scene_list_file, 'r') as f:
                scene_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(scene_ids)} scene IDs.")
            return scene_ids
        except FileNotFoundError:
            logger.error(f"Scene list file not found: {self.config.scene_list_file}")
            return [] # Return empty list on error
    
    def _process_single_scene(self, scene_id: str) -> Dict[str, Any] | None:
        """Processes metadata for sampled frames in a single scene."""
        # Define base paths using config
        processed_scene_base = self.config.processed_dir
        
        # Define paths for data
        desc_path = Path(processed_scene_base) / f"description/{scene_id}_Medium_②.json"
        pose_path = Path(processed_scene_base) / f"pose3d/hmr4d_results_merged.json"
        
        # --- 1. Read Pose3D Data ---
        try:
            with open(pose_path, 'r') as f:
                pose3d_data = json.load(f)
            all_body_pose = np.array(pose3d_data['1']['net_outputs']['pred_smpl_params_incam']['body_pose'])[0] # (T, 63)
            all_global_orient = np.array(pose3d_data['1']['net_outputs']['pred_smpl_params_incam']['global_orient'])[0] # (T, 3)
        except FileNotFoundError:
            logger.error(f"Pose3D data file not found for scene {scene_id}: {pose_path}")
            return None
        
        # --- 2. Read Action Description ---
        try:
            with open(desc_path, 'r') as f:
                desc_data = json.load(f)
            description = desc_data['description']
        except FileNotFoundError:
            logger.error(f"Description file not found for scene {scene_id}: {desc_path}")
            return None
        
        # --- 3. Process Each Sample Frame ---
        frame_data = []
        for i, (body_pose, global_orient) in enumerate(zip(all_body_pose, all_global_orient)):
            # --- 3a. Process Full Body Pose --- 
            body_pose = SMPL21Joint._check_body_pose(body_pose) # (21, 3)
            full_body_pose = np.vstack([global_orient, body_pose], axis=0) # (22, 3)
            # --- 3b. Assemble Frame Information ---
            frame_info = {
                "frame_index": i,
                "body_pose": full_body_pose, # (22, 3)
            }
            frame_data.append(frame_info)
        
        # --- 4. Final Scene Summary ---
        return {
            "joint_names": SMPL21Joint.JOINT_NAMES,
            "description": description,
            "frames": frame_data,
        }


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Process Pose3DMotionSeq frame metadata based on the original data.")
    
    # Arguments for input data locations
    parser.add_argument('--processed_dir', type=str, default=Pose3DMotionProcessorConfig.processed_dir, 
                        help='Directory containing sampled description/pose3d subdirs.')
    parser.add_argument('--scene_list_file', type=str, default=Pose3DMotionProcessorConfig.scene_list_file,
                        help='Path to the text file listing scene IDs to process.')
    
    # Arguments for output and processing behavior (from BaseProcessorConfig)
    parser.add_argument('--save_dir', type=str, default=BaseProcessorConfig.save_dir, # Use base default initially
                        help=f'Directory to save the output JSON metadata (defaults to processed_dir: {Pose3DMotionProcessorConfig.processed_dir}).')
    parser.add_argument('--output_filename', type=str, default="Pose3DMotionSeq_metadata.json", # Keep default name
                        help='Name of the output JSON file.')
    parser.add_argument('--num_workers', type=int, default=BaseProcessorConfig.num_workers,
                        help='Number of worker processes for parallel processing.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Allow overwriting existing output file.')
    parser.add_argument('--random_seed', type=int, default=BaseProcessorConfig.random_seed,
                        help='Random seed for operations.')

    args = parser.parse_args()
    
    # Create config object using parsed arguments
    config = Pose3DMotionProcessorConfig(
        processed_dir=args.processed_dir,
        scene_list_file=args.scene_list_file,
        save_dir=args.save_dir, # Pass user value or base default
        output_filename=args.output_filename,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        random_seed=args.random_seed
    )
    
    # Initialize and run the processor
    logger.info("Starting Pose3DMotionSeq metadata processing...")
    processor = Pose3DMotionProcessor(config)
    processor.process_all_scenes()


if __name__ == "__main__":
    main()

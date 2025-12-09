import json
import argparse
import tqdm
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, TypeVar, Generic
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import src.question_templates as prompt_templates

logger = logging.getLogger(__name__)

TConfig = TypeVar('TConfig', bound='BaseQAGeneratorConfig')


@dataclass
class BaseQAGeneratorConfig:
    """Base configuration for question-answer generation."""
    output_dir: str = "output"
    num_workers: int = 1
    num_subsample: int = 6
    output_filename: str = "" # Should be set by subclasses or arguments
    processed_data_path: str = "" # Should be set by subclasses or arguments
    split_type: str = "" # Should be set by subclasses or arguments
    question_template: str = "" # Should be set by subclasses or arguments
    dataset : str = "" # Should be set by subclasses or arguments
    task_name: str = "" # Should be set by subclasses or arguments
    
    def __post_init__(self):
        self.processed_data_path += f"/{self.dataset}"
        self.question_type = f"{self.task_name}"
        if self.output_filename == "":
            self.output_filename = f"qa_{self.task_name}.json"
        if self.output_filename.split(".")[-1] != "json":
            raise ValueError("output_filename must end with .json")
        

class BaseQAGenerator(ABC, Generic[TConfig]):
    
    def __init__(self, config: TConfig):
        self.config = config
        logger.info(f"Initialized QAGenerator with config:")
        for arg, value in vars(self.config).items():
            logger.info(f"  {arg}: {value}")
        
        assert self.config.processed_data_path is not None, "processed_data_path must be provided"
        assert self.config.dataset is not None, "dataset must be provided"
        assert self.config.split_type is not None, "split_type must be provided"
        
        self._load_annotations()
        self.question_template = None if self.config.question_template is None else getattr(prompt_templates, self.config.question_template)
    
    @classmethod
    def _get_common_parser(cls, desc: str) -> argparse.ArgumentParser:
        """Get a common argument parser for all QA generators."""
        parser = argparse.ArgumentParser(description=desc)
        
        # Common arguments
        parser.add_argument('--split_type', type=str, help='Type of split to use (e.g., "train", "val", "test").')
        parser.add_argument('--processed_data_path', type=str, required=True, help='Path to the processed data directory.')
        parser.add_argument('--output_dir', type=str, help='Directory to save the output QA JSON file.')
        parser.add_argument('--dataset', type=str, help='Name of the dataset.')
        parser.add_argument('--num_subsample', type=int, default=BaseQAGeneratorConfig.num_subsample, help='Max number of questions to generate per scene.')
        parser.add_argument('--num_workers', type=int, default=BaseQAGeneratorConfig.num_workers, help='Number of worker processes for parallel scene processing.')
        
        return parser
    
    @classmethod
    def _add_specific_arguments(cls, parser: argparse.ArgumentParser):
        """To be overridden by subclasses to add task-specific arguments."""
        pass
    
    @classmethod
    def parse_args(cls, desc: str='Base QA Generator arg parser') -> argparse.Namespace:
        parser = cls._get_common_parser(desc)
        cls._add_specific_arguments(parser)
        return parser.parse_args()
    
    def _load_annotations(self):
        split_type = self.config.split_type
        dataset_name_lower = self.config.dataset.lower()
        metadata_path = Path(self.config.processed_data_path)
        assert metadata_path.exists(), f"Metadata path does not exist: {metadata_path}"
        scene_meta_path = metadata_path / f"{dataset_name_lower}_metadata_{split_type}.json"
        self.scene_annos = self._load_json(scene_meta_path)
        self.scene_list = list(self.scene_annos.keys()) if self.scene_annos else []
        
        logger.info(f"Processed data paths constructed based on : {self.config.processed_data_path}, Dataset: {self.config.dataset}, Split: {split_type}")
        if not self.scene_annos:
            logger.warning(f"Scene metadata from {scene_meta_path} is empty or failed to load.")
        else:
            logger.info(f"Scene metadata path: {scene_meta_path}")
    
    def _load_json(self, file_path):
        """Loads JSON data from a file."""
        if not file_path:
            logger.warning("No path provided for a metadata file.")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded metadata from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred loading {file_path}: {e}")
            return None
    
    def _get_scene_info(self, scene_name):
        """Retrieve multi object pose3D data for a given scene."""
        if scene_name not in self.scene_annos:
            logger.warning(f"Scene {scene_name} not found in annotations.")
            return {}
        
        scene_info = self.scene_annos[scene_name]
        objects_body_pose = defaultdict(list)
        for frame_data in scene_info['frames']:
            for obj_idx, body_pose in enumerate(frame_data['body_poses']):
                objects_body_pose[obj_idx].append(body_pose)
        return {
            "description": scene_info["description"],
            "joint_names": scene_info["joint_names"],
            "objects_body_pose": objects_body_pose,
            "num_frames": len(scene_info['frames']),
            "num_objects": len(objects_body_pose),
        }
    
    @abstractmethod
    def _process_scene_qa(self, scene_name: str) -> List[Dict[str, Any]]:
        """
        Generate QA pairs for a single scene, using both scene and frame info.

        Args:
            scene_name (str): The name of the scene.
            scene_info (dict): Annotation information for the scene from scene meta.
            frame_info_for_scene (dict): Annotation info for the scene from frame meta.

        Returns:
            list: A list of QA dictionaries for the scene.
        """
        pass
    
    def _save_results(self, all_qa_list: List):
        output_dir = Path(self.config.output_dir) / self.config.split_type
        qa_path = output_dir / self.config.output_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving QA pairs to {qa_path}...")
        try:
            with open(qa_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_list, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved QA pairs to {qa_path}")
        except IOError as e:
            logger.error(f"Failed to write output file {qa_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during saving: {e}")
    
    def run(self):
        scene_list = self.scene_list
        if not scene_list:
            logger.warning("No scenes to loaded. Exiting processing.")
            return
        
        aggregated_results = []
        num_workers = max(1, self.config.num_workers)
        logger.info(f"Starting QA generation with {num_workers} worker(s) for {len(scene_list)} scenes...")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map scenes to the processing function
            futures = {executor.submit(self._process_scene_qa, scene_name): scene_name for scene_name in scene_list}
            
            # Collect results as they complete, with progress bar
            for future in tqdm.tqdm(as_completed(futures), total=len(scene_list), desc="Processing Scenes"):
                scene_name = futures[future]
                try:
                    scene_qa_list = future.result()
                    if scene_qa_list: # Only extend if results were generated
                        aggregated_results.extend(scene_qa_list)
                except Exception as exc:
                    logger.exception(f'Scene {scene_name} generated an exception: {exc}') # Log exception with traceback

        # Re-assign IDs sequentially after collecting all results
        logger.info("Assigning final IDs...")
        for i, qa in enumerate(aggregated_results):
            qa["id"] = i # Assign a unique, sequential ID
        
        qa_count = len(aggregated_results)
        logger.info(f"Total number of QA pairs generated: {qa_count}")
        if qa_count > 0:
            self._save_results(aggregated_results)
        else:
            logger.warning("No QA pairs were generated. Check your scene data and processing logic.")

import random
import logging
import json
import numpy as np

from dataclasses import dataclass

from src.tasks.base_qa_generator import BaseQAGenerator, BaseQAGeneratorConfig


logger = logging.getLogger(__name__)


@dataclass
class SinglePoseQAGeneratorConfig(BaseQAGeneratorConfig):
    """Configuration for single pose question-answer generation."""
    output_filename: str = "" # If empty, will be set based on task_name
    dataset: str = "SinglePose"
    question_template: str = "SINGLE_POSE_TEMPLATE"
    task_name: str = "single_pose"


class SinglePoseQAGenerator(BaseQAGenerator[SinglePoseQAGeneratorConfig]):
    """Generator for single pose question-answer pairs."""
    
    def __init__(self, config: SinglePoseQAGeneratorConfig):
        super().__init__(config)
    
    @classmethod
    def _add_specific_arguments(cls, parser):
        parser.add_argument('--dataset', type=str, default=SinglePoseQAGeneratorConfig.dataset, help='Name of the dataset.')
        parser.add_argument('--question_template', type=str, default=SinglePoseQAGeneratorConfig.question_template, help='Name of the question template constant from question_templates.py.')
        parser.add_argument('--output_filename', type=str, default=SinglePoseQAGeneratorConfig.output_filename, help='the output JSON filename (e.g., "qa_obj_count").')
    
    def _generate_json_answer(self, body_pose):
        """Generate JSON formatted answer for body pose."""
        answer = np.round(np.array(body_pose)[1:], decimals=2).tolist() # Exclude root joint
        json_str = json.dumps(answer)
        return f"```JSON\n{json_str}\n```"
    
    def _process_scene_qa(self, scene_name):
        """Generate object count QA pairs for a single scene."""
        scene_qa_list = []
        scene_info = self._get_scene_info(scene_name)
        body_pose = scene_info["objects_body_pose"][0][0]  # Single object, single frame
        
        qa = {
            "dataset": self.config.dataset,
            "scene_name": scene_name,
            "question_type": self.config.question_type,
            "question": self.question_template.format(
                ACTION_DESC=scene_info["description"],
                JOINT_NAMES=scene_info["joint_names"]),
            "ground_truth": self._generate_json_answer(body_pose),
        }
        scene_qa_list.append(qa)
        
        # Subsample questions per scene if more were generated than requested
        if len(scene_qa_list) > self.config.num_subsample:
            scene_qa_list = random.sample(scene_qa_list, self.config.num_subsample)
        
        return scene_qa_list


def main():
    # Set up basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = SinglePoseQAGenerator.parse_args(desc="Single Pose QA Generator")
    
    # Create config object using parsed arguments
    config = SinglePoseQAGeneratorConfig(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        num_subsample=args.num_subsample,
        output_filename=args.output_filename,
        processed_data_path=args.processed_data_path,
        split_type=args.split_type,
        question_template=args.question_template,
        dataset=args.dataset
    )
    
    # Initialize the generator
    generator = SinglePoseQAGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()


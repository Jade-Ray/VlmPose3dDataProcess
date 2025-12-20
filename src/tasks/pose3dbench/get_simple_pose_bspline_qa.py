import random
import logging
import json
import numpy as np

from dataclasses import dataclass

from src.tasks.base_qa_generator import BaseQAGenerator, BaseQAGeneratorConfig
from src.metadata_generation.b_spline_fitter import CorrdinateBSplineFitter as BSplineFitter


logger = logging.getLogger(__name__)


@dataclass
class SimplePoseBSplineQAGeneratorConfig(BaseQAGeneratorConfig):
    """Configuration for simple pose B-spline question-answer generation."""
    output_filename: str = "" # If empty, will be set based on task_name
    processed_data_path: str = "processed_data"
    dataset: str = "SimplePose"
    split_type: str = "train"
    question_template: str = "SIMPLE_POSE_BSPLINE_TEMPLATE"
    task_name: str = "simple_pose_bspline"


class SimplePoseBSplineQAGenerator(BaseQAGenerator[SimplePoseBSplineQAGeneratorConfig]):
    """Generator for simple pose B-spline question-answer pairs."""
    
    def __init__(self, config: SimplePoseBSplineQAGeneratorConfig):
        super().__init__(config)
    
    @classmethod
    def _add_specific_arguments(cls, parser):
        parser.add_argument('--question_template', type=str, default=SimplePoseBSplineQAGeneratorConfig.question_template, help='Name of the question template constant from question_templates.py.')
        parser.add_argument('--output_filename', type=str, default=SimplePoseBSplineQAGeneratorConfig.output_filename, help='the output JSON filename (e.g., "qa_obj_count").')
    
    def _generate_json_answer(self, fitter: BSplineFitter):
        """Generate JSON formatted answer for body pose."""
        joint_coefs = []
        for splines in fitter.splines.values():
            joint_coefs.append(
                [BSplineFitter.get_spline_vaild_coefficients(spline).tolist() for spline in splines]
            )
        json_str = json.dumps(joint_coefs)
        return f"```JSON\n{json_str}\n```"
    
    def _process_scene_qa(self, scene_name):
        """Generate object count QA pairs for a single scene."""
        scene_qa_list = []
        scene_info = self._get_scene_info(scene_name)
        body_pose = scene_info["objects_body_pose"][0]  # Single object, (T, 22, 3)
        
        fitter = BSplineFitter(degree=3, num_control_points=4)
        fitter.fit_all_joints(np.transpose(body_pose, (1, 0, 2)))  # (22, T, 3)
        
        qa = {
            "dataset": self.config.dataset,
            "scene_name": scene_name,
            "question_type": self.config.question_type,
            "question": self.question_template.format(
                ACTION_DESC=scene_info["description"],
                JOINT_NAMES=scene_info["joint_names"]),
            "ground_truth": self._generate_json_answer(fitter),
        }
        scene_qa_list.append(qa)
        
        # Subsample questions per scene if more were generated than requested
        if len(scene_qa_list) > self.config.num_subsample:
            scene_qa_list = random.sample(scene_qa_list, self.config.num_subsample)
        
        return scene_qa_list


def main():
    # Set up basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = SimplePoseBSplineQAGenerator.parse_args(desc="Simple Pose QA Generator")
    
    # Create config object using parsed arguments
    config = SimplePoseBSplineQAGeneratorConfig(
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
    generator = SimplePoseBSplineQAGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()


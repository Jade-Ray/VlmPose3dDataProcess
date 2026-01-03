import random
import logging
from pathlib import Path

from dataclasses import dataclass

from src.tasks.base_qa_generator import BaseQAGenerator, BaseQAGeneratorConfig


logger = logging.getLogger(__name__)


@dataclass
class VaePoseQAGeneratorConfig(BaseQAGeneratorConfig):
    """Configuration for vae pose question-answer generation."""
    output_filename: str = "" # If empty, will be set based on task_name
    dataset: str = "HumanML3D"
    question_template: str = "VAE_POSE_TEMPLATE"
    task_name: str = "vae_pose"
    vae_latent_dir: str = "data/HumanML3D/latents"


class VaePoseQAGenerator(BaseQAGenerator[VaePoseQAGeneratorConfig]):
    """Generator for simple pose question-answer pairs."""
    
    def __init__(self, config: VaePoseQAGeneratorConfig):
        super().__init__(config)
    
    @classmethod
    def _add_specific_arguments(cls, parser):
        parser.add_argument('--vae_latent_dir', type=str, default=VaePoseQAGeneratorConfig.vae_latent_dir, help='Directory containing VAE latent files.')
        parser.set_defaults(
            output_filename=VaePoseQAGeneratorConfig.output_filename, 
            question_template=VaePoseQAGeneratorConfig.question_template, 
            dataset=VaePoseQAGeneratorConfig.dataset
        )
    
    def _get_scene_info(self, scene_name):
        if scene_name not in self.scene_annos:
            logger.warning(f"Scene {scene_name} not found in annotations.")
            return []
        scene_anno = self.scene_annos[scene_name]
        return scene_anno if isinstance(scene_anno, list) else [scene_anno]
    
    def _process_scene_qa(self, scene_name):
        """Generate object count QA pairs for a single scene."""
        scene_qa_list = []
        scene_info = self._get_scene_info(scene_name)
        
        for info in scene_info:
            latent_name = info["name"]
            latent_file = Path(self.config.vae_latent_dir) / f"{latent_name}.npy"
            if not latent_file.exists():
                logger.warning(f"VAE latent file {latent_file} not found. Skipping.")
                continue
                
            description = info["description"]
            if len(description.strip()) > 600:
                logger.warning(f"Too long description ({len(description.strip())}) for {latent_name}: {description} Skipping.")
                continue
            
            qa = {
                "dataset": self.config.dataset,
                "scene_name": scene_name,
                "question_type": self.config.question_type,
                "question": self.question_template.format(ACTION_DESC=description),
                "latent_name": latent_name,
                "length": info["length"],
            }
            scene_qa_list.append(qa)
        
        # Subsample questions per scene if more were generated than requested
        if len(scene_qa_list) > self.config.num_subsample:
            scene_qa_list = random.sample(scene_qa_list, self.config.num_subsample)
        
        return scene_qa_list


def main():
    # Set up basic logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    args = VaePoseQAGenerator.parse_args(desc="Vae Pose QA Generator")
    
    # Create config object using parsed arguments
    config = VaePoseQAGeneratorConfig(
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        num_subsample=args.num_subsample,
        output_filename=args.output_filename,
        processed_data_path=args.processed_data_path,
        split=args.split,
        question_template=args.question_template,
        dataset=args.dataset,
        vae_latent_dir=args.vae_latent_dir,
    )
    
    # Initialize the generator
    generator = VaePoseQAGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()


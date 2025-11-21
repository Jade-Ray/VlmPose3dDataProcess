# Stage 2 QA Generation Tasks

This directory contains Python scripts for generating Question-Answer (QA) pairs for 3D pose estimation tasks.

## Structure

- `base_qa_generation.py`: Defines the `BaseQAGenerator` abstract base class, which handles common functionalities like argument parsing, data loading, scene iteration, subsampling, and saving results.
- `pose3dbench/`: Directory containing task scripts inheriting from `BaseQAGenerator` for pose3d benchmark tasks (3D human pose sequence generation). Examples include:
  - `get_simple_pose_qa.py`: Generates simple pose-related QA pairs.
- `all_generate.py`: A script to run multiple QA generation tasks sequentially (can be modified to run specific tasks).

## Usage

Each task script can be run independently from the command line.

**Common Arguments (Handled by BaseQAGenerator):**

- `--processed_data_path`: Base directory containing processed data (metadata, etc.). Paths to specific data types are derived from this.
- `--dataset`: Name of the dataset (e.g., `Pos3DMotionSeq`). Used to construct data paths.
- `--split_type`: Type of the data split (e.g., `train`, `val`, `test`). Used to construct data paths.
- `--output_dir`: Directory to save the output QA JSON file (e.g., `../data/qa_pairs`). The script will create a subdirectory based on `split_type`.
- `--question_template`: Name of the question template constant from `src.question_templates`. (Usually set by default within the script).
- `--num_subsample`: Number of questions to subsample per scene (default: 6, can be overridden by task or in `all_generate.py`).
- `--num_workers`: Number of parallel processes to use for scene processing (default: 1).

**Running a Single Task:**

Navigate to the project's root directory or ensure `src` is in your Python path. Use `python -m <module_path>` to run scripts as modules.

```bash
# Example: Run simple pose QA generation (Pose3D benchmark)
python -m src.tasks.pose3dbench.get_simple_pose_qa \
    --processed_data_path data/processed/Pos3DMotionSeq \
    --dataset Pos3DMotionSeq \
    --split_type train \
    --output_dir data/qa_output \
    --num_workers 4
```

Each script will generate a JSON file named like `<output_filename>.json` (defined in the script's get_default_args) inside the `output_dir/split_type/` directory.

**Running All Task:**

Modify and run `all_generate.py` to execute multiple generation scripts. Ensure you provide the necessary common arguments and that the desired task scripts are included in its `script_configs` dictionary.

```bash
python -m src.tasks.all_generate \
    --split_type train \
    --processed_data_path data/processed_data/Pos3DMotionSeq \
    --dataset Pos3DMotionSeq \
    --output_dir data/qa_output \
    --num_subsample 10 \
    --num_workers 64 # Add other arguments as needed
```

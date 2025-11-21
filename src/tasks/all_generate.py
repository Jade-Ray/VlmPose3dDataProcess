import argparse
import subprocess
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run multiple Python scripts with common arguments.")
    parser.add_argument("--split_type", required=True, help="Split type")
    parser.add_argument("--processed_data_path", required=True, help="Path to the processed data")
    parser.add_argument("--output_dir", required=True, help="Output directory for QA pairs")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--num_subsample", type=int, default=10000, help="Number of subsamples")
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes for parallel scene processing.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration for each script
    script_configs = {
        "get_simple_pose_qa.py": {"num_subsample": str(args.num_subsample)}
    }
    
    # List of Python files to run (derived from the keys of the config dict)
    python_files_to_run = list(script_configs.keys())    
    
    # Common arguments
    common_args = [
        "--split_type", args.split_type,
        "--processed_data_path", args.processed_data_path,
        "--dataset", args.dataset,
        "--num_workers", str(args.num_workers)
    ]
    
    # Loop through each configured Python file and run it
    for python_file in python_files_to_run:
        configs = script_configs[python_file]
        if not isinstance(configs, list):
            configs = [configs]
        
        for config in configs:
            module_name = f"src.tasks.{python_file[:-3]}"
            command = [
                "python", "-m", module_name,
                *common_args,
                "--output_dir", args.output_dir,
                "--num_subsample", config.get("num_subsample", str(args.num_subsample)),
            ]
            
            print(f"Running: {' '.join(command)}")
            try:
                subprocess.run(command, check=True, text=True)
                print(f"Successfully finished running {python_file}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error running {python_file}:")
                print(f"Return code: {e.returncode}")
                print("Stopping execution due to error.")
                return

    print("All scripts have been executed successfully.")


if __name__ == "__main__":
    main()
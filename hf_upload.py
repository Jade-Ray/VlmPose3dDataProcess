import json
import argparse
import logging
from pathlib import Path
from huggingface_hub import login, HfApi


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def convert_to_conversational_format(file_path):
    if file_path.exists():
        data = load_json_data(file_path)
        converted_data = []
        for item in data:
            question = item['question']
            answer = item.get('ground_truth', "")
            conversations = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer}
            ]
            converted_data.append({
                "id": item["id"],
                "data_source": item["dataset"],
                "scene_name": item["scene_name"],
                "latent_name": item["latent_name"],
                "motion_length": item["length"],
                "question_type": item["question_type"],
                "conversations": conversations,
            })
        return converted_data
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert QA JSON files to conversational format and optionally upload to Hugging Face Hub.")
    parser.add_argument('--input_dir', type=str, default='data/qa_output', help='Directory containing the input JSON files.')
    parser.add_argument("--split_type", type=str, default='val', choices=["train", "val", "test"], help="Data split type.")
    parser.add_argument("--output_dir", default="data/vlm_3tp_data", help="Output directory for converted JSON files")
    parser.add_argument("--repo_name", type=str, default="", help="Hugging Face Hub repository name to upload the data.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert each JSON file in the input directory
    input_path = Path(args.input_dir) / args.split_type
    output_split_dir = Path(args.output_dir) / f"vsibench_{args.split_type}"
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入目录是否存在
    if not input_path.exists():
        logging.warning(f"Input directory {input_path} does not exist.")
        return
    
    # 转换所有JSON文件
    logging.info("Starting Convert JSON files to conversational format...")
    processed_files = 0
    for input_file in input_path.glob("*.json"):
        logging.info(f"Processing file: {input_file}")
        try:
            converted_data = convert_to_conversational_format(input_file)
            output_file = output_split_dir / f"{input_file.stem}_conversation.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=4, ensure_ascii=False)
            
            logging.info(f"Converted data saved to: {output_file}")
            processed_files += 1
            
        except Exception as e:
            logging.error(f"Error processing file {input_file}: {e}")
    
    if processed_files == 0:
        logging.warning("No JSON files found to process.")
        return
    
    logging.info(f"Successfully processed {processed_files} files.")
    
    # Optionally upload to Hugging Face Hub
    repo_name = args.repo_name
    if repo_name == "":
        logging.warning("No repository name provided. Skipping upload to Hugging Face Hub.")
    else:
        logging.info(f"Uploading to Hugging Face repository: {repo_name}")
        
        try:
            # 登录 Hugging Face (可能需要token)
            login()
            
            # 创建 API 实例
            api = HfApi()
            
            # 检查仓库是否存在，如果不存在则创建
            try:
                api.repo_info(repo_id=repo_name, repo_type="dataset")
                logging.info(f"Repository {repo_name} exists.")
            except Exception:
                logging.info(f"Creating new dataset repository: {repo_name}")
                api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
            
            # 上传整个转换后的文件夹
            logging.info("Uploading files...")
            api.upload_folder(
                folder_path=output_split_dir,
                repo_id=repo_name,
                repo_type="dataset",
                path_in_repo=f"data/vsibench_{args.split_type}",  # 在仓库中的路径
                commit_message=f"Upload {args.split_type} split conversational data",
                ignore_patterns=["*.pyc", "__pycache__"]  # 忽略不需要的文件
            )
            
            logging.info("---------------------")
            logging.info(f"✅ Successfully uploaded {output_split_dir} to {repo_name}")
            logging.info(f"   Repository URL: https://huggingface.co/datasets/{repo_name}")
            logging.info("---------------------")
            
        except Exception as e:
            logging.error("---------------------")
            logging.error(f"❌ Error uploading to Hugging Face: {e}")
            logging.error("---------------------")
            logging.error("\nPlease make sure:")
            logging.error("1. You are logged in with 'huggingface-cli login'")
            logging.error("2. You have write access to the repository")
            logging.error("3. The repository name is correct")


if __name__ == "__main__":
    main()

**`metadata.json`:**

A JSON file containing metadata for each scene.

```json
{
    "scene_name": {
        "description": "A brief description of the human pose action.",
        "joint_names": "List of joint names used in the dataset.",
        "frames": [
            {
                "frame_id": 0,
                "body_poses": [
                    [ // 22个SMPL3D人体关节点的轴角表示（包括根节点，不包括手指）
                        [x1, y1, z1],
                        [x2, y2, z2],
                        ...
                        [x22, y22, z22]
                    ],
                    ... // other bodies in the same frame
                ],
            },
            ... // other frames
        ],
    },
    ... // other scenes
}
```

## 1. Upload Processed Data to Hugging Face Hub

After generating the QA datasets, due to different tasks having different answer formats (multiple-choice, JSON, etc.), we provide a script `hf_upload.py` to convert these datasets into a conversational format and upload them to the Hugging Face Hub.

### Prerequisites

Ensure you have the required libraries installed:

```bash
pip install huggingface_hub
```

### Usage

- Just covert the generated QA JSON files to conversational format, and save them to a specified output directory.

    ```bash
    python hf_upload.py --input_dir data/qa_output --split_type val --output_dir data/vlm_pose3d_data
    ```

- Convert and upload the datasets to Hugging Face Hub. Make sure to set your repository name.

    ```bash
    python hf_upload.py --input_dir data/qa_output --split_type val --output_dir data/vlm_pose3d_data --repo_name your_hf_repo_name
    ```


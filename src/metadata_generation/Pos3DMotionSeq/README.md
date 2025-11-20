# 1. preprocess

This section outlines the initial steps to prepare the raw Pos3DMotionSeq dataset for further processing.

## 1.1 Download the Dataset

```text
├──Pos3DMotionSeq/
│   ├──data/
│   │   ├──video_name/
│   │   │   ├──description
│   │   │   |   ├──video_name.json
│   │   │   ├──pose3d/
│   │   │   │   ├──hmr4d_results_merged.json
│   │   ├──...
│   ├──Pos3DMotionSeq_train.txt
│   ├──Pos3DMotionSeq_val.txt
│   ├──Pos3DMotionSeq_test.txt
```

# 2. get metadata

This step gathers information from the preprocessed dataset. It typically includes details about the scenes name, the description of the human pose action, and the 3D joint axis-angle representations for each frame.

## 2.1 train

```bash
python -m src.metadata_generation.Pos3DMotionSeq.pos3dmotion_metadata \
    --processed_dir "data/Pose3DMotionSeq" \
    --scene_list_file "data/Pose3DMotionSeq/Pose3DMotionSeq_train.txt" \
    --save_dir "data/processed_data/Pose3DMotionSeq/train" \
    --output_filename "pos3dmotion_metadata_train.json" \
    --num_workers "64" \
    --overwrite \
    --random_seed "42"
```

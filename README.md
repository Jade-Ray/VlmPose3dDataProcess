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
                "body_pos": [ // 22个SMPL3D人体关节点的轴角表示（包括根节点，不包括手指）
                    [x1, y1, z1],
                    [x2, y2, z2],
                    ...
                    [x22, y22, z22]
                ],
            },
            ... // other frames
        ],
    },
    ... // other scenes
}
```

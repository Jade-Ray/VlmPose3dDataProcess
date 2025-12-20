SIMPLE_POSE_TEMPLATE = """
请根据以下人体动作描述生成连续 {NUM_FRAMES} 帧的 {NUM_OBJ} 个 3D 人体姿态轴角序列。
动作描述：{ACTION_DESC}
要求：
1. 每个人体骨架包含22个关节，顺序如下：
{JOINT_NAMES}
2. 每个关节以轴角表示，格式为 [ax, ay, az]，角度单位为弧度。
3. 每个人体 body_pose 的列表维度应为 [{NUM_FRAMES}, 22, 3]。
4. 若某帧无明显变化，可保持与前一帧一致。
5. 以 JSON 格式回答，形式如下：
```JSON
[{{"body_pose": [[[ax, ay, az], ... (22 joints)]], ... (NUM_FRAMES)]}}, ... (NUM_OBJ)]
```
""".strip()

SINGLE_POSE_TEMPLATE = """
请根据以下人体动作描述生成人体骨架。
动作描述：{ACTION_DESC}
要求：
1. 人体骨架包含22个关节，顺序如下：
{JOINT_NAMES}
2. 关节以三维坐标[x, y, z]表示，单位为米。
3. 根节点默认为零，生成其余21个关节的坐标。
4. 以 JSON 格式回答，形式如下：
```JSON
[[x, y, z], ... (21 joints)]
```
""".strip()

SIMPLE_POSE_BSPLINE_TEMPLATE = """
请根据以下人体动作描述生成人体骨架序列。
动作描述：{ACTION_DESC}
要求：
1. 人体骨架包含22个关节，顺序如下：
{JOINT_NAMES}
2. 关节变化以三次B样条曲线的控制点系数表示。
4. 以 JSON 格式回答，形式如下：
```JSON
[[[c1, ... (other coef)], ... (3 dim)], ... (22 joints)]
```
""".strip()
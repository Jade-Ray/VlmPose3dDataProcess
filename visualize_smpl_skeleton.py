import argparse
import json
import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation

"""
可视化说明:
- 读取 hmr4d_results_samplevideo.json 或 hmr4d_results.json
- 使用 pred_smpl_params_incam 中的 body_pose (N,63), global_orient (N,3), transl (N,3), betas (10,)
- 通过 smplx 的 smpl 模型前向计算每帧的 joints (J,3)
- 用 matplotlib 3D 绘制骨架, 支持多人, 支持动画播放
"""

"""
将结果保存为视频：python visualize_smpl_skeleton.py hmr4d_results_samplevideo.json --center_group --save_anim skeleton_animation.mp4 
在Linux中运行：python visualize_smpl_skeleton.py hmr4d_results_samplevideo.json --center_group

"""

MODEL_PATH = '/home/ssm/shanshan/text2dance/src/algorithms/pose3d_gvhmr/inputs/checkpoints/body_models/'


def load_person_params(person_data, max_frames=None):
    """从person字典中提取 pred_smpl_params_incam (或 smpl_params_incam)"""
    smpl_params = None
    if 'net_outputs' in person_data and 'pred_smpl_params_incam' in person_data['net_outputs']:
        smpl_params = person_data['net_outputs']['pred_smpl_params_incam']
    elif 'pred_smpl_params_incam' in person_data:
        smpl_params = person_data['pred_smpl_params_incam']
    elif 'smpl_params_incam' in person_data:
        smpl_params = person_data['smpl_params_incam']
    else:
        raise KeyError('未找到 pred_smpl_params_incam/smpl_params_incam 字段')

    body_pose = np.array(smpl_params['body_pose'], dtype=np.float32)
    global_orient = np.array(smpl_params['global_orient'], dtype=np.float32)
    transl = np.array(smpl_params['transl'], dtype=np.float32)
    betas = np.array(smpl_params['betas'], dtype=np.float32)

    if max_frames is not None:
        body_pose = body_pose[:max_frames]
        global_orient = global_orient[:max_frames]
        transl = transl[:max_frames]

    return body_pose, global_orient, transl, betas


def pad_body_pose(body_pose):
    """将 (N,63) 的 body_pose 末尾补零到 (N,69), 以匹配SMPL的23个关节(23*3=69)。"""
    # Handle cases where data might be wrapped in an extra dimension, e.g. (1, N, 63)
    if body_pose.ndim > 2:
        body_pose = body_pose.squeeze(0)

    # After squeezing, it must be 2D
    if body_pose.ndim != 2:
        raise ValueError(f"body_pose is expected to be a 2D array, but got shape {body_pose.shape}")

    N, D = body_pose.shape
    if D == 63:
        pad = np.zeros((N, 69 - 63), dtype=np.float32)
        return np.concatenate([body_pose, pad], axis=1)
    elif D == 69:
        return body_pose
    else:
        raise ValueError(f'body_pose 维度不符合预期: {D}, 期望 63 或 69')


def compute_group_center(all_person_params):
    """计算所有人的第一帧平移的平均值, 用于将整体居中至原点。"""
    initial_trans = []
    for body_pose, global_orient, transl, betas in all_person_params:
        if transl.ndim == 2 and transl.shape[0] > 0:
            initial_trans.append(transl[0])
    if initial_trans:
        return np.mean(initial_trans, axis=0)
    return np.zeros(3, dtype=np.float32)


def build_smpl_model():
    model = smplx.create(
        model_path=MODEL_PATH,
        model_type='smpl',
        gender='neutral',
        use_pca=False,
        batch_size=1
    )
    return model


def forward_joints(model, body_pose, global_orient, transl, betas):
    """批量前向, 返回 (N, J, 3) joints。"""
    # Handle extra dimensions for orient and transl, similar to pad_body_pose
    if global_orient.ndim > 2:
        global_orient = global_orient.squeeze(0)
    if transl.ndim > 2:
        transl = transl.squeeze(0)

    # 统一形状
    body_pose_full = torch.tensor(body_pose, dtype=torch.float32)  # (N, 69)
    global_orient = torch.tensor(global_orient, dtype=torch.float32)  # (N, 3)
    transl = torch.tensor(transl, dtype=torch.float32)  # (N, 3)

    # betas should be shape (1, 10) to be broadcasted by smplx
    betas = torch.tensor(betas, dtype=torch.float32)
    if betas.ndim > 2:
        betas = betas.squeeze(0)  # Handles (1, N, 10) -> (N, 10)
    if betas.ndim == 2:
        betas = betas[0:1, :]  # Take first frame's betas -> (1, 10)
    elif betas.ndim == 1:
        betas = betas.unsqueeze(0) # (10,) -> (1, 10)

    with torch.no_grad():
        out = model(global_orient=global_orient, body_pose=body_pose_full, transl=transl, betas=betas)
        joints = out.joints.detach().cpu().numpy()  # (N, J, 3)
    return joints


def get_edges_from_parents(model):
    """根据SMPL模型的kintree parents生成骨架的边列表。"""
    parents = model.parents.cpu().numpy().tolist()
    edges = []
    for i in range(1, len(parents)):
        p = parents[i]
        if p >= 0:
            edges.append((p, i))
    return edges


def set_axes_equal(ax, xyz_min, xyz_max):
    """设置3D坐标轴等比例显示。"""
    ranges = xyz_max - xyz_min
    max_range = ranges.max()
    mid = (xyz_max + xyz_min) / 2.0
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)


def main():
    parser = argparse.ArgumentParser(description='可视化SMPL骨架运动 (matplotlib)')
    parser.add_argument('input_json', help='JSON文件路径')
    parser.add_argument('--persons', type=str, default=None, help='要可视化的人物ID, 逗号分隔, 如 "1,2"; 默认全部')
    parser.add_argument('--max_frames', type=int, default=None, help='限制最大帧数, 避免过长动画')
    parser.add_argument('--center_group', action='store_true', help='将所有人的群体中心平移到世界原点')
    parser.add_argument('--save_svg', type=str, default=None, help='保存第一帧为SVG到该路径(可选)')
    parser.add_argument('--no_show', action='store_true', help='不展示窗口(仅保存SVG时使用)')
    parser.add_argument('--save_anim', type=str, default=None, help='将动画保存为视频文件(如 .mp4)')
    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    # 选择人物ID
    person_ids = list(data.keys())
    if args.persons:
        chosen = [pid.strip() for pid in args.persons.split(',')]
        person_ids = [pid for pid in person_ids if pid in chosen]
    if not person_ids:
        raise ValueError('未找到要可视化的人物ID')

    # 加载所有人的参数
    all_params = []
    for pid in person_ids:
        body_pose, global_orient, transl, betas = load_person_params(data[pid], max_frames=args.max_frames)
        body_pose = pad_body_pose(body_pose)
        all_params.append((body_pose, global_orient, transl, betas))

    # 计算群体中心
    group_center = compute_group_center(all_params) if args.center_group else np.zeros(3, dtype=np.float32)

    # 前向计算所有人的 joints
    model = build_smpl_model()
    edges = get_edges_from_parents(model)

    persons_joints = {}
    max_frames = 0
    xyz_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    xyz_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

    for (pid, (body_pose, global_orient, transl, betas)) in zip(person_ids, all_params):
        # 群体居中
        transl_centered = transl - group_center
        joints = forward_joints(model, body_pose, global_orient, transl_centered, betas)
        persons_joints[pid] = joints  # (T, J, 3)
        max_frames = max(max_frames, joints.shape[0])
        # 更新可视化范围
        xyz_min = np.minimum(xyz_min, joints.reshape(-1, 3).min(axis=0))
        xyz_max = np.maximum(xyz_max, joints.reshape(-1, 3).max(axis=0))

    # Matplotlib 3D 图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('SMPL Skeleton Motion')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax, xyz_min, xyz_max)
    # 通过添加下面两句代码使得视角改变
    ax.view_init(elev=90, azim=90)
    ax.invert_xaxis()  # 翻转X轴
    

    # 为每个人创建绘图对象
    color = (191/255.0, 202/255.0, 249/255.0)  # #BFCAF9
    person_lines = {pid: [] for pid in person_ids}
    person_scatters = {pid: None for pid in person_ids}

    for pid in person_ids:
        # 初始化线段
        for _ in edges:
            line, = ax.plot([], [], [], color=color, lw=2)
            person_lines[pid].append(line)
        # 初始化关节散点
        scatter = ax.scatter([], [], [], color=color, s=10)
        person_scatters[pid] = scatter

    def update(frame_idx):
        for pid in person_ids:
            joints = persons_joints[pid]
            if frame_idx >= joints.shape[0]:
                continue
            js = joints[frame_idx]  # (J,3)
            # 更新线段
            for k, (i, j) in enumerate(edges):
                xs = [js[i, 0], js[j, 0]]
                ys = [js[i, 1], js[j, 1]]
                zs = [js[i, 2], js[j, 2]]
                person_lines[pid][k].set_data(xs, ys)
                person_lines[pid][k].set_3d_properties(zs)
            # 更新关节散点
            person_scatters[pid]._offsets3d = (js[:, 0], js[:, 1], js[:, 2])
        return []

    # 可选: 保存第一帧为SVG
    if args.save_svg:
        update(0)
        plt.savefig(args.save_svg, format='svg', dpi=300)
        print(f'SVG已保存到: {args.save_svg}')
        if args.no_show:
            return

    ani = FuncAnimation(fig, update, frames=max_frames, interval=33, blit=False, repeat=True)

    # 保存或显示
    if args.save_anim:
        ani.save(args.save_anim, writer='ffmpeg', fps=30)
        print(f'动画已保存到: {args.save_anim}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
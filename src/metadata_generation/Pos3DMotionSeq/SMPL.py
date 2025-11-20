import numpy as np
import matplotlib.pyplot as plt


def get_edges_from_parents(parents: list | np.ndarray):
    """根据父节点索引数组生成骨架的边列表。"""
    edges = []
    for i in range(1, len(parents)):
        p = parents[i]
        if p >= 0:
            edges.append((p, i))
    return edges


class SMPL:
    """SMPL 3D Human Model Class"""
    NUM_JOINTS = 23 # 标准 SMPL 模型的关节数量（不包含根节点pelvis）
    JOINT_NAMES = [
        'pelvis',        # 0  (根节点, parent=-1)
        'left_hip',      # 1  (parent=0, pelvis的子节点)
        'right_hip',     # 2  (parent=0)
        'spine1',        # 3  (parent=0)
        'left_knee',     # 4  (parent=1, left_hip的子节点)
        'right_knee',    # 5  (parent=2)
        'spine2',        # 6  (parent=3)
        'left_ankle',    # 7  (parent=4)
        'right_ankle',   # 8  (parent=5)
        'spine3',        # 9  (parent=6)
        'left_foot',     # 10 (parent=7)
        'right_foot',    # 11 (parent=8)
        'neck',          # 12 (parent=9)
        'left_collar',   # 13 (parent=9)
        'right_collar',  # 14 (parent=9)
        'head',          # 15 (parent=12)
        'left_shoulder', # 16 (parent=13)
        'right_shoulder',# 17 (parent=14)
        'left_elbow',    # 18 (parent=16)
        'right_elbow',   # 19 (parent=17)
        'left_wrist',    # 20 (parent=18)
        'right_wrist',   # 21 (parent=19)
        'left_hand',     # 22 (parent=20)
        'right_hand',    # 23 (parent=21)
    ]
    PARENT_INDICES = np.array([
        -1,  # 0: pelvis (根节点)
        0,   # 1: left_hip -> pelvis
        0,   # 2: right_hip -> pelvis
        0,   # 3: spine1 -> pelvis
        1,   # 4: left_knee -> left_hip
        2,   # 5: right_knee -> right_hip
        3,   # 6: spine2 -> spine1
        4,   # 7: left_ankle -> left_knee
        5,   # 8: right_ankle -> right_knee
        6,   # 9: spine3 -> spine2
        7,   # 10: left_foot -> left_ankle
        8,   # 11: right_foot -> right_ankle
        9,   # 12: neck -> spine3
        9,   # 13: left_collar -> spine3
        9,   # 14: right_collar -> spine3
        12,  # 15: head -> neck
        13,  # 16: left_shoulder -> left_collar
        14,  # 17: right_shoulder -> right_collar
        16,  # 18: left_elbow -> left_shoulder
        17,  # 19: right_elbow -> right_shoulder
        18,  # 20: left_wrist -> left_elbow
        19,  # 21: right_wrist -> right_elbow
        20,  # 22: left_hand -> left_wrist
        21,  # 23: right_hand -> right_wrist
    ])
    # SMPL标准T-pose关节位置 (基于SMPL neutral模型)
    # 单位: 米, 坐标系: Y轴向上, X轴向右, Z轴向前
    TEMPLATE_TPOSE = np.array([
        # 0: Pelvis (根节点,原点)
        [0.0000, 0.0000, 0.0000],
        
        # 1-2: 髋部 (Hip joints) - 左右对称
        [-0.0895, -0.0597, 0.0036],   # 1: left_hip
        [0.0895, -0.0597, 0.0036],    # 2: right_hip
        
        # 3: 脊柱底部 (Spine1)
        [-0.0020, 0.0774, -0.0102],   # 3: spine1
        
        # 4-5: 膝盖 (Knees) - 大腿长度约0.39m
        [-0.0905, -0.4504, 0.0080],   # 4: left_knee
        [0.0905, -0.4504, 0.0080],    # 5: right_knee
        
        # 6: 脊柱中段 (Spine2)
        [0.0003, 0.1926, -0.0125],    # 6: spine2
        
        # 7-8: 脚踝 (Ankles) - 小腿长度约0.42m
        [-0.0838, -0.8671, -0.0181],  # 7: left_ankle
        [0.0838, -0.8671, -0.0181],   # 8: right_ankle
        
        # 9: 脊柱上段 (Spine3)
        [-0.0012, 0.3860, 0.0176],    # 9: spine3
        
        # 10-11: 脚部 (Feet) - 脚长约0.10m
        [-0.0679, -0.9672, 0.0665],   # 10: left_foot
        [0.0679, -0.9672, 0.0665],    # 11: right_foot
        
        # 12: 颈部 (Neck)
        [0.0059, 0.5674, 0.0350],     # 12: neck
        
        # 13-14: 锁骨 (Collars)
        [-0.0774, 0.4492, 0.0301],    # 13: left_collar
        [0.0774, 0.4492, 0.0301],     # 14: right_collar
        
        # 15: 头部 (Head) - 颈部到头顶约0.16m
        [0.0118, 0.7283, 0.0561],     # 15: head
        
        # 16-17: 肩膀 (Shoulders) - 肩宽约0.38m
        [-0.1766, 0.4490, 0.0297],    # 16: left_shoulder
        [0.1766, 0.4490, 0.0297],     # 17: right_shoulder
        
        # 18-19: 肘部 (Elbows) - 上臂长度约0.28m
        [-0.4230, 0.3623, 0.0398],    # 18: left_elbow
        [0.4230, 0.3623, 0.0398],     # 19: right_elbow
        
        # 20-21: 手腕 (Wrists) - 前臂长度约0.26m
        [-0.6656, 0.2759, 0.0495],    # 20: left_wrist
        [0.6656, 0.2759, 0.0495],     # 21: right_wrist
        
        # 22-23: 手部 (Hands) - 手长约0.09m
        [-0.7565, 0.2456, 0.0533],    # 22: left_hand
        [0.7565, 0.2456, 0.0533],     # 23: right_hand
    ])
    SKELETON_CONNECTIONS = get_edges_from_parents(PARENT_INDICES)
    
    @classmethod
    def _check_body_pose(cls, body_pose: np.ndarray):
        """检查并调整body_pose的形状，确保为(J, 3)"""
        if body_pose.ndim not in [1, 2]:
            raise ValueError(f"body_pose should be 1D or 2D array, but got shape {body_pose.shape}")
        if body_pose.ndim == 1:
            if body_pose.shape[0] != cls.JOINT_NUMS * 3:
                raise ValueError(f"1D body_pose should have shape ({cls.JOINT_NUMS * 3},), but got {body_pose.shape}")
            body_pose = body_pose.reshape((cls.JOINT_NUMS, 3))
        if body_pose.shape != (cls.JOINT_NUMS, 3):
            raise ValueError(f"2D body_pose should have shape ({cls.JOINT_NUMS}, 3), but got {body_pose.shape}")
        return body_pose
    
    @classmethod
    def forward_joints(
        cls, 
        body_pose: np.ndarray,
        global_orient: np.ndarray = None,
        transl: np.ndarray = None,
        betas: np.ndarray = None,
    ) -> np.ndarray:
        """
        简化版SMPL前向传播,模拟smplx.lbs的核心流程
        
        Args:
            body_pose: (J-1, 3) 或 ((J-1)*3,) 身体姿态(不含根关节)
            global_orient: (3,) 根关节全局旋转
            transl: (3,) 全局平移
            betas: (10,) 形状参数
        
        Returns:
            joints_3d: (J, 3) 3D关节位置
        """
        body_pose = cls._check_body_pose(body_pose)
        
        # 1. Shape Blending: 应用betas调整T-pose
        v_shaped = cls.TEMPLATE_TPOSE.copy()  # (J, 3)
        if betas is not None and len(betas) > 0:
            # 简化: 只用第一个beta作为整体缩放
            shape_scale = 1.0 + betas[0] * 0.1
            v_shaped = v_shaped * shape_scale
        
        # 2. 从T-pose顶点回归关节位置(模拟J_regressor)
        # 简化: 直接使用v_shaped作为关节位置
        J = v_shaped  # (J, 3) - 在T-pose下的关节位置
        
        # 3. 合并global_orient和body_pose
        if global_orient is None:
            global_orient = np.zeros(3)
        full_pose = np.vstack([global_orient.reshape(1, 3), body_pose])  # (J, 3)
        
        # 4. Pose Blend Shapes (简化版: 跳过)
        # 标准SMPL会在这里添加姿态相关的形变
        # pose_offsets = ...
        # v_posed = v_shaped + pose_offsets
        v_posed = v_shaped  # 简化版直接使用v_shaped
        
        # 5. Forward Kinematics: 批量刚性变换
        J_transformed = cls._batch_rigid_transform(full_pose, J)
        
        # 6. 应用全局平移
        if transl is not None:
            J_transformed = J_transformed + transl.reshape(1, 3)
        
        return J_transformed
    
    @classmethod
    def _batch_rigid_transform(
        cls,
        rot_vecs: np.ndarray,  # (J, 3) 轴角
        joints: np.ndarray,    # (J, 3) T-pose关节位置
    ) -> np.ndarray:
        """
        批量刚性变换 - 模拟smplx.lbs.batch_rigid_transform
        
        核心思想:
        1. 计算相对父节点的骨骼向量(rel_joints)
        2. 沿kinematic tree累积变换
        3. 提取变换后的关节位置
        """
        num_joints = len(cls.PARENT_INDICES)
        
        # Step 1: 计算相对关节位置(骨骼向量)
        rel_joints = joints.copy()
        for i in range(1, num_joints):
            parent_idx = cls.PARENT_INDICES[i]
            if parent_idx >= 0:
                rel_joints[i] = joints[i] - joints[parent_idx]
        # rel_joints[0] 保持不变(根节点的绝对位置)
        
        # Step 2: 构建局部变换矩阵
        transforms_mat = np.zeros((num_joints, 4, 4))
        for i in range(num_joints):
            rot_mat = cls._axis_angle_to_matrix(rot_vecs[i])
            transforms_mat[i, :3, :3] = rot_mat
            transforms_mat[i, :3, 3] = rel_joints[i]  # 平移 = 骨骼向量
            transforms_mat[i, 3, 3] = 1.0
        
        # Step 3: 沿kinematic tree累积全局变换
        transform_chain = [transforms_mat[0]]  # 根节点
        for i in range(1, num_joints):
            parent_idx = cls.PARENT_INDICES[i]
            # 全局变换 = 父节点全局变换 @ 当前局部变换
            curr_transform = transform_chain[parent_idx] @ transforms_mat[i]
            transform_chain.append(curr_transform)
        
        transforms = np.stack(transform_chain, axis=0)  # (J, 4, 4)
        
        # Step 4: 提取变换后的关节位置(变换矩阵的最后一列前3行)
        posed_joints = transforms[:, :3, 3]  # (J, 3)
        
        return posed_joints
    
    @classmethod
    def _axis_angle_to_matrix(cls, axis_angle: np.ndarray) -> np.ndarray:
        """
        轴角 -> 旋转矩阵 (Rodrigues公式)
        对应 smplx.lbs.batch_rodrigues
        """
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-8:
            return np.eye(3)
        
        # 单位旋转轴
        axis = axis_angle / angle
        
        # 反对称矩阵 K
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        return R


class SMPLVisualizerBase:
    """SMPL 3D Skeleton Visualization Base Class"""
    BODY_PART_COLORS = {
        'torso': '#FF6B6B',
        'left_leg': '#4ECDC4',
        'right_leg': '#45B7D1',
        'left_arm': '#FFA07A',
        'right_arm': '#98D8C8',
    }
    
    @classmethod
    def visualize_skeleton(
        cls, 
        joints_3d: np.ndarray,
        skeleton_connections: list,
        joint_names: list = None,
        show_joint_labels: bool = False,
        title: str = "SMPL Skeleton",
        fig_size: tuple = (12, 10),
        skeleton_linewidth: int = 3,
        skeleton_alpha: float = 0.8,
        joint_linewidth: int = 1,
        joint_alpha: float = 0.6,
    ):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制骨架连接
        for connection in skeleton_connections:
            joint_start, joint_end = connection
            xs = [joints_3d[joint_start, 0], joints_3d[joint_end, 0]]
            ys = [joints_3d[joint_start, 1], joints_3d[joint_end, 1]]
            zs = [joints_3d[joint_start, 2], joints_3d[joint_end, 2]]
            ax.plot(xs, ys, zs, c=cls._get_connection_color(connection), linewidth=skeleton_linewidth, alpha=skeleton_alpha)
            
        # 绘制关节点
        ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c='red', s=50, alpha=joint_alpha, edgecolors='black', linewidth=joint_linewidth)
        
        # 显示关节点标签
        if show_joint_labels and joint_names is not None:
            for i, joint_name in enumerate(joint_names):
                ax.text(joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2], joint_name, fontsize=8, alpha=0.7)
        
        cls._set_axes_equal(ax, joints_3d)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.view_init(elev=10, azim=45)
        
        plt.tight_layout()
        plt.show()
    
    @classmethod
    def animate_skeleton_sequence(
        cls, 
        joints_3d_sequence: list,
        skeleton_connections: list,
        joint_names: list = None,
        show_joint_labels: bool = False,
        fps: int = 30,
        save_path: str = None,
        fig_size: tuple = (8, 8),
        skeleton_linewidth: int = 3,
        skeleton_alpha: float = 0.8,
        joint_linewidth: int = 1,
        joint_alpha: float = 0.6,
    ):
        from matplotlib.animation import FuncAnimation
        
        num_frames = len(joints_3d_sequence)
        all_joints = np.array(joints_3d_sequence)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        lines = []
        for connection in skeleton_connections:
            line, = ax.plot([], [], [], c=cls._get_connection_color(connection), linewidth=skeleton_linewidth, alpha=skeleton_alpha)
            lines.append(line)
        
        scatter = ax.scatter([], [], [], c='red', s=50, alpha=joint_alpha, edgecolors='black', linewidth=joint_linewidth)
        title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, ha='center', fontsize=14)
        
        labels = []
        if show_joint_labels:
            for i, joint_name in enumerate(joint_names):
                label = ax.text(0, 0, 0, joint_name, fontsize=8, alpha=0.7)
                labels.append(label)
        
        # 设置坐标轴范围（基于所有帧）
        all_points = all_joints.reshape(-1, 3)
        cls._set_axes_equal(ax, all_points)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            scatter._offsets3d = ([], [], [])
            title.set_text('')
            if show_joint_labels:
                for label in labels:
                    label.set_position((0, 0))
                    label.set_3d_properties(0, zdir='z')
            return lines + [scatter, title] + (labels if show_joint_labels else [])
        
        def update(frame):
            joints = all_joints[frame]
            for i, connection in enumerate(skeleton_connections):
                joint_start, joint_end = connection
                xs = [joints[joint_start, 0], joints[joint_end, 0]]
                ys = [joints[joint_start, 1], joints[joint_end, 1]]
                zs = [joints[joint_start, 2], joints[joint_end, 2]]
                lines[i].set_data(xs, ys)
                lines[i].set_3d_properties(zs)
        
            # 更新关节点
            scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
            
            if show_joint_labels:
                for i, label in enumerate(labels):
                    label.set_position((joints[i, 0], joints[i, 1]))
                    label.set_3d_properties(joints[i, 2], zdir='z')
            
            # 更新标题
            title.set_text(f'Frame {frame + 1}/{num_frames}')
            
            # 旋转视角（可选）
            ax.view_init(elev=10, azim=45 + frame * 0.5)
            
            return lines + [scatter, title] + (labels if show_joint_labels else [])
        
        # 创建动画
        anim = FuncAnimation(fig, update, frames=num_frames, 
                           init_func=init, blit=False,
                           interval=1000/fps, repeat=True)
        
        # 保存动画
        if save_path:
            print(f"Saving animation to {save_path}...")
            if save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
            elif save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            print("Animation saved!")
        
        plt.show()
        return anim
    
    @classmethod
    def _get_connection_color(cls, connection):
        """根据连接关系返回颜色"""
        pass
    
    @classmethod
    def _set_axes_equal(cls, ax, points):
        """设置3D坐标轴等比例"""
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    

class SMPL21Joint(SMPL):
    """SMPL 3D Human Model with 21 Joints (without hands)"""
    JOINT_NUMS = 21
    JOINT_NAMES = [
        'pelvis',        # 0  (根节点, parent=-1)
        'left_hip',      # 1  (parent=0, pelvis的子节点)
        'right_hip',     # 2  (parent=0)
        'spine1',        # 3  (parent=0)
        'left_knee',     # 4  (parent=1, left_hip的子节点)
        'right_knee',    # 5  (parent=2)
        'spine2',        # 6  (parent=3)
        'left_ankle',    # 7  (parent=4)
        'right_ankle',   # 8  (parent=5)
        'spine3',        # 9  (parent=6)
        'left_foot',     # 10 (parent=7)
        'right_foot',    # 11 (parent=8)
        'neck',          # 12 (parent=9)
        'left_collar',   # 13 (parent=9)
        'right_collar',  # 14 (parent=9)
        'head',          # 15 (parent=12)
        'left_shoulder', # 16 (parent=13)
        'right_shoulder',# 17 (parent=14)
        'left_elbow',    # 18 (parent=16)
        'right_elbow',   # 19 (parent=17)
        'left_wrist',    # 20 (parent=18)
        'right_wrist',   # 21 (parent=19)
    ]
    PARENT_INDICES = np.array([
        -1,  # 0: pelvis (根节点)
        0,   # 1: left_hip -> pelvis
        0,   # 2: right_hip -> pelvis
        0,   # 3: spine1 -> pelvis
        1,   # 4: left_knee -> left_hip
        2,   # 5: right_knee -> right_hip
        3,   # 6: spine2 -> spine1
        4,   # 7: left_ankle -> left_knee
        5,   # 8: right_ankle -> right_knee
        6,   # 9: spine3 -> spine2
        7,   # 10: left_foot -> left_ankle
        8,   # 11: right_foot -> right_ankle
        9,   # 12: neck -> spine3
        9,   # 13: left_collar -> spine3
        9,   # 14: right_collar -> spine3
        12,  # 15: head -> neck
        13,  # 16: left_shoulder -> left_collar
        14,  # 17: right_shoulder -> right_collar
        16,  # 18: left_elbow -> left_shoulder
        17,  # 19: right_elbow -> right_shoulder
        18,  # 20: left_wrist -> left_elbow
        19,  # 21: right_wrist -> right_elbow
    ])
    TEMPLATE_TPOSE = np.array([
        # 0: Pelvis (根节点,原点)
        [0.0000, 0.0000, 0.0000],
        
        # 1-2: 髋部 (Hip joints) - 左右对称
        [-0.0895, -0.0597, 0.0036],   # 1: left_hip
        [0.0895, -0.0597, 0.0036],    # 2: right_hip
        
        # 3: 脊柱底部 (Spine1)
        [-0.0020, 0.0774, -0.0102],   # 3: spine1
        
        # 4-5: 膝盖 (Knees) - 大腿长度约0.39m
        [-0.0905, -0.4504, 0.0080],   # 4: left_knee
        [0.0905, -0.4504, 0.0080],    # 5: right_knee
        
        # 6: 脊柱中段 (Spine2)
        [0.0003, 0.1926, -0.0125],    # 6: spine2
        
        # 7-8: 脚踝 (Ankles) - 小腿长度约0.42m
        [-0.0838, -0.8671, -0.0181],  # 7: left_ankle
        [0.0838, -0.8671, -0.0181],   # 8: right_ankle
        
        # 9: 脊柱上段 (Spine3)
        [-0.0012, 0.3860, 0.0176],    # 9: spine3
        
        # 10-11: 脚部 (Feet) - 脚长约0.10m
        [-0.0679, -0.9672, 0.0665],   # 10: left_foot
        [0.0679, -0.9672, 0.0665],    # 11: right_foot
        
        # 12: 颈部 (Neck)
        [0.0059, 0.5674, 0.0350],     # 12: neck
        
        # 13-14: 锁骨 (Collars)
        [-0.0774, 0.4492, 0.0301],    # 13: left_collar
        [0.0774, 0.4492, 0.0301],     # 14: right_collar
        
        # 15: 头部 (Head) - 颈部到头顶约0.16m
        [0.0118, 0.7283, 0.0561],     # 15: head
        
        # 16-17: 肩膀 (Shoulders) - 肩宽约0.38m
        [-0.1766, 0.4490, 0.0297],    # 16: left_shoulder
        [0.1766, 0.4490, 0.0297],     # 17: right_shoulder
        
        # 18-19: 肘部 (Elbows) - 上臂长度约0.28m
        [-0.4230, 0.3623, 0.0398],    # 18: left_elbow
        [0.4230, 0.3623, 0.0398],     # 19: right_elbow
        
        # 20-21: 手腕 (Wrists) - 前臂长度约0.26m
        [-0.6656, 0.2759, 0.0495],    # 20: left_wrist
        [0.6656, 0.2759, 0.0495],     # 21: right_wrist
    ])
    SKELETON_CONNECTIONS = get_edges_from_parents(PARENT_INDICES)
    

class SMPL21Visualizer(SMPL21Joint, SMPLVisualizerBase):
    """SMPL-21 3D Skeleton Visualizer"""
    @classmethod
    def visualize_skeleton(
        cls, 
        joints_3d: np.ndarray,
        show_joint_labels: bool = True,
        title: str = "SMPL-21 Skeleton",
        **kwargs
    ):
        """可视化单帧3D骨架
        Args:
            joints_3d: (21, 3) 3D关节位置
            show_joint_labels: 是否显示关节点标签
            title: 图表标题
            **kwargs: 其他可选参数传递给基础可视化方法
        """
        super().visualize_skeleton(
            joints_3d=joints_3d,
            skeleton_connections=cls.SKELETON_CONNECTIONS,
            joint_names=cls.JOINT_NAMES,
            show_joint_labels=show_joint_labels,
            title=title,
            **kwargs
        )
    
    @classmethod
    def animate_skeleton_sequence(
        cls, 
        joints_3d_sequence: list,
        show_joint_labels: bool = False,
        save_path: str = None,
        **kwargs
    ):
        """动画展示3D骨架序列
        Args:
            joints_3d_sequence: 3D关节位置序列列表
            show_joint_labels: 是否显示关节点标签
            save_path: 保存动画的路径（可选）
            **kwargs: 其他可选参数传递给基础动画方法
        """
        return super().animate_skeleton_sequence(
            joints_3d_sequence=joints_3d_sequence,
            skeleton_connections=cls.SKELETON_CONNECTIONS,
            joint_names=cls.JOINT_NAMES,
            show_joint_labels=show_joint_labels,
            save_path=save_path,
            **kwargs
        )
    
    @classmethod
    def _get_connection_color(cls, connection):
        """根据连接关系返回颜色"""
        parent, child = connection
        if parent in [0, 3, 6, 9, 12]:
            return cls.BODY_PART_COLORS['torso']
        elif parent in [1, 4, 7]:
            return cls.BODY_PART_COLORS['left_leg']
        elif parent in [2, 5, 8]:
            return cls.BODY_PART_COLORS['right_leg']
        elif parent in [13, 16, 18]:
            return cls.BODY_PART_COLORS['left_arm']
        elif parent in [14, 17]:
            return cls.BODY_PART_COLORS['right_arm']
        return '#333333'


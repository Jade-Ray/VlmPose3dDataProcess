from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SMPLModel:
    """SMPL 绑定姿态模型数据类"""
    # 绑定姿态关节名称列表
    joint_names: list
    # 绑定姿态关节父节点索引数组
    parent_indices: np.ndarray
    # 绑定姿态关节位置。单位: 米, 坐标系: Y轴向上, X轴向右, Z轴向前。
    rest_joints: np.ndarray
    

def get_template_tpose_smpl() -> SMPLModel:
    """获取SMPL-neutral模型的标准T-pose绑定姿态数据。"""
    return SMPLModel(
        joint_names=[
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
        ],
        parent_indices=np.array([
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
        ]),
        rest_joints=np.array([
            # 0: Pelvis (根节点,原点)
            [-1.79505953e-03, -2.23333446e-01, 2.82191255e-02],
            # 1-2: 髋部 (Hip joints)
            [ 6.77246757e-02, -3.14739671e-01,  2.14037877e-02],  # 1: left_hip
            [-6.94655406e-02, -3.13855126e-01,  2.38993038e-02],  # 2: right_hip
            # 3: 脊柱底部 (Spine1)
            [-4.32792313e-03, -1.14370215e-01,  1.52281192e-03],  # 3: spine1
            # 4-5: 膝盖 (Knees)
            [ 1.02001221e-01, -6.89938274e-01,  1.69079858e-02],  # 4: left_knee
            [-1.07755594e-01, -6.96424140e-01,  1.50492738e-02],  # 5: right_knee
            # 6: 脊柱中段 (Spine2)
            [1.15910534e-03,  2.08102144e-02,  2.61528404e-03],   # 6: spine2
            # 7-8: 脚踝 (Ankles)
            [8.84055199e-02, -1.08789863e+00, -2.67853442e-02],   # 7: left_ankle
            [-9.19818258e-02, -1.09483879e+00, -2.72625243e-02],  # 8: right_ankle
            # 9: 脊柱上段 (Spine3)
            [2.61610388e-03,  7.37324481e-02,  2.80398521e-02],   # 9: spine3
            # 10-11: 脚部 (Feet)
            [1.14763659e-01, -1.14368952e+00,  9.25030544e-02],   # 10: left_foot
            [-1.17353574e-01, -1.14298274e+00,  9.60854266e-02],  # 11: right_foot
            # 12: 颈部 (Neck)
            [-1.62284535e-04,  2.87602804e-01, -1.48171829e-02],  # 12: neck
            # 13-14: 锁骨 (Collars)
            [8.14608431e-02,  1.95481750e-01, -6.04975478e-03],   # 13: left_collar
            [-7.91430834e-02,  1.92565283e-01, -1.05754332e-02],  # 14: right_collar
            # 15: 头部 (Head)
            [4.98955543e-03,  3.52572414e-01,  3.65317875e-02],   # 15: head
            # 16-17: 肩膀 (Shoulders)
            [1.72437770e-01,  2.25950646e-01, -1.49179062e-02],   # 16: left_shoulder
            [-1.75155461e-01,  2.25116450e-01, -1.97185045e-02],  # 17: right_shoulder
            # 18-19: 肘部 (Elbows)
            [4.32050017e-01,  2.13178586e-01, -4.23743412e-02],   # 18: left_elbow
            [-4.28897421e-01,  2.11787231e-01, -4.11194829e-02],  # 19: right_elbow
            # 20-21: 手腕 (Wrists)
            [6.81283645e-01,  2.22164620e-01, -4.35452575e-02],   # 20: left_wrist
            [-6.84195501e-01,  2.19559526e-01, -4.66786778e-02],  # 21: right_wrist
            # 22-23: 手部 (Hands)
            [7.65325829e-01,  2.14003084e-01, -5.84906248e-02],   # 22: left_hand
            [-7.68817426e-01,  2.13442268e-01, -5.69937621e-02],  # 23: right_hand
        ])
    )


def get_edges_from_parents(parents: list | np.ndarray):
    """根据父节点索引数组生成骨架的边列表。"""
    edges = []
    for i in range(1, len(parents)):
        p = parents[i]
        if p >= 0:
            edges.append((p, i))
    return edges


def rodrigues_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rodrigues旋转公式: 绕任意轴旋转
        
    R = I + sin(θ)K + (1-cos(θ))K²
    """
    axis = axis / np.linalg.norm(axis)
    # 反对称矩阵 K
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
        
    # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
    return R


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """轴角向量转换为旋转矩阵 (Rodrigues公式)"""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3)
    # 单位旋转轴
    axis = axis_angle / angle
        
    return rodrigues_rotation_matrix(axis, angle)


def rotation_matrix_to_axis_angle(rotation_matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵转换为轴角向量"""
    # 计算旋转角度
    angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    # 计算旋转轴单位向量
    rx = rotation_matrix[2, 1] - rotation_matrix[1, 2]
    ry = rotation_matrix[0, 2] - rotation_matrix[2, 0]
    rz = rotation_matrix[1, 0] - rotation_matrix[0, 1]
    axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
    return axis * angle


class SMPL:
    """SMPL 3D Human Model Class"""
    NUM_JOINTS = 23 # 标准 SMPL 模型的关节数量（不包含根节点pelvis）
    JOINT_NAMES = get_template_tpose_smpl().joint_names
    PARENT_INDICES = get_template_tpose_smpl().parent_indices
    TEMPLATE_TPOSE = get_template_tpose_smpl().rest_joints
    SKELETON_CONNECTIONS = get_edges_from_parents(PARENT_INDICES)
    
    @classmethod
    def _check_body_pose(cls, body_pose: np.ndarray):
        """检查并调整body_pose的形状，确保为(J, 3)"""
        if body_pose.ndim not in [1, 2]:
            raise ValueError(f"body_pose should be 1D or 2D array, but got shape {body_pose.shape}")
        if body_pose.ndim == 1:
            if body_pose.shape[0] != cls.NUM_JOINTS * 3:
                raise ValueError(f"1D body_pose should have shape ({cls.NUM_JOINTS * 3},), but got {body_pose.shape}")
            body_pose = body_pose.reshape((cls.NUM_JOINTS, 3))
        if body_pose.shape != (cls.NUM_JOINTS, 3):
            raise ValueError(f"2D body_pose should have shape ({cls.NUM_JOINTS}, 3), but got {body_pose.shape}")
        return body_pose
    
    @classmethod
    def forward_joints(
        cls, 
        body_pose: np.ndarray,
        global_orient: np.ndarray = None,
        rest_joints: np.ndarray = None,
        parent_indices: np.ndarray = None,
    ) -> np.ndarray:
        """
        简化版SMPL前向传播,模拟smplx.lbs的核心流程
        
        Args:
            body_pose: (J-1, 3) 或 ((J-1)*3,) 身体姿态(不含根关节)
            global_orient: (3,) 根节点相对于世界坐标的旋转向量(轴角表示)
            rest_joints: (J, 3) 绑定姿态的关节位置 (可选,默认使用类内置T-pose)
            parent_indices: (J,) 绑定姿态的关节父节点索引 (可选,默认使用类内置父节点索引)
        
        Returns:
            joints_3d: (J, 3) 3D关节位置
        """
        body_pose = cls._check_body_pose(body_pose)
        
        if rest_joints is None:
            rest_joints = cls.TEMPLATE_TPOSE
        if parent_indices is None:
            parent_indices = cls.PARENT_INDICES
        
        # 1. 简化: 直接使用T-pose关节位置
        J = rest_joints.copy()  # (J, 3)
        
        # 2. 合并global_orient和body_pose
        if global_orient is None:
            global_orient = np.zeros(3)
        full_pose = np.vstack([global_orient.reshape(1, 3), body_pose])  # (J, 3)
        
        # 3. 简化: 跳过标准SMPL添加姿态相关的形变
        
        # 4. 层级前向传播，计算全局变换矩阵
        J_transformed = cls._batch_rigid_transform(full_pose, J, parent_indices)  # (J, 3)
        
        return J_transformed
    
    @classmethod
    def _batch_rigid_transform(
        cls,
        rot_vecs: np.ndarray,  # (J, 3) 轴角
        joints: np.ndarray,    # (J, 3) T-pose关节位置
        parent_indices: np.ndarray, # (J,) 父节点索引
    ) -> np.ndarray:
        """
        批量刚性变换 - 模拟smplx.lbs.batch_rigid_transform
        
        核心思想:
        1. 计算相对父节点的骨骼向量(rel_joints)
        2. 沿kinematic tree累积变换
        3. 提取变换后的关节位置
        """
        num_joints = len(parent_indices)
        
        # Step 1: 计算相对关节位置(骨骼向量)
        rel_joints = joints.copy()
        for i in range(1, num_joints):
            parent_idx = parent_indices[i]
            if parent_idx >= 0:
                rel_joints[i] = joints[i] - joints[parent_idx]
        # rel_joints[0] 保持不变(根节点的绝对位置)
        
        # Step 2: 构建局部变换矩阵
        transforms_mat = np.zeros((num_joints, 4, 4))
        for i in range(num_joints):
            rot_mat = axis_angle_to_rotation_matrix(rot_vecs[i])
            transforms_mat[i, :3, :3] = rot_mat
            transforms_mat[i, :3, 3] = rel_joints[i]  # 平移 = 骨骼向量
            transforms_mat[i, 3, 3] = 1.0
        
        # Step 3: 沿kinematic tree累积全局变换
        transform_chain = [transforms_mat[0]]  # 根节点
        for i in range(1, num_joints):
            parent_idx = parent_indices[i]
            # 全局变换 = 父节点全局变换 @ 当前局部变换
            curr_transform = transform_chain[parent_idx] @ transforms_mat[i]
            transform_chain.append(curr_transform)
        
        transforms = np.stack(transform_chain, axis=0)  # (J, 4, 4)
        
        # Step 4: 提取变换后的关节位置(变换矩阵的最后一列前3行)
        posed_joints = transforms[:, :3, 3]  # (J, 3)
        
        return posed_joints

    @classmethod
    def joints_to_axis_angles(
        cls, 
        joints: np.ndarray,  
        rest_joints: np.ndarray = None,
        parent_indices: np.ndarray = None
    ) -> np.ndarray:
        """
        简化版的逆向运动学，将关节3D坐标(未知绑定姿态)转换到指定绑定姿态下的轴角表示姿态。实现不同骨架系统间的动作重定向。
        
        Args:
            joints: (J, 3) 关节3D坐标（x, y, z）
            rest_joints: (J, 3) 绑定姿态的关节位置 (可选,默认使用类内置T-pose)
            parent_indices: (J,) 绑定姿态的关节父节点索引 (可选,默认使用类内置父节点索引)
        
        Returns:
            axis_angles: (J, 3) 轴角表示的旋转向量
        """
        if rest_joints is None:
            rest_joints = cls.TEMPLATE_TPOSE
        if parent_indices is None:
            parent_indices = cls.PARENT_INDICES
        
        num_joints = cls.NUM_JOINTS + 1 # 包含根节点
        assert len(joints) == num_joints, f"Expected {num_joints} joints, but got {len(joints)}"
        
        axis_angles = np.zeros((num_joints, 3))
        # 父节点的全局旋转矩阵列表，I表示相对父节点无旋转
        global_rotations = [np.eye(3) for _ in range(num_joints)]
        
        for child_idx, parent_idx in enumerate(parent_indices):
            if child_idx == 0 and parent_idx == -1:
                # 根节点(pelvis)旋转向量需要单独估计
                global_rotation = cls._estimate_pelvis_rotation(joints, rest_joints)
                global_rotations[0] = global_rotation
                axis_angles[0] = rotation_matrix_to_axis_angle(global_rotation)
                continue  
            
            # 1. 计算绑定姿势下的骨骼向量(局部坐标系)并归一化
            # 由于绑定姿态下各父节点无旋转，局部坐标系即世界坐标系
            bone_rest_local = rest_joints[child_idx] - rest_joints[parent_idx]
            bone_rest_local = bone_rest_local / np.linalg.norm(bone_rest_local)
            
            # 2.1 计算当前观测骨骼向量(世界坐标系)
            bone_current_world = joints[child_idx] - joints[parent_idx]
            # 2.2 转换到父节点局部坐标系并归一化
            bone_current_local = np.dot(global_rotations[parent_idx].T, bone_current_world)
            bone_length_current = np.linalg.norm(bone_current_local)
            assert bone_length_current > 1e-8, f"Bone length too small for joint {child_idx}->{parent_idx}"
            bone_current_local = bone_current_local / bone_length_current
            
            # 3 计算局部旋转(相对绑定姿势)，R * rest_local = current_local
            local_axis_angle = cls._rotation_between_vectors(bone_rest_local, bone_current_local)
            axis_angles[child_idx] = local_axis_angle
            
            # 4 更新该关节的全局旋转(用于后续子节点)
            local_rotation = axis_angle_to_rotation_matrix(local_axis_angle)
            global_rotations[child_idx] = global_rotations[parent_idx] @ local_rotation
        
        return axis_angles
    
    @classmethod
    def _estimate_pelvis_rotation(cls, observed_joints: np.ndarray, rest_joints: np.ndarray) -> np.ndarray:
        """
        估计根节点(pelvis)的全局旋转，使得绑定姿态下的骨骼向量与观测姿态对齐。
        
        Args:
            observed_joints: (J, 3) 观测到的关节3D坐标
            rest_joints: (J, 3) 绑定姿态下的关节3D坐标
        
        Returns:
            pelvis_rotation: (3,3) 根节点的全局旋转矩阵
        """
        # 选择几个关键骨骼进行对齐, 双髋+脊柱构成骨盆平面
        pelvis_idx, left_hip_idx, right_hip_idx, spine_idx = 0, 1, 2, 3
        
        # 1.1 计算观测姿态下的x轴方向并归一化
        x_axis_obs = observed_joints[right_hip_idx] - observed_joints[left_hip_idx]
        x_axis_obs = x_axis_obs / np.linalg.norm(x_axis_obs)
        # 1.2 计算观测姿态下的y轴方向
        y_axis_obs = observed_joints[spine_idx] - observed_joints[pelvis_idx]
        # 1.3 正交化y轴，确保与x轴垂直
        y_axis_obs = y_axis_obs - np.dot(y_axis_obs, x_axis_obs) * x_axis_obs
        y_axis_obs = y_axis_obs / np.linalg.norm(y_axis_obs)
        # 1.4 计算z轴方向(右手坐标系)
        z_axis_obs = np.cross(x_axis_obs, y_axis_obs)
        # 1.5 构建观测姿态下的骨盆旋转矩阵
        R_obs = np.stack([x_axis_obs, y_axis_obs, z_axis_obs], axis=1)  # (3, 3)
        
        # 2.1 计算绑定姿态下的x轴方向并归一化
        x_axis_rest = rest_joints[right_hip_idx] - rest_joints[left_hip_idx]
        x_axis_rest = x_axis_rest / np.linalg.norm(x_axis_rest)
        # 2.2 计算绑定姿态下的y轴方向
        y_axis_rest = rest_joints[spine_idx] - rest_joints[pelvis_idx]
        # 2.3 正交化y轴，确保与x轴垂直
        y_axis_rest = y_axis_rest - np.dot(y_axis_rest, x_axis_rest) * x_axis_rest
        y_axis_rest = y_axis_rest / np.linalg.norm(y_axis_rest)
        # 2.4 计算z轴方向(右手坐标系)
        z_axis_rest = np.cross(x_axis_rest, y_axis_rest)
        # 2.5 构建绑定姿态下的骨盆旋转矩阵
        R_rest = np.stack([x_axis_rest, y_axis_rest, z_axis_rest], axis=1)  # (3, 3)
        
        # 3. 计算从绑定姿态到观测姿态的旋转矩阵, R * R_rest = R_obs
        R_global = R_obs @ R_rest.T
        return R_global
    
    @classmethod
    def _rotation_between_vectors(cls, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        计算从向量v1旋转到v2方向对齐所需的轴角旋转向量。
        
        Returns:
            axis_angle: (3,) 旋转向量,模长=角度,方向=旋转轴
        """
        # 1. 归一化，提取纯方向
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # 2.1 计算旋转轴
        axis = np.cross(v1, v2) # 旋转轴同时垂直于v1和v2，即 v1 × v2
        axis_norm = np.linalg.norm(axis)
        
        # 2.2 计算“最短弧”旋转角度
        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0) # 夹角余弦值
        theta = np.arccos(cos_theta) # 旋转角度
        
        # 3. 构造轴角向量
        if axis_norm > 1e-8: # 如果叉积不为零向量，有明确的旋转轴
            axis_unit = axis / axis_norm # 归一化得到单位旋转轴
            rotation_vector = axis_unit * theta
        else:
            # 叉积为零，说明两个向量平行（同向或反向）
            if cos_theta > 0: # 同向，不需要旋转
                rotation_vector = np.zeros(3)
            else: # 反向，需要旋转180度。此时有无穷多个旋转轴可选（任何垂直于v1的轴）
                if np.abs(v1[0]) < 0.9:
                    perp_axis = np.array([1.0, 0.0, 0.0])
                else:
                    perp_axis = np.array([0.0, 1.0, 0.0])
                # 使用格拉姆-施密特过程得到垂直轴
                axis_unit = perp_axis - np.dot(perp_axis, v1) * v1
                axis_unit = axis_unit / np.linalg.norm(axis_unit)
                rotation_vector = axis_unit * np.pi
        
        return cls._normalize_axis_angle(rotation_vector)
    
    @classmethod
    def _normalize_axis_angle(cls, axis_angle: np.ndarray) -> np.ndarray:
        """
        规范化轴角到标准范围
        
        SMPL规范:
        - 模长 ∈ [0, π]: 自动满足(arccos输出范围)
        - 如果模长 > π,翻转方向并取补角
        
        Args:
            axis_angle: (3,) 轴角向量
        
        Returns:
            normalized: (3,) 规范化后的轴角,模长 ∈ [0, π]
        """
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-8:
            return np.zeros(3)
        
        # SMPL使用最短路径旋转,angle应该已经 ∈ [0, π]
        # 但为安全起见,处理可能的数值误差
        if angle > np.pi:
            # 翻转到 [-π, 0) 或等效表示
            axis = axis_angle / angle
            angle = angle - 2 * np.pi * np.ceil((angle - np.pi) / (2 * np.pi))
            return axis * angle
        
        return axis_angle


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
        elev: float = 10,
        azim: float = 45,
        roll: float = 0,
        vertical_axis: str = 'y',
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
        ax.view_init(elev=elev, azim=azim, roll=roll, vertical_axis=vertical_axis)
        
        plt.tight_layout()
        plt.show()
    
    @classmethod
    def animate_skeleton_sequence(
        cls, 
        joints_3d_sequence: list | np.ndarray,
        skeleton_connections: list | np.ndarray,
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
        all_joints = np.array(joints_3d_sequence) if isinstance(joints_3d_sequence, list) else joints_3d_sequence  # (N, J, 3)
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
    
    @classmethod
    def normalize_pose_to_ground(cls, joints_3d: np.ndarray, **kwargs) -> np.ndarray:
        """
        将3D骨架姿态归一化到x-y平面直立站立
        
        Args:
            joints_3d: (J, 3) 或 (N, J, 3) 关节3D坐标
        
        Returns:
            normalized_joints: 归一化后的关节坐标
        """
        original_shape = joints_3d.shape
        
        # 确保是(N, J, 3)格式
        if joints_3d.ndim == 2:
            joints_3d = joints_3d[np.newaxis, ...]  # (J, 3) -> (1, J, 3)
            single_frame = True
        else:
            single_frame = False
        
        N, J, _ = joints_3d.shape
        normalized = np.zeros_like(joints_3d)
        
        for i in range(N):
            normalized[i] = cls._normalize_single_frame(joints_3d[i], **kwargs)
        
        # 恢复原始形状
        if single_frame:
            normalized = normalized[0]
        
        return normalized
    
    @classmethod
    def _normalize_single_frame(cls, joints: np.ndarray, pelvis_idx=0, left_ankle_idx=7, right_ankle_idx=8, spine3_idx=9) -> np.ndarray:
        """归一化单帧姿态"""
        joints = joints.copy()
        
        # Step 1: 平移pelvis到原点
        pelvis = joints[pelvis_idx].copy()
        joints = joints - pelvis
        
        # Step 2: 将脚踝放在Z=0平面
        ground_z = min(joints[left_ankle_idx, 2], joints[right_ankle_idx, 2])
        joints[:, 2] -= ground_z
        
        # Step 3: 旋转使脊柱沿Z轴垂直向上
        spine_vector = joints[spine3_idx] - joints[pelvis_idx]
        joints = cls._align_vector_to_z_axis(joints, spine_vector)
        
        # Step 4: 归一化朝向(肩膀沿X轴)
        left_shoulder_idx = 16
        right_shoulder_idx = 17
        
        shoulder_vector = joints[right_shoulder_idx] - joints[left_shoulder_idx]
        angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        
        # 绕Z轴旋转
        R_z = cls._rotation_matrix_z(-angle)
        joints = (R_z @ joints.T).T
        
        return joints

    @classmethod
    def _align_vector_to_z_axis(cls, joints: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """旋转关节使指定向量对齐到Z轴"""
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        z_axis = np.array([0, 0, 1])
        
        rotation_axis = np.cross(vector, z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            return joints
        
        rotation_axis = rotation_axis / rotation_axis_norm
        cos_angle = np.dot(vector, z_axis)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        R = rodrigues_rotation_matrix(rotation_axis, angle)
        joints_rotated = (R @ joints.T).T
        
        return joints_rotated
    
    @classmethod
    def _rotation_matrix_z(cls, angle: float) -> np.ndarray:
        """绕Z轴旋转矩阵"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])


class SMPLVisualizer(SMPL, SMPLVisualizerBase):
    @classmethod
    def visualize_skeleton(
        cls, 
        joints_3d: np.ndarray,
        normalize_pose: bool = False,
        show_joint_labels: bool = True,
        title: str = "SMPL-23 Skeleton",
        **kwargs
    ):
        """可视化单帧3D骨架
        Args:
            joints_3d: (23, 3) 3D关节位置
            show_joint_labels: 是否显示关节点标签
            title: 图表标题
            **kwargs: 其他可选参数传递给基础可视化方法
        """
        if normalize_pose:
            joints_3d = cls.normalize_pose_to_ground(joints_3d)
        
        super().visualize_skeleton(
            joints_3d=joints_3d,
            skeleton_connections=cls.SKELETON_CONNECTIONS,
            joint_names=cls.JOINT_NAMES,
            show_joint_labels=show_joint_labels,
            title=title,
            **kwargs
        )


class SMPL21Joint(SMPL):
    """SMPL 3D Human Model with 21 Joints (without hands)"""
    NUM_JOINTS = 21
    JOINT_NAMES = get_template_tpose_smpl().joint_names[:22]  # 包含根节点pelvis
    PARENT_INDICES = get_template_tpose_smpl().parent_indices[:22]
    TEMPLATE_TPOSE = get_template_tpose_smpl().rest_joints[:22]
    SKELETON_CONNECTIONS = get_edges_from_parents(PARENT_INDICES)


class SMPL21Visualizer(SMPL21Joint, SMPLVisualizerBase):
    """SMPL-21 3D Skeleton Visualizer"""
    @classmethod
    def visualize_skeleton(
        cls, 
        joints_3d: np.ndarray,
        normalize_pose: bool = False,
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
        if normalize_pose:
            joints_3d = cls.normalize_pose_to_ground(joints_3d)
        
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
        normalize_pose: bool = True,
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
        if normalize_pose:
            sequence_array = np.array(joints_3d_sequence)
            joints_3d_sequence = cls.normalize_pose_to_ground(sequence_array)
            
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


class SMPL5Joint(SMPL):
    """SMPL 3D Human Model with 5 Joints"""
    NUM_JOINTS = 5
    JOINT_NAMES = get_template_tpose_smpl().joint_names[:6]  # 包含根节点pelvis
    PARENT_INDICES = get_template_tpose_smpl().parent_indices[:6]
    TEMPLATE_TPOSE = get_template_tpose_smpl().rest_joints[:6]
    SKELETON_CONNECTIONS = get_edges_from_parents(PARENT_INDICES)


class SMPL5Visualizer(SMPL5Joint, SMPLVisualizerBase):
    """SMPL-5 3D Skeleton Visualizer"""
    @classmethod
    def visualize_skeleton(
        cls, 
        joints_3d: np.ndarray,
        normalize_pose: bool = False,
        show_joint_labels: bool = True,
        title: str = "SMPL-5 Skeleton",
        **kwargs
    ):
        """可视化单帧3D骨架
        Args:
            joints_3d: (5, 3) 3D关节位置
            show_joint_labels: 是否显示关节点标签
            title: 图表标题
            **kwargs: 其他可选参数传递给基础可视化方法
        """
        if normalize_pose:
            joints_3d = cls.normalize_pose_to_ground(joints_3d)
        
        super().visualize_skeleton(
            joints_3d=joints_3d,
            skeleton_connections=cls.SKELETON_CONNECTIONS,
            joint_names=cls.JOINT_NAMES,
            show_joint_labels=show_joint_labels,
            title=title,
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
    
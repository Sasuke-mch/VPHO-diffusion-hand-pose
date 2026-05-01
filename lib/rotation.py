import torch
import numpy as np
from manopth.manolayer import ManoLayer

def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(axis_angle)
    if angle < 1e-6:
        return torch.eye(3, device=axis_angle.device)

    axis = axis_angle / angle
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c

    R = torch.zeros(3, 3, device=axis_angle.device)
    R[0, 0] = t * x * x + c
    R[0, 1] = t * x * y - s * z
    R[0, 2] = t * x * z + s * y
    R[1, 0] = t * x * y + s * z
    R[1, 1] = t * y * y + c
    R[1, 2] = t * y * z - s * x
    R[2, 0] = t * x * z - s * y
    R[2, 1] = t * y * z + s * x
    R[2, 2] = t * z * z + c
    return R

def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    return R[:2, :].reshape(-1)

def axis_angle_to_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    if not isinstance(axis_angle, torch.Tensor):
        axis_angle = torch.tensor(axis_angle, dtype=torch.float32)
    R = axis_angle_to_rotation_matrix(axis_angle)
    return rotation_matrix_to_6d(R)

def pose_48_to_96(pose_48: np.ndarray) -> np.ndarray:
    is_batch = len(pose_48.shape) == 2
    if not is_batch:
        pose_48 = pose_48.reshape(1, -1)

    B = pose_48.shape[0]
    pose_48 = pose_48.reshape(B, 16, 3)
    pose_96 = []

    for b in range(B):
        joints_6d = []
        for j in range(16):
            aa = torch.tensor(pose_48[b, j], dtype=torch.float32)
            r6d = axis_angle_to_6d(aa)
            joints_6d.append(r6d.numpy())
        pose_96.append(np.concatenate(joints_6d))

    pose_96 = np.stack(pose_96)
    return pose_96[0] if not is_batch else pose_96


def pca_to_axis_angle(pose_48_pca: np.ndarray, mano_layer) -> np.ndarray:
    if pose_48_pca.ndim == 1:
        pose_48_pca = pose_48_pca.reshape(1, -1)
        return_single = True
    else:
        return_single = False

    B = pose_48_pca.shape[0]
    hands_components = mano_layer.smpl_data["hands_components"]
    hands_mean = mano_layer.smpl_data["hands_mean"]

    pose_48_aa_list = []
    for b in range(B):
        global_orient = pose_48_pca[b, :3]
        hand_pose_pca = pose_48_pca[b, 3:48]
        hand_pose_aa = np.matmul(hand_pose_pca, hands_components) + hands_mean
        pose_48_aa = np.concatenate([global_orient, hand_pose_aa])
        pose_48_aa_list.append(pose_48_aa)

    pose_48_aa = np.stack(pose_48_aa_list).astype(np.float32)
    return pose_48_aa[0] if return_single else pose_48_aa

def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    if angle < 1e-6:
        return torch.zeros(3, device=R.device)

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = torch.stack([rx, ry, rz]) / (2 * torch.sin(angle))
    axis = axis / torch.norm(axis)

    return axis * angle


def r6d_to_axis_angle(r6d: torch.Tensor) -> torch.Tensor:
    """
    6D连续表示 → 轴角，添加数值稳定性处理
    """
    r1 = r6d[:3]
    r2 = r6d[3:6]

    # 防止除零
    norm1 = torch.norm(r1)
    if norm1 < 1e-6:
        r1 = torch.tensor([1.0, 0.0, 0.0], device=r6d.device)
    else:
        r1 = r1 / norm1

    # 正交化
    r3 = torch.cross(r1, r2, dim=-1)
    norm3 = torch.norm(r3)
    if norm3 < 1e-6:
        # r1 和 r2 共线，需要找一个垂直向量
        # 取 r1 的垂直方向
        if abs(r1[0]) < 0.9:
            r3 = torch.cross(r1, torch.tensor([1.0, 0.0, 0.0], device=r6d.device))
        else:
            r3 = torch.cross(r1, torch.tensor([0.0, 1.0, 0.0], device=r6d.device))
        r3 = r3 / torch.norm(r3)
    else:
        r3 = r3 / norm3

    r2 = torch.cross(r3, r1, dim=-1)
    r2 = r2 / torch.norm(r2)

    R = torch.stack([r1, r2, r3], dim=0)  # [3, 3]
    return rotation_matrix_to_axis_angle(R)

def pose_96_to_48(pose_96: torch.Tensor) -> torch.Tensor:
    """
    96维6D表示 → 48维轴角，添加容错
    """
    is_batch = pose_96.dim() == 2
    if not is_batch:
        pose_96 = pose_96.unsqueeze(0)

    B = pose_96.shape[0]
    pose_96 = pose_96.reshape(B, 16, 6)
    pose_48_list = []

    for b in range(B):
        joints_aa = []
        for j in range(16):
            r6d = pose_96[b, j]

            # 检查 6D 向量是否有效
            if torch.isnan(r6d).any() or torch.isinf(r6d).any():
                print(f"警告: 候选 {b} 关节 {j} 的 6D 表示无效，使用单位旋转")
                aa = torch.zeros(3, device=pose_96.device)
            else:
                try:
                    aa = r6d_to_axis_angle(r6d)
                    if torch.isnan(aa).any():
                        aa = torch.zeros(3, device=pose_96.device)
                except Exception as e:
                    print(f"警告: r6d_to_axis_angle 失败: {e}")
                    aa = torch.zeros(3, device=pose_96.device)
            joints_aa.append(aa)
        pose_48_list.append(torch.cat(joints_aa))

    pose_48 = torch.stack(pose_48_list)
    return pose_48[0] if not is_batch else pose_48

def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """ 将6D连续表示转换为3x3旋转矩阵 """
    if r6d.dim() == 1:
        r6d = r6d.unsqueeze(0)
        single = True
    else:
        single = False

    a1 = r6d[:, :3]
    a2 = r6d[:, 3:6]

    b1 = a1 / torch.norm(a1, dim=-1, keepdim=True)
    b2 = a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=1)
    return R[0] if single else R

def pose_96_to_3d_joints(theta_96, mano_layer, device):
    """ 将96维的手部姿态向量通过MANO层转换为3D关节坐标，单位：米 """
    theta_48_pca = pose_96_to_48(theta_96)
    theta_48_aa = pca_to_axis_angle(theta_48_pca.detach().cpu().numpy(), mano_layer)
    theta_48_aa = torch.tensor(theta_48_aa).to(device)
    beta = torch.zeros(theta_48_aa.shape[0], 10).to(device)

    with torch.no_grad():
        _, joints = mano_layer(theta_48_aa, beta)
    return joints / 1000.0
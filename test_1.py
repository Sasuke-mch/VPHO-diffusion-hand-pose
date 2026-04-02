"""
   初始框架，不包括特征精炼模块
"""

import torch
import numpy as np
from manopth.manolayer import ManoLayer
import open3d as o3d
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import pickle
import cv2
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    轴角 → 旋转矩阵 (Rodrigues公式)

    Args:
        axis_angle: [3] 轴角向量，模为旋转角度（弧度），方向为旋转轴
    Returns:
        R: [3, 3] 旋转矩阵
    """

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
    """
    旋转矩阵 → 6D连续表示（取前两列）

    Args:
        R: [3, 3] 旋转矩阵
    Returns:
        r6d: [6] 6D表示
    """
    return R[:2, :].reshape(-1)

def axis_angle_to_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    单个关节轴角 → 6D连续表示
    """
    if not isinstance(axis_angle, torch.Tensor):
        axis_angle = torch.tensor(axis_angle, dtype=torch.float32)
    R = axis_angle_to_rotation_matrix(axis_angle)
    return rotation_matrix_to_6d(R)

def pose_48_to_96(pose_48: np.ndarray) -> np.ndarray:
    """
    48维轴角 (16关节 × 3) → 96维6D表示 (16关节 × 6)

    Args:
        pose_48: [48] 或 [B, 48] 轴角参数
    Returns:
        pose_96: [96] 或 [B, 96] 6D表示
    """
    # 统一转为 torch 处理
    is_batch = len(pose_48.shape) == 2
    if not is_batch:
        pose_48 = pose_48.reshape(1, -1)

    B = pose_48.shape[0]
    pose_48 = pose_48.reshape(B, 16, 3)  # [B, 16, 3]
    pose_96 = []

    for b in range(B):
        joints_6d = []
        for j in range(16):
            aa = torch.tensor(pose_48[b, j], dtype=torch.float32)
            r6d = axis_angle_to_6d(aa)  # [6]
            joints_6d.append(r6d.numpy())
        pose_96.append(np.concatenate(joints_6d))  # [96]

    pose_96 = np.stack(pose_96)  # [B, 96]

    if not is_batch:
        return pose_96[0]
    return pose_96


class TimeEmbedding(nn.Module):
    """将时间步编码为向量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # 使用正弦位置编码（
        half_dim = self.dim // 2
        device = t.device
        emb = torch.log(torch.tensor(10000.0, device= device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def precompute_diffusion_coeffs(T=1000, beta_start=1e-4, beta_end=0.02, device = None):
    beta = torch.linspace(beta_start, beta_end, T, device = device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


class DexYCBDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 T: int = 1000):
        """
        Args:
            data_root: DexYCB 数据集根目录
            split: 'train' 或 'test'（用于划分）
            transform: 图像预处理函数
            T: 扩散总步数
        """
        self.data_root = data_root
        self.transform = transform
        self.T = T

        # 预计算扩散系数（全局，所有样本共享）
        self.beta, self.alpha, self.alpha_bar = precompute_diffusion_coeffs(T)

        # 收集所有样本路径
        self.samples = self._load_samples(split)

        print(f"Loaded {len(self.samples)} samples from DexYCB ({split} split)")

    def _load_samples(self, split: str) -> List[Tuple[str, str, np.ndarray]]:
        """
        遍历数据集目录，收集所有 (image_path, label_path, betas) 对
        """
        samples = []

        if not os.path.exists(self.data_root):
            print(f"ERROR: Data root does not exist: {self.data_root}")
            return samples

        # 获取所有 subject
        all_subjects = [d for d in os.listdir(self.data_root)
                        if os.path.isdir(os.path.join(self.data_root, d))]

        # 根据 split 选择过滤 subject-01 和 subject-02
        if split == 'train':
            subjects = [d for d in all_subjects if 'subject-01' in d or 'subject-02' in d]
        elif split == 'test':
            subjects = [d for d in all_subjects if 'subject-03' in d ]
        else:
            subjects = all_subjects

        print(f"=" * 60)
        print(f"开始扫描数据集")
        print(f"数据根目录: {self.data_root}")
        print(f"找到 {len(subjects)} 个 subject 目录")
        print(f"=" * 60)

        total_subjects = len(subjects)
        total_samples_found = 0
        total_sequences = 0
        total_cameras = 0

        for subject_idx, subject_dir in enumerate(subjects):
            print(f"\n[{subject_idx + 1}/{total_subjects}] 正在处理 subject: {subject_dir}")
            subject_path = os.path.join(self.data_root, subject_dir)

            # 获取序列目录
            sequences = [d for d in os.listdir(subject_path)
                         if os.path.isdir(os.path.join(subject_path, d))]
            print(f"  ├─ 找到 {len(sequences)} 个序列")
            total_sequences += len(sequences)

            subject_samples = 0

            for seq_idx, seq_dir in enumerate(sequences):
                # 每处理5个序列打印一次进度
                if seq_idx % 5 == 0:
                    print(f"  │  └─ 处理序列进度: {seq_idx + 1}/{len(sequences)}")

                seq_path = os.path.join(subject_path, seq_dir)

                # 加载该序列的形状参数
                betas = self._load_sequence_betas(seq_path)

                # 获取相机目录
                cam_dirs = [d for d in os.listdir(seq_path)
                            if os.path.isdir(os.path.join(seq_path, d))]

                for cam_dir in cam_dirs:
                    cam_path = os.path.join(seq_path, cam_dir)
                    total_cameras += 1

                    # 查找所有 .npz 标注文件
                    npz_files = [f for f in os.listdir(cam_path)
                                 if f.endswith('.npz') and 'labels_' in f]

                    for file in npz_files:
                        frame_id = file.replace('labels_', '').replace('.npz', '')
                        label_path = os.path.join(cam_path, file)

                        # 查找对应的 RGB 图像
                        rgb_path = None
                        for ext in ['.jpg', '.png']:
                            candidate = os.path.join(cam_path, f'color_{frame_id}{ext}')
                            if os.path.exists(candidate):
                                rgb_path = candidate
                                break

                        if rgb_path:
                            samples.append((rgb_path, label_path, betas))
                            total_samples_found += 1
                            subject_samples += 1

                            # 每找到 500 个样本打印一次统计
                            if total_samples_found % 500 == 0:
                                print(f"  │     ✓ 已找到 {total_samples_found} 个样本...")

            print(f"  └─ {subject_dir} 完成，本 subject 找到 {subject_samples} 个样本")
            print(f"     累计总样本数: {total_samples_found}")

        print(f"\n" + "=" * 60)
        print(f"扫描完成！统计信息:")
        print(f"  - Subject 数量: {total_subjects}")
        print(f"  - 序列数量: {total_sequences}")
        print(f"  - 相机数量: {total_cameras}")
        print(f"  - 有效样本数: {len(samples)}")
        print(f"=" * 60)

        return samples

    def _load_sequence_betas(self, seq_path: str) -> np.ndarray:
        """
        加载序列级别的形状参数
        """
        # 查找 info 文件
        betas = np.zeros(10, dtype=np.float32)

        for file in os.listdir(seq_path):
            if file.startswith('info') and (file.endswith('.pkl') or file.endswith('.npz')):
                info_path = os.path.join(seq_path, file)
                try:
                    if info_path.endswith('.pkl'):
                        with open(info_path, 'rb') as f:
                            info = pickle.load(f, encoding='latin1')
                            if 'betas' in info:
                                betas = np.array(info['betas'][:10], dtype=np.float32)
                            elif 'mano_shape' in info:
                                betas = np.array(info['mano_shape'][:10], dtype=np.float32)
                    elif info_path.endswith('.npz'):
                        info = np.load(info_path, allow_pickle=True)
                        if 'betas' in info:
                            betas = info['betas'][:10].astype(np.float32)
                        elif 'mano_shape' in info:
                            betas = info['mano_shape'][:10].astype(np.float32)
                except Exception as e:
                    print(f"Warning: Failed to load betas from {info_path}: {e}")
                break

        return betas

    def __getitem__(self, idx: int):
        image_path, label_path, betas = self.samples[idx]

        # 加载图像
        image = self._load_image(image_path)

        # 加载 MANO 参数
        theta_0 = self._load_pose_from_label(label_path)
        theta_0 = torch.tensor(theta_0, dtype=torch.float32)
        betas = torch.tensor(betas, dtype=torch.float32)

        # 图像预处理
        if self.transform:
            image = self.transform(image)
        else:
            default_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            image = default_transform(image)

        # 随机采样时间步
        t = torch.randint(1, self.T + 1, (1,)).item()
        eps = torch.randn(96)

        # 加噪
        alpha_bar_t = self.alpha_bar[t - 1]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
        theta_t = sqrt_alpha_bar_t * theta_0 + sqrt_one_minus * eps

        return theta_t, torch.tensor(t, dtype=torch.long), image, theta_0, eps

    def _load_pose_from_label(self, label_path: str) -> np.ndarray:
        """
        从标注文件加载姿态参数
        """
        data = np.load(label_path, allow_pickle=True)
        pose_m = data['pose_m']

        # 提取前 48 维
        if pose_m.shape == (1, 51):
            pose_48 = pose_m[0][:48]
        elif pose_m.shape == (51,):
            pose_48 = pose_m[:48]
        else:
            pose_48 = pose_m.flatten()[:48]

        # 转换为 96 维 6D 表示
        pose_96 = pose_48_to_96(pose_48)

        return pose_96.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mano_params(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 DexYCB 的 .npz 文件中加载 MANO 参数

        DexYCB 的 pose_m 是 (1, 51) 维：
        - 前 48 维：16个关节的轴角表示
        - 后 3 维：全局旋转（或平移）
        """
        # 加载 .npz 文件
        data = np.load(label_path, allow_pickle=True)

        # 获取 pose_m
        if 'pose_m' in data:
            pose_m = data['pose_m']
        else:
            raise KeyError(f"Cannot find pose_m. Available keys: {list(data.keys())}")

        # pose_m 的形状是 (1, 51)，提取前 48 维
        if pose_m.shape == (1, 51):
            pose_51 = pose_m[0]
            pose_48 = pose_51[:48]
        elif pose_m.shape == (51,):
            pose_48 = pose_m[:48]
        elif pose_m.shape == (48,):
            pose_48 = pose_m
        else:
            print(f"Warning: Unexpected pose_m shape: {pose_m.shape}")
            # 尝试展平并取前 48 维
            pose_flat = pose_m.flatten()
            if len(pose_flat) >= 48:
                pose_48 = pose_flat[:48]
            else:
                pose_48 = np.pad(pose_flat, (0, 48 - len(pose_flat)), 'constant')

        # 转换为 96 维 6D 表示
        pose_96 = pose_48_to_96(pose_48)

        betas = np.zeros(10, dtype=np.float32)

        if 'betas' in data:
            betas_data = data['betas']
            if isinstance(betas_data, np.ndarray):
                betas = betas_data[:10].astype(np.float32)
        elif 'mano_shape' in data:
            shape_data = data['mano_shape']
            if isinstance(shape_data, np.ndarray):
                betas = shape_data[:10].astype(np.float32)

        return pose_96.astype(np.float32), betas

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        加载并预处理图像
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



class ConditionalDiffusionModel(nn.Module):
    """条件扩散模型：预测噪声"""
    def __init__(self, pose_dim=96, cond_dim=2048, time_dim=256, hidden_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 条件 ψ 的嵌入
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 主网络：输入姿态 + 时间嵌入 + 条件嵌入
        self.net = nn.Sequential(
            nn.Linear(pose_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, theta_t, t, psi):
        # theta_t: [B, pose_dim]
        # t: [B] (标量)
        # psi: [B, cond_dim]
        t_emb = self.time_mlp(t)          # [B, hidden_dim]
        cond_emb = self.cond_mlp(psi)     # [B, hidden_dim]
        inp = torch.cat([theta_t, t_emb, cond_emb], dim=-1)  # [B, pose_dim+2*hidden_dim]
        eps_pred = self.net(inp)           # [B, pose_dim]
        return eps_pred


def test_model_with_real_image(model, feature_extractor, image_path, device,
                               pose_dim, cond_dim, alpha, alpha_bar, T, num_candidates=10):

    model.eval()
    image_bgr = cv2.imread(image_path)
    # 预处理前将其转换为 RGB，供模型和 torchvision Transform 使用
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)  # [1, 3, 256, 256]

    # 提取特征并生成
    with torch.no_grad():
        psi = feature_extractor(image_tensor)  # [1, 2048]

    candidates = generate_candidates(
        model=model, psi=psi, num_candidates=num_candidates, T=T,
        device=device, alpha=alpha, alpha_bar=alpha_bar, pose_dim=pose_dim
    )

    print(f"基于图像 {image_path} 生成了 {candidates.shape[0]} 个候选姿态")
    visualize_hand(candidates[0:1], device, "Generated Hand from Real Image")
    cv2.destroyAllWindows()

    return candidates


import random

def test_model_on_subject_03(model, feature_extractor, data_root, device,
                             pose_dim, cond_dim, alpha, alpha_bar, T, num_samples=3):
    """
    在 subject-03 数据上随机抽取图像并测试模型
    """
    if not os.path.exists(data_root):
        print(f"数据根目录不存在: {data_root}")
        return

    # 寻找 subject-03 的目录
    all_subjects = os.listdir(data_root)
    sub3_dirs = [d for d in all_subjects if 'subject-03' in d and os.path.isdir(os.path.join(data_root, d))]

    if not sub3_dirs:
        print("未找到 subject-03 的数据目录！")
        return

    # 收集 subject-03 下的所有图像路径
    image_paths = []
    for sub_dir in sub3_dirs:
        sub_path = os.path.join(data_root, sub_dir)
        for root_dir, _, files in os.walk(sub_path):
            for f in files:
                if f.startswith('color_') and (f.endswith('.jpg') or f.endswith('.png')):
                    image_paths.append(os.path.join(root_dir, f))

    if not image_paths:
        print("在 subject-03 的目录中未找到任何 RGB 图像！")
        return

    # 随机选择若干张图像
    selected_images = random.sample(image_paths, min(num_samples, len(image_paths)))
    print(f"已在 subject-03 中找到 {len(image_paths)} 张图像，将随机抽测 {len(selected_images)} 张。")

    for i, img_path in enumerate(selected_images):
        print(f"\n[{i + 1}/{len(selected_images)}] 正在测试图像: {img_path}")
        test_model_with_real_image(
            model=model,
            feature_extractor=feature_extractor,
            image_path=img_path,
            device=device,
            pose_dim=pose_dim,
            cond_dim=cond_dim,
            alpha=alpha,
            alpha_bar=alpha_bar,
            T=T,
            num_candidates=10
        )


def reverse_diffusion(model, psi, T, device, alpha, alpha_bar, pose_dim):
    """
    反向扩散过程：从噪声生成姿态

    Args:
        model: 扩散模型
        psi: 条件特征 [1, cond_dim]
        T: 总步数
        alpha, alpha_bar: 扩散系数
        pose_dim: 姿态维度

    Returns:
        theta_0: [1, pose_dim] 生成的姿态
    """
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)

    B = psi.shape[0]

    # 从纯噪声开始
    theta_t = torch.randn(B, pose_dim).to(device)

    # 逐步去噪
    for t in reversed(range(1, T + 1)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # 模型预测噪声
        eps_pred = model(theta_t, t_tensor, psi)

        # 获取系数
        alpha_t = alpha[t - 1]
        alpha_bar_t = alpha_bar[t - 1]
        beta_t = 1 - alpha_t

        # 去噪公式
        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

        theta_det = coeff1 * (theta_t - coeff2 * eps_pred)

        # 添加随机噪声（t > 1 时）
        if t > 1:
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn_like(theta_t)
            theta_t = theta_det + sigma_t * z
        else:
            theta_t = theta_det

    return theta_t

def plot_loss_curve(losses: list, save_path: str = "train_loss_curve_test_1.png"):
    """
    绘制并保存训练 Loss 曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, linestyle='-', color='b', label='Train Loss')
    plt.title("Training Loss Over Batches")
    plt.xlabel("Batch Iterations")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss 曲线已保存到: {save_path}")

def visualize_hand(theta_96, device, title="Hand"):
    """
    可视化手部网格
    theta_96: 可以是 [B, 96] 或张量列表
    """
    # 强制将输入统一规整为二维 [B, 96]
    theta_96 = theta_96.view(-1, 96)

    # 转换为 48 维轴角，此时得到的也是 [B, 48]
    theta_48 = pose_96_to_48(theta_96)

    # 形状参数（平均手型）
    beta = torch.zeros(theta_48.shape[0], 10).to(device)

    # 生成手部网格
    with torch.no_grad():
        vertices, joints = mano_layer(theta_48, beta)

    # 可视化 (取出第0个样本)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[0].cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(mano_layer.th_faces.cpu().numpy())
    mesh.compute_vertex_normals()

    mesh.paint_uniform_color([0.8, 0.7, 0.6])
    o3d.visualization.draw_geometries([mesh], window_name=title)


def generate_candidates(model, psi, num_candidates, T, device, alpha, alpha_bar, pose_dim):
    candidates = []
    for _ in range(num_candidates):
        theta = reverse_diffusion(model, psi, T, device, alpha, alpha_bar, pose_dim)
        candidates.append(theta)
    return torch.stack(candidates)


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    旋转矩阵 → 轴角
    R: [3, 3] 旋转矩阵
    returns: [3] 轴角向量
    """
    # 计算旋转角
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    if angle < 1e-6:
        return torch.zeros(3, device=R.device)

    # 计算旋转轴
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = torch.stack([rx, ry, rz]) / (2 * torch.sin(angle))
    axis = axis / torch.norm(axis)

    return axis * angle


def r6d_to_axis_angle(r6d: torch.Tensor) -> torch.Tensor:
    """
    6D连续表示 → 轴角
    r6d: [6] 6D表示 = [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
    returns: [3] 轴角向量
    """
    # 重建旋转矩阵的前两列
    r1 = r6d[:3]
    r2 = r6d[3:6]

    # 正交化 (添加 dim=-1 消除警告)
    r1 = r1 / torch.norm(r1)
    r3 = torch.cross(r1, r2, dim=-1)
    r3 = r3 / torch.norm(r3)
    r2 = torch.cross(r3, r1, dim=-1)

    # 重建旋转矩阵
    R = torch.stack([r1, r2, r3], dim=1)  # [3, 3]

    # 转换为轴角
    return rotation_matrix_to_axis_angle(R)



def pose_96_to_48(pose_96: torch.Tensor) -> torch.Tensor:
    """
    96维6D表示 (16关节 × 6) → 48维轴角 (16关节 × 3)

    Args:
        pose_96: [B, 96] 或 [96] 6D表示
    Returns:
        pose_48: [B, 48] 或 [48] 轴角
    """

    is_batch = pose_96.dim() == 2
    if not is_batch:
        pose_96 = pose_96.unsqueeze(0)  # [1, 96]

    B = pose_96.shape[0]
    pose_96 = pose_96.reshape(B, 16, 6)  # [B, 16, 6]
    pose_48_list = []

    for b in range(B):
        joints_aa = []
        for j in range(16):
            r6d = pose_96[b, j]  # [6]
            aa = r6d_to_axis_angle(r6d)  # [3]
            joints_aa.append(aa)
        pose_48_list.append(torch.cat(joints_aa))  # [48]

    pose_48 = torch.stack(pose_48_list)  # [B, 48]

    if not is_batch:
        return pose_48[0]
    return pose_48


# 初始化 MANO 层
mano_layer = ManoLayer(
    mano_root='./mano',  # 放 MANO 模型文件的路径
    use_pca=False,
    ncomps=6,
    flat_hand_mean=True,
    side='right'
).to(device)


# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    #归一化
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置模型超参数
    batch_size = 12
    T = 1000
    pose_dim = 96
    cond_dim = 2048
    lr = 1e-4
    epochs = 3

    # 生成扩散模型系数
    beta, alpha, alpha_bar = precompute_diffusion_coeffs(T, device=device)

    # 指定数据集路径
    data_root = "/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb"

    # 数据加载器
    dataset_train = DexYCBDataset(data_root,  split = "train", transform=transform, T=T)
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size, shuffle=True, num_workers=0)

    # 特征提取器
    feature_extractor = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(
        *list(feature_extractor.children())[:-2],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    ).to(device)
    feature_extractor.eval()  # 冻结特征提取器权重

    # 实例化条件扩散模型
    model = ConditionalDiffusionModel(
        pose_dim=pose_dim,
        cond_dim=cond_dim,
        time_dim=256,
        hidden_dim=512
    ).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("开始训练")
    batch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (theta_t, t, image, theta_0, eps_true) in enumerate(dataloader_train):
            # 将数据传入设备
            theta_t = theta_t.to(device)
            t = t.to(device)
            image = image.to(device)
            eps_true = eps_true.to(device)

            psi = feature_extractor(image)
            eps_pred = model(theta_t, t, psi)

            loss = nn.MSELoss()(eps_pred, eps_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                batch_losses.append(loss.item())

        avg_loss = total_loss / len(dataloader_train)
        print(f"Epoch {epoch + 1}/{epochs} 完成, 平均损失: {avg_loss:.4f}")

    print("训练完成！")
    plot_loss_curve(batch_losses, "train_loss_curve_batch_test_1.png")

    image_bgr = cv2.imread("train_loss_curve_batch_test_1.png")

    # 显示真实图像
    cv2.imshow("loss curve of test 1", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    torch.save(model.state_dict(), "diffusion_model_test_1.pth")
    print("模型已保存到 diffusion_model_test_1.pth")

    # 加载训练好的权重
    model_path = "diffusion_model_test_1.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("训练完成！开始在 subject-03 测试集上计算 Loss...")
    dataset_test = DexYCBDataset(data_root, split="test", transform=transform, T=T)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    # 切换为验证模式
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (theta_t, t, image, theta_0, eps_true) in enumerate(dataloader_test):
            # 将数据传入设备
            theta_t = theta_t.to(device)
            t = t.to(device)
            image = image.to(device)
            eps_true = eps_true.to(device)

            # 提取特征并推断
            psi = feature_extractor(image)
            eps_pred = model(theta_t, t, psi)

            # 计算 Loss
            loss = nn.MSELoss()(eps_pred, eps_true)
            test_loss += loss.item()

            # 打印测试进度
            if batch_idx % 50 == 0:
                print(f"Subject-03 测试进度: Batch [{batch_idx}/{len(dataloader_test)}], 当前 Loss: {loss.item():.4f}")

    avg_test_loss = test_loss / len(dataloader_test)
    print(f"Subject-03 测试集平均损失 (Loss): {avg_test_loss:.4f}")

    test_model_on_subject_03(
        model=model,
        feature_extractor=feature_extractor,
        data_root=data_root,
        device=device,
        pose_dim=pose_dim,
        cond_dim=cond_dim,
        alpha=alpha,
        alpha_bar=alpha_bar,
        T=T,
        num_samples=3
    )
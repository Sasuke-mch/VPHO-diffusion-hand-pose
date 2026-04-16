"""
   使用vpho中的条件扩散模型，并且包括了特征精炼模块
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


class BottleneckBlock(nn.Module):
    """标准瓶颈残差块 (He et al., 2016)"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # 1x1 降维
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 卷积（可带 stride）
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # 跳跃连接（若尺寸或通道数变化则适配）
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class FeatureRefinement(nn.Module):
    """特征精炼模块：两个残差块 + 全局平均池化"""
    def __init__(self, in_channels=2048):
        super().__init__()
        # 两个残差块，输入输出通道均为 2048（因为 expansion=4，planes=512 时输出 2048）
        self.block1 = BottleneckBlock(in_channels, planes=512)
        self.block2 = BottleneckBlock(2048, planes=512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x shape: [B, 2048, 8, 8]
        x = self.block1(x)   # [B, 2048, 8, 8]
        x = self.block2(x)   # [B, 2048, 8, 8]
        x = self.gap(x)      # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 2048]
        return x


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
        samples = []
        if not os.path.exists(self.data_root):
            return samples

        all_subjects = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]

        if split == 'train':
            subjects = [d for d in all_subjects if any(f'subject-{i:02d}' in d for i in range(1, 9))]
        elif split == 'test':
            subjects = [d for d in all_subjects if 'subject-09' in d or 'subject-10' in d]
        else:
            subjects = all_subjects

        for subject_dir in subjects:
            subject_path = os.path.join(self.data_root, subject_dir)
            sequences = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

            for seq_dir in sequences:
                seq_path = os.path.join(subject_path, seq_dir)
                betas = self._load_sequence_betas(seq_path)
                cam_dirs = [d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))]

                for cam_dir in cam_dirs:
                    cam_path = os.path.join(seq_path, cam_dir)
                    npz_files = [f for f in os.listdir(cam_path) if f.endswith('.npz') and 'labels_' in f]

                    for file in npz_files:
                        frame_id = file.replace('labels_', '').replace('.npz', '')
                        label_path = os.path.join(cam_path, file)

                        rgb_path = None
                        for ext in ['.jpg', '.png']:
                            candidate = os.path.join(cam_path, f'color_{frame_id}{ext}')
                            if os.path.exists(candidate):
                                rgb_path = candidate
                                break

                        if rgb_path:
                            samples.append((rgb_path, label_path, betas))

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

        return theta_t, torch.tensor(t, dtype=torch.long), image, theta_0, eps, betas

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
            pose_51 = pose_m[0]  # 变为 (51,)
            pose_48 = pose_51[:48]  # 取前 48 维
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

        # 注意：形状参数可能需要在其他地方获取
        betas = np.zeros(10, dtype=np.float32)

        # 如果有 betas 字段则使用
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
    """
        VPHO中的条件扩散模型
        输入：theta_t (姿态), t (时间步), psi (图像特征), beta (形状参数)
        输出：预测的噪声
    """

    def __init__(self,
                 pose_dim=96,
                 cond_dim=2048,
                 beta_dim=10,
                 time_dim=256,
                 hidden_dim=512):
        super().__init__()

        # 1. 时间步嵌入
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. 条件特征 ψ 的嵌入
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. 形状参数 β 的嵌入
        self.beta_mlp = nn.Sequential(
            nn.Linear(beta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4. 主网络
        self.net = nn.Sequential(
            # 输入：pose_dim + hidden_dim (time) + hidden_dim (cond) + hidden_dim (beta)
            nn.Linear(pose_dim + hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim),
        )


    def forward(self, theta_t, t, psi, beta):
        # 时间嵌入
        t_emb = self.time_mlp(t)  # [B, hidden_dim]

        # 条件特征嵌入
        cond_emb = self.cond_mlp(psi)  # [B, hidden_dim]

        # 形状参数嵌入
        beta_emb = self.beta_mlp(beta)  # [B, hidden_dim]

        # 拼接所有输入
        inp = torch.cat([theta_t, t_emb, cond_emb, beta_emb], dim=-1)  # [B, pose_dim + 3*hidden_dim]

        # 预测噪声
        eps_pred = self.net(inp)  # [B, pose_dim]

        return eps_pred


def test_model_with_real_image(model, feature_extractor, image_path, device,
                               pose_dim, cond_dim, alpha, alpha_bar, T, num_candidates=10):

    model.eval()
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    cv2.imshow("Original Image", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        psi = feature_extractor(image_tensor)
        beta = torch.zeros(1, 10).to(device)

    candidates = generate_candidates(
        model=model, psi=psi, beta=beta,num_candidates=num_candidates, T=T,
        device=device, alpha=alpha, alpha_bar=alpha_bar, pose_dim=pose_dim
    )

    print(f"Generated {candidates.shape[0]} candidates for {image_path}")
    visualize_hand(candidates[0:1], device, "Generated Hand from Real Image")
    cv2.destroyAllWindows()

    return candidates


import random

def test_model_on_subject_09(model, feature_extractor, data_root, device,
                             pose_dim, cond_dim, alpha, alpha_bar, T, num_samples=3):
    """
    在 subject-09 数据上随机抽取图像并测试模型
    """
    if not os.path.exists(data_root):
        return

    all_subjects = os.listdir(data_root)
    sub9_dirs = [d for d in all_subjects if 'subject-09' in d and os.path.isdir(os.path.join(data_root, d))]

    if not sub9_dirs:
        return

    # 收集 subject-09 下的所有图像路径
    image_paths = []
    for sub_dir in sub9_dirs:
        sub_path = os.path.join(data_root, sub_dir)
        for root_dir, _, files in os.walk(sub_path):
            for f in files:
                if f.startswith('color_') and (f.endswith('.jpg') or f.endswith('.png')):
                    image_paths.append(os.path.join(root_dir, f))

    if not image_paths:
        return

    # 随机选择若干张图像
    selected_images = random.sample(image_paths, min(num_samples, len(image_paths)))
    for i, img_path in enumerate(selected_images):
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


import matplotlib.pyplot as plt

def plot_loss_curve(losses: list, save_path: str = "train_loss_curve.png"):
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



def reverse_diffusion(model, psi, beta, T, device, alpha, alpha_bar, pose_dim):
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
        eps_pred = model(theta_t, t_tensor, psi, beta)

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


def pca_to_axis_angle(pose_48_pca: np.ndarray) -> np.ndarray:
    """
    将 PCA 格式转换为轴角格式

    Args:
        pose_48_pca: [48] 或 [B, 48] PCA格式
    Returns:
        pose_48_aa: [48] 或 [B, 48] 轴角格式
    """
    # 检查输入维度
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

    if return_single:
        return pose_48_aa[0]
    return pose_48_aa


def visualize_hand(theta_96_pca, device, title="Hand"):
    """
    可视化手部网格
    theta_96_pca: PCA格式的96维6D表示，可以是 [B, 96] 或 [96]
    """
    # 确保有 batch 维度
    if theta_96_pca.dim() == 1:
        theta_96_pca = theta_96_pca.unsqueeze(0)

    # 转为 48 维 PCA 格式
    theta_48_pca = pose_96_to_48(theta_96_pca)  # [B, 48] PCA格式

    # PCA格式 → 轴角格式
    theta_48_aa = pca_to_axis_angle(theta_48_pca.detach().cpu().numpy())
    theta_48_aa = torch.tensor(theta_48_aa).to(device)

    # 确保有 batch 维度
    if theta_48_aa.dim() == 1:
        theta_48_aa = theta_48_aa.unsqueeze(0)

    # 形状参数
    beta = torch.zeros(theta_48_aa.shape[0], 10).to(device)

    # 生成手部网格
    with torch.no_grad():
        vertices, joints = mano_layer(theta_48_aa, beta)

    # 可视化（取第一个样本）
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[0].cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(mano_layer.th_faces.cpu().numpy())
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.7, 0.6])
    o3d.visualization.draw_geometries([mesh], window_name=title)


def generate_candidates(model, psi, beta, num_candidates, T, device, alpha, alpha_bar, pose_dim):
    candidates = []
    for i in range(num_candidates):
        if i % 100 == 0:
            print(f"  生成候选 {i + 1}/{num_candidates}...")  # 添加进度
        theta = reverse_diffusion(model, psi, beta, T, device, alpha, alpha_bar, pose_dim)
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
    mano_root='./mano',
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

    batch_size = 12
    T = 1000
    pose_dim = 96
    cond_dim = 2048
    lr = 1e-4
    epochs = 3

    # 生成扩散模型的系数
    beta, alpha, alpha_bar = precompute_diffusion_coeffs(T, device=device)

    # 指定数据集路径
    data_root = "/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb"

    # 数据加载器
    dataset_train = DexYCBDataset(data_root,  split = "train", transform=transform, T=T)
    dataloader_train = DataLoader(dataset_train,batch_size=batch_size, shuffle=True, num_workers=0)

    # 特征提取器（带特征精炼）
    resnet_backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
    feature_extractor = nn.Sequential(
        resnet_backbone,
        FeatureRefinement(in_channels=2048)
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

        for batch_idx, (theta_t, t, image, theta_0, eps_true, betas) in enumerate(dataloader_train):
            # 将数据传入设备
            theta_t = theta_t.to(device)
            t = t.to(device)
            image = image.to(device)
            eps_true = eps_true.to(device)
            betas = betas.to(device)

            with torch.no_grad():
                psi = feature_extractor(image)

            eps_pred = model(theta_t, t, psi, betas)

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
    plot_loss_curve(batch_losses, "train_loss_curve_batch_test_3_s0.png")

    image_bgr = cv2.imread("train_loss_curve_batch_test_3_s0.png")

    # 显示真实图像
    cv2.imshow("loss curve of test_3_s0", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    torch.save(model.state_dict(), "diffusion_model_test_3_s0.pth")
    print("模型已保存到 diffusion_model_test_3_s0.pth")

    # 加载训练好的权重
    model_path = "diffusion_model_hand_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("训练完成！开始在 subject-09 测试集上计算 Loss...")
    dataset_test = DexYCBDataset(data_root, split="test", transform=transform, T=T)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    # 切换为验证模式
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (theta_t, t, image, theta_0, eps_true, betas) in enumerate(dataloader_test):
            # 将数据传入设备
            theta_t = theta_t.to(device)
            t = t.to(device)
            image = image.to(device)
            eps_true = eps_true.to(device)
            betas = betas.to(device)

            # 提取特征并推断
            psi = feature_extractor(image)
            eps_pred = model(theta_t, t, psi, betas)

            # 计算 Loss
            loss = nn.MSELoss()(eps_pred, eps_true)
            test_loss += loss.item()

            # 打印测试进度
            if batch_idx % 50 == 0:
                print(f"Subject-09 测试进度: Batch [{batch_idx}/{len(dataloader_test)}], 当前 Loss: {loss.item():.4f}")

    avg_test_loss = test_loss / len(dataloader_test)
    print(f"Subject-09 测试集平均损失 (Loss): {avg_test_loss:.4f}")

    test_model_on_subject_09(
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
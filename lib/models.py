import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 把根目录加进去
from backbone_FPN_HFL import FPN

class HandleDiffusionModel(nn.Module):
    """ 基于MLP的手部条件扩散预测模型 (Hand Diffusion Engine) """
    def __init__(self, pose_dim=96, cond_dim=1024, beta_dim=10, time_dim=256, hidden_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(beta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(pose_dim + hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, theta_t, t, psi, beta):
        t_emb = self.time_mlp(t)
        cond_emb = self.cond_mlp(psi)
        beta_emb = self.beta_mlp(beta)
        inp = torch.cat([theta_t, t_emb, cond_emb, beta_emb], dim=-1)
        return self.net(inp)

class ObjectDiffusionModel(nn.Module):
    """ 基于MLP的物体条件扩散预测模型 (Object Diffusion Engine) """
    def __init__(self, cond_dim=1024, pose_dim=9, time_dim=256, hidden_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(pose_dim + hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, phi_t, t, psi_obj):
        t_emb = self.time_mlp(t)
        cond_emb = self.cond_mlp(psi_obj)
        inp = torch.cat([phi_t, t_emb, cond_emb], dim=-1)
        return self.net(inp)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        device = t.device
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class FeatureEncoder(nn.Module):
    def __init__(self, input_channels=256, num_heatmaps=21, output_dim=1024, roi_size=32):
        super().__init__()
        total_channels = input_channels + num_heatmaps
        self.encoder = nn.Sequential(
            nn.Conv2d(total_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),          # 32→16
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2),          # 16→8
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → 1×1
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.roi_size = roi_size

    def forward(self, feat_roi, heatmap):
        # feat_roi: [B, 256, 32, 32]
        # heatmap: [B, J, 64, 64]
        heatmap_resized = F.interpolate(heatmap, size=(self.roi_size, self.roi_size),
                                        mode='bilinear', align_corners=False)
        combined = torch.cat([feat_roi, heatmap_resized], dim=1)
        x = self.encoder(combined).flatten(1)
        return self.fc(x)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = FPN(pretrained=True)  # 直接用官方的

    def forward(self, image):
        """
        Args:
            image: [B, 3, 256, 256]
        Returns:
            hand_feat: [B, 256, 64, 64]  手部特征图
            obj_feat:  [B, 256, 64, 64]  物体特征图
        """
        hand_feat, obj_feat = self.fpn(image)
        return hand_feat, obj_feat


class HeatmapPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # 手部：32×32 → 64×64
        self.hand_deconv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32→64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 21, kernel_size=1),
            nn.Sigmoid()
        )

        # 物体：32×32 → 64×64
        self.object_deconv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32→64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 27, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, hand_roi, obj_roi):
        # 输入：[B, 256, 32, 32]（ROI Align 裁剪后的特征）
        # 输出：[B, 21, 64, 64] 和 [B, 27, 64, 64]
        hand_heatmap = self.hand_deconv(hand_roi)
        object_heatmap = self.object_deconv(obj_roi)
        return hand_heatmap, object_heatmap

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

# ============================================================================
# VPHO-style regression heads and handroot object diffusion additions
# ============================================================================

from lib.rotation import (
    pose_96_to_48,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from lib.config import mano_layer_left, mano_layer_right


def pose96_to_axis_angle_torch(theta_96: torch.Tensor) -> torch.Tensor:
    """theta_96 [B,96] -> axis-angle [B,48]."""
    return pose_96_to_48(theta_96)


def axis_angle_to_pose96_torch(pose_aa: torch.Tensor) -> torch.Tensor:
    """axis-angle [B,48] -> theta_96 [B,96]."""
    if pose_aa.dim() == 1:
        pose_aa = pose_aa.unsqueeze(0)
    B = pose_aa.shape[0]
    R = axis_angle_to_matrix(pose_aa.reshape(B, 16, 3))
    return matrix_to_rotation_6d(R).reshape(B, 96)


def mano_forward_mixed_side(
    pose_aa: torch.Tensor,
    betas: torch.Tensor,
    is_right: torch.Tensor,
    root_relative: bool = True,
):
    """
    Run left/right MANO according to is_right.

    Args:
        pose_aa: [B,48]
        betas: [B,10]
        is_right: [B] bool
        root_relative: subtract wrist joint if True

    Returns:
        verts: [B,778,3] in meters
        joints: [B,21,3] in meters
    """
    if pose_aa.dim() == 1:
        pose_aa = pose_aa.unsqueeze(0)
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)

    device = pose_aa.device
    dtype = pose_aa.dtype
    B = pose_aa.shape[0]
    is_right = is_right.to(device=device).bool().reshape(-1)

    verts = torch.zeros(B, 778, 3, device=device, dtype=dtype)
    joints = torch.zeros(B, 21, 3, device=device, dtype=dtype)

    right_mask = is_right
    left_mask = ~is_right

    if right_mask.any():
        vr, jr = mano_layer_right(
            pose_aa[right_mask].to(device),
            betas[right_mask].to(device),
        )
        verts[right_mask] = vr.to(device=device, dtype=dtype) / 1000.0
        joints[right_mask] = jr.to(device=device, dtype=dtype) / 1000.0

    if left_mask.any():
        vl, jl = mano_layer_left(
            pose_aa[left_mask].to(device),
            betas[left_mask].to(device),
        )
        verts[left_mask] = vl.to(device=device, dtype=dtype) / 1000.0
        joints[left_mask] = jl.to(device=device, dtype=dtype) / 1000.0

    if root_relative:
        root = joints[:, 0:1].clone()
        verts = verts - root
        joints = joints - root
    return verts, joints


class HandRegressionHead(nn.Module):
    """
    VPHO-style MANO regression head.

    Input:
        psi_h: [B,1024] hand visual condition encoding.
    Output:
        pose_aa: [B,48] MANO axis-angle pose.
        shape: [B,10] MANO beta.

    Internally the pose head predicts 96D rot6d, then decodes it to axis-angle.
    This follows the official VPHO idea that regression result can be used as an
    additional candidate for aggregation.
    """

    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pose6d_head = nn.Linear(hidden_dim, 16 * 6)
        self.shape_head = nn.Linear(hidden_dim, 10)

    def forward(self, psi_h):
        feat = self.trunk(psi_h)
        pose6d = self.pose6d_head(feat).reshape(-1, 16, 6)
        pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pose6d)).reshape(-1, 48)
        shape = self.shape_head(feat)
        return pose_aa, shape

    def get_loss(self, pd_pose_aa, pd_shape, gt_theta_96, gt_betas, is_right):
        gt_pose_aa = pose96_to_axis_angle_torch(gt_theta_96)
        pd_theta_96 = axis_angle_to_pose96_torch(pd_pose_aa)

        mano_pose_loss = F.mse_loss(pd_theta_96, gt_theta_96)
        mano_shape_loss = F.mse_loss(pd_shape, gt_betas)

        pd_vert, pd_joint = mano_forward_mixed_side(pd_pose_aa, pd_shape, is_right)
        with torch.no_grad():
            gt_vert, gt_joint = mano_forward_mixed_side(gt_pose_aa, gt_betas, is_right)

        vert_loss = F.mse_loss(pd_vert, gt_vert)
        joint_loss = F.mse_loss(pd_joint, gt_joint)
        return {
            "mano_pose_loss": mano_pose_loss,
            "mano_shape_loss": mano_shape_loss,
            "vert_loss": vert_loss,
            "joint_loss": joint_loss,
        }


class ObjectDiffusionModelWithHandRoot(nn.Module):
    """
    Object diffusion model conditioned on object visual code, hand visual code,
    and 2D hand-object geometry.
    """

    def __init__(self, cond_dim=1024, pose_dim=9, geom_dim=12, time_dim=256, hidden_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obj_cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hand_cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(pose_dim + hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, phi_t, t, psi_obj, psi_hand, geom_cond):
        t_emb = self.time_mlp(t)
        obj_emb = self.obj_cond_mlp(psi_obj)
        hand_emb = self.hand_cond_mlp(psi_hand)
        geom_emb = self.geom_mlp(geom_cond)
        return self.net(torch.cat([phi_t, t_emb, obj_emb, hand_emb, geom_emb], dim=-1))


class ObjectRegressionHead(nn.Module):
    """
    Simple object regression anchor.

    It directly predicts phi=[rot6d, relative translation]. It is not a full
    official object head, but it gives aggregation a stable candidate.
    """

    def __init__(self, in_dim=1024 + 1024 + 12, hidden_dim=512, pose_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, psi_o, psi_h=None, geom_cond=None):
        parts = [psi_o]
        if psi_h is not None:
            parts.append(psi_h)
        if geom_cond is not None:
            parts.append(geom_cond)
        return self.net(torch.cat(parts, dim=-1))

    def get_loss(self, pred_phi, gt_phi, rot_weight=1.0, trans_weight=10.0):
        rot_loss = F.mse_loss(pred_phi[:, :6], gt_phi[:, :6])
        trans_loss = F.mse_loss(pred_phi[:, 6:], gt_phi[:, 6:])
        return {
            "obj_reg_rot6d_loss": rot_loss,
            "obj_reg_trans_loss": trans_loss,
            "obj_reg_total_loss": rot_weight * rot_loss + trans_weight * trans_loss,
        }

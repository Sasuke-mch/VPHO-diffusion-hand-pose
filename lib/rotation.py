"""
Rotation conversion utilities.

This file intentionally uses one consistent 6D convention:
    matrix_to_rotation_6d(R) = R[..., :2, :].reshape(..., 6)
which matches PyTorch3D's official implementation: drop the last row.

Important:
- pose_48_to_96: axis-angle MANO pose [16,3] -> rot6d [16,6]
- pose_96_to_48: rot6d [16,6] -> axis-angle [16,3]
- Do NOT call pca_to_axis_angle after pose_96_to_48. PCA decoding only happens once in dataset.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_tensor(x, dtype=torch.float32, device=None):
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device if device is not None else x.device)
    return torch.tensor(x, dtype=dtype, device=device)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Vectorized Rodrigues formula. axis_angle: [...,3] -> matrix [...,3,3]."""
    axis_angle = _to_tensor(axis_angle)
    orig_shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)

    angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    ca = torch.cos(angle[:, 0])
    sa = torch.sin(angle[:, 0])
    C = 1.0 - ca

    R = torch.zeros((aa.shape[0], 3, 3), dtype=aa.dtype, device=aa.device)
    R[:, 0, 0] = ca + x * x * C
    R[:, 0, 1] = x * y * C - z * sa
    R[:, 0, 2] = x * z * C + y * sa
    R[:, 1, 0] = y * x * C + z * sa
    R[:, 1, 1] = ca + y * y * C
    R[:, 1, 2] = y * z * C - x * sa
    R[:, 2, 0] = z * x * C - y * sa
    R[:, 2, 1] = z * y * C + x * sa
    R[:, 2, 2] = ca + z * z * C

    small = (torch.linalg.norm(aa, dim=-1) < 1e-8)
    if small.any():
        R[small] = torch.eye(3, dtype=aa.dtype, device=aa.device)

    return R.reshape(*orig_shape, 3, 3)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix [...,3,3] -> axis-angle [...,3]."""
    matrix = _to_tensor(matrix)
    orig_shape = matrix.shape[:-2]
    R = matrix.reshape(-1, 3, 3)

    cos_angle = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1.0) * 0.5
    cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)

    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)
    axis = F.normalize(axis, dim=-1, eps=1e-8)
    aa = axis * angle[:, None]

    small = angle < 1e-6
    if small.any():
        aa[small] = 0.0

    return aa.reshape(*orig_shape, 3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """PyTorch3D convention: drop the last row. matrix [...,3,3] -> [...,6]."""
    matrix = _to_tensor(matrix)
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].clone().reshape(*batch_dim, 6)


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias. For a single [3,3] matrix, returns [6]."""
    return matrix_to_rotation_6d(R)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """PyTorch3D convention. d6 [...,6] -> matrix [...,3,3]."""
    d6 = _to_tensor(d6)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def axis_angle_to_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def pose_48_to_96(pose_48: np.ndarray | torch.Tensor) -> np.ndarray:
    """MANO axis-angle pose [48] or [B,48] -> rot6d [96] or [B,96]. Returns numpy."""
    is_torch = torch.is_tensor(pose_48)
    single = pose_48.ndim == 1
    x = _to_tensor(pose_48).float()
    if single:
        x = x.unsqueeze(0)
    B = x.shape[0]
    rot = axis_angle_to_matrix(x.reshape(B, 16, 3))
    out = matrix_to_rotation_6d(rot).reshape(B, 96)
    if is_torch:
        return out[0] if single else out
    out_np = out.detach().cpu().numpy().astype(np.float32)
    return out_np[0] if single else out_np


def pose_96_to_48(pose_96: torch.Tensor | np.ndarray) -> torch.Tensor:
    """MANO rot6d pose [96] or [B,96] -> axis-angle [48] or [B,48]. Returns torch.Tensor."""
    single = pose_96.ndim == 1
    x = _to_tensor(pose_96).float()
    if single:
        x = x.unsqueeze(0)
    B = x.shape[0]
    R = rotation_6d_to_matrix(x.reshape(B, 16, 6))
    aa = matrix_to_axis_angle(R).reshape(B, 48)
    return aa[0] if single else aa


def pca_to_axis_angle(pose_48_pca: np.ndarray, mano_layer) -> np.ndarray:
    """
    Decode DexYCB pose_m[:48] from MANO PCA to axis-angle.
    pose_48_pca = [global_orient(3), hand_pca(45)].
    """
    if pose_48_pca.ndim == 1:
        pose_48_pca = pose_48_pca.reshape(1, -1)
        return_single = True
    else:
        return_single = False

    hands_components = mano_layer.smpl_data["hands_components"]
    hands_mean = mano_layer.smpl_data["hands_mean"]

    out = []
    for b in range(pose_48_pca.shape[0]):
        global_orient = pose_48_pca[b, :3]
        hand_pose_pca = pose_48_pca[b, 3:48]
        hand_pose_aa = np.matmul(hand_pose_pca, hands_components) + hands_mean
        out.append(np.concatenate([global_orient, hand_pose_aa]))
    out = np.stack(out).astype(np.float32)
    return out[0] if return_single else out


def pose_96_to_3d_joints(theta_96, mano_layer, device, betas=None):
    """
    rot6d pose -> MANO joints in meters, root-relative.
    This function does NOT apply PCA decoding.
    """
    theta_48_aa = pose_96_to_48(theta_96).to(device=device, dtype=torch.float32)
    if theta_48_aa.dim() == 1:
        theta_48_aa = theta_48_aa.unsqueeze(0)
    if betas is None:
        betas = torch.zeros(theta_48_aa.shape[0], 10, device=device, dtype=torch.float32)
    else:
        betas = _to_tensor(betas, device=device)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
    with torch.no_grad():
        _, joints = mano_layer(theta_48_aa, betas)
    joints = joints / 1000.0
    return joints - joints[:, 0:1]

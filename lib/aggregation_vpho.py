"""
Lightweight VPHO-style visual aggregation.

This module is designed for your current simplified project:
- hand candidates are MANO axis-angle [B,N,48]
- object candidates are phi=[rot6d, root-relative translation] [B,N,9]
- heatmaps are ROI heatmaps [B,J,64,64]
- bbox_px is in the resized 256x256 image coordinate system
- cam_intrinsic is the 256-scaled camera intrinsic matrix
- root_joint is the GT wrist/root in camera coordinates, used for DexYCB label-backed eval

For real deployment without labels, root_joint must be predicted by another module.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None

from lib.rotation import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)
from lib.models import mano_forward_mixed_side
from lib.geometry import compute_object_keypoints_3d


def project_points(points_cam: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """points_cam [B,...,3], K [B,3,3] -> uv [B,...,2]."""
    uvw = torch.einsum("bij,b...j->b...i", K, points_cam)
    return uvw[..., :2] / uvw[..., 2:].clamp_min(1e-8)


def roi_grid_from_points(uv_px: torch.Tensor, bbox_px: torch.Tensor) -> torch.Tensor:
    """uv_px [B,N,J,2], bbox_px [B,4] -> grid [B,N,J,2] in [-1,1]."""
    box = bbox_px[:, None, None, :]
    xy = (uv_px - box[..., :2]) / (box[..., 2:] - box[..., :2]).clamp_min(1e-6)
    return xy * 2.0 - 1.0


def heatmap_score_points(heatmap: torch.Tensor, uv_px: torch.Tensor, bbox_px: torch.Tensor) -> torch.Tensor:
    """
    Score projected 2D points by sampling corresponding heatmap channels.

    heatmap: [B,C,H,W]
    uv_px: [B,N,J,2]
    bbox_px: [B,4]
    returns: score [B,N]
    """
    B, C, H, W = heatmap.shape
    _, N, J, _ = uv_px.shape
    grid = roi_grid_from_points(uv_px, bbox_px)
    J_use = min(C, J)
    score = heatmap.new_zeros(B, N)
    valid = ((grid[..., 0] >= -1) & (grid[..., 0] <= 1) & (grid[..., 1] >= -1) & (grid[..., 1] <= 1)).float()

    for j in range(J_use):
        hm_j = heatmap[:, j:j + 1]
        grid_j = grid[:, :, j:j + 1, :].reshape(B, N, 1, 2)
        val = F.grid_sample(hm_j, grid_j, mode="bilinear", align_corners=False)
        val = val.squeeze(1).squeeze(-1)  # [B,N]
        score = score + val * valid[:, :, j]
    return score


def weighted_project_so3(R: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted average of rotations via SVD projection. R [B,K,J,3,3], weights [B,K]."""
    W = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    M = (R * W[:, :, None, None, None]).sum(dim=1)  # [B,J,3,3]
    U, _, Vh = torch.linalg.svd(M)
    R_avg = U @ Vh
    det = torch.det(R_avg)
    bad = det < 0
    if bad.any():
        U = U.clone()
        U[bad, :, -1] *= -1
        R_avg = U @ Vh
    return R_avg


def topk_weights(score: torch.Tensor, k: int, temperature: float = 0.1):
    K = min(k, score.shape[1])
    val, idx = score.topk(K, dim=1)
    w = torch.softmax(val / max(temperature, 1e-6), dim=1)
    return val, idx, w


class HandAggregator:
    """Heatmap-based hand candidate selector/fuser."""

    def __init__(self, topk: int = 10, temperature: float = 0.1):
        self.topk = topk
        self.temperature = temperature

    def select_by_heatmap(
        self,
        pose_candidates_aa: torch.Tensor,
        shape_candidates: torch.Tensor,
        is_right: torch.Tensor,
        root_joint: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        heatmap: torch.Tensor,
        bbox_px: torch.Tensor,
        fuse: bool = True,
    ):
        """
        Args:
            pose_candidates_aa: [B,N,48]
            shape_candidates: [B,N,10]
            root_joint: [B,3]
            cam_intrinsic: [B,3,3]
            heatmap: [B,21,64,64]
            bbox_px: [B,4]
        """
        B, N, _ = pose_candidates_aa.shape
        pose_flat = pose_candidates_aa.reshape(B * N, 48)
        shape_flat = shape_candidates.reshape(B * N, 10)
        side_flat = is_right[:, None].expand(B, N).reshape(B * N)

        _, joints = mano_forward_mixed_side(pose_flat, shape_flat, side_flat, root_relative=True)
        joints = joints.reshape(B, N, 21, 3)
        joints_cam = joints + root_joint[:, None, None, :]
        uv = project_points(joints_cam, cam_intrinsic)
        score = heatmap_score_points(heatmap, uv, bbox_px)

        val, idx, w = topk_weights(score, self.topk, self.temperature)
        batch = torch.arange(B, device=pose_candidates_aa.device)[:, None]
        top_pose = pose_candidates_aa[batch, idx]
        top_shape = shape_candidates[batch, idx]

        if not fuse:
            agg_pose = top_pose[:, 0]
            agg_shape = top_shape[:, 0]
        else:
            R = axis_angle_to_matrix(top_pose.reshape(B, -1, 16, 3))
            R_avg = weighted_project_so3(R, w)
            agg_pose = matrix_to_axis_angle(R_avg).reshape(B, 48)
            agg_shape = (top_shape * w[:, :, None]).sum(dim=1)

        agg_vert, agg_joint = mano_forward_mixed_side(agg_pose, agg_shape, is_right, root_relative=True)
        return {
            "agg_pose_aa": agg_pose,
            "agg_shape": agg_shape,
            "agg_vert": agg_vert,
            "agg_joint": agg_joint,
            "score": score,
            "topk_idx": idx,
            "topk_score": val,
            "topk_weight": w,
        }


class ObjectKeypointBank:
    def __init__(self):
        self.cache = {}

    def get(self, mesh_path: str) -> torch.Tensor:
        if mesh_path not in self.cache:
            if o3d is None:
                raise ImportError("open3d is required to load object mesh keypoints")
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(mesh_path)
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if mesh.is_empty():
                raise ValueError(f"empty mesh: {mesh_path}")
            verts = np.asarray(mesh.vertices)
            if verts.size > 0 and float(verts.max() - verts.min()) > 10.0:
                mesh.scale(1.0 / 1000.0, center=(0, 0, 0))
            self.cache[mesh_path] = torch.tensor(compute_object_keypoints_3d(mesh), dtype=torch.float32)
        return self.cache[mesh_path]


class ObjectAggregator:
    """Object candidate selector/fuser by object keypoint heatmap reprojection."""

    def __init__(self, topk: int = 5, temperature: float = 0.1):
        self.topk = topk
        self.temperature = temperature
        self.bank = ObjectKeypointBank()

    def _project_candidates(self, pose_candidates_9d, object_mesh_paths, root_joint, K):
        B, N, _ = pose_candidates_9d.shape
        all_uv = []
        for b in range(B):
            local = self.bank.get(object_mesh_paths[b]).to(pose_candidates_9d.device, pose_candidates_9d.dtype)
            R = rotation_6d_to_matrix(pose_candidates_9d[b, :, :6])  # [N,3,3]
            t = pose_candidates_9d[b, :, 6:] + root_joint[b:b + 1]
            pts = torch.einsum("vj,nij->nvi", local, R.transpose(-1, -2)) + t[:, None, :]
            uv = project_points(pts[None], K[b:b + 1])[0]
            all_uv.append(uv)
        return torch.stack(all_uv, dim=0)  # [B,N,J,2]

    def select_by_heatmap(
        self,
        pose_candidates_9d: torch.Tensor,
        object_mesh_paths,
        root_joint: torch.Tensor,
        cam_intrinsic: torch.Tensor,
        heatmap: torch.Tensor,
        bbox_px: torch.Tensor,
        fuse: bool = True,
    ):
        uv = self._project_candidates(pose_candidates_9d, object_mesh_paths, root_joint, cam_intrinsic)
        score = heatmap_score_points(heatmap, uv, bbox_px)
        val, idx, w = topk_weights(score, self.topk, self.temperature)

        B = pose_candidates_9d.shape[0]
        batch = torch.arange(B, device=pose_candidates_9d.device)[:, None]
        top_pose = pose_candidates_9d[batch, idx]

        if not fuse:
            agg = top_pose[:, 0]
        else:
            R = rotation_6d_to_matrix(top_pose[:, :, :6])  # [B,K,3,3]
            # for object there is one rotation, use weighted SVD projection
            W = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            M = (R * W[:, :, None, None]).sum(dim=1)
            U, _, Vh = torch.linalg.svd(M)
            R_avg = U @ Vh
            det = torch.det(R_avg)
            bad = det < 0
            if bad.any():
                U = U.clone()
                U[bad, :, -1] *= -1
                R_avg = U @ Vh
            rot6d = matrix_to_rotation_6d(R_avg)
            trans = (top_pose[:, :, 6:] * W[:, :, None]).sum(dim=1)
            agg = torch.cat([rot6d, trans], dim=-1)

        return {
            "agg_pose_9d": agg,
            "score": score,
            "topk_idx": idx,
            "topk_score": val,
            "topk_weight": w,
        }

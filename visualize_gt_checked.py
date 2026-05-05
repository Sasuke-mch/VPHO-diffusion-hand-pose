
import os
import sys
import argparse
import random

sys.path.insert(0, ".")

import numpy as np
import torch
import open3d as o3d
import yaml

from lib.config import device, transform, mano_layer_left, mano_layer_right
from lib.dataset import (
    DexYCBDataset,
    normalize_path,
    load_sequence_meta_from_label,
    select_transform_from_pose_y,
    get_target_object_id_and_name,
    parse_camera_serial_from_path,
)
from lib.rotation import pca_to_axis_angle, rotation_matrix_to_6d, rotation_6d_to_matrix


def extract_pose48(pose_m):
    pose_m = np.asarray(pose_m)
    if pose_m.shape == (1, 51):
        return pose_m[0, :48].astype(np.float32)
    if pose_m.shape == (51,):
        return pose_m[:48].astype(np.float32)
    return pose_m.reshape(-1)[:48].astype(np.float32)


def extract_root(pose_m, joint_3d, root_mode):
    if root_mode == "joint3d_root":
        return np.asarray(joint_3d)[0, 0].astype(np.float32)
    pose_m = np.asarray(pose_m)
    if pose_m.shape == (1, 51):
        return pose_m[0, -3:].astype(np.float32)
    if pose_m.shape == (51,):
        return pose_m[-3:].astype(np.float32)
    return pose_m.reshape(-1)[-3:].astype(np.float32)


def load_betas_from_seq(label_path):
    seq_dir = os.path.dirname(os.path.dirname(label_path))
    pose_npz = os.path.join(seq_dir, "pose.npz")
    if os.path.exists(pose_npz):
        data = np.load(pose_npz, allow_pickle=True)
        for k in ["betas", "mano_shape", "shape"]:
            if k in data:
                arr = np.asarray(data[k]).reshape(-1)
                if arr.size >= 10:
                    return arr[:10].astype(np.float32)
    return np.zeros(10, dtype=np.float32)


def normalize_mesh_unit_to_meter(mesh):
    verts = np.asarray(mesh.vertices)
    if verts.size and float(verts.max() - verts.min()) > 10.0:
        mesh.scale(1.0 / 1000.0, center=(0, 0, 0))
    return mesh


def make_gt_hand_mesh_from_label(label_path, mano_layer, hand_pose_mode, color):
    data = np.load(label_path, allow_pickle=True)
    pose48 = extract_pose48(data["pose_m"])
    betas = load_betas_from_seq(label_path)

    if hand_pose_mode == "axis_angle":
        theta_aa = pose48.reshape(1, 48).astype(np.float32)
    elif hand_pose_mode == "pca":
        theta_aa = pca_to_axis_angle(pose48.reshape(1, 48), mano_layer)
        if theta_aa.ndim == 1:
            theta_aa = theta_aa.reshape(1, 48)
    else:
        raise ValueError(f"未知 hand_pose_mode: {hand_pose_mode}")

    theta = torch.tensor(theta_aa, dtype=torch.float32, device=device)
    beta = torch.tensor(betas.reshape(1, -1), dtype=torch.float32, device=device)

    with torch.no_grad():
        verts, joints = mano_layer(theta, beta)

    verts = verts[0].detach().cpu().numpy() / 1000.0
    joints = joints[0].detach().cpu().numpy() / 1000.0
    wrist = joints[0].copy()

    verts -= wrist
    joints -= wrist

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(mano_layer.th_faces.detach().cpu().numpy())
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh, joints


def make_gt_object_mesh_from_label(label_path, object_mesh_path, root_mode, color):
    data = np.load(label_path, allow_pickle=True)
    R, T, obj_idx = select_transform_from_pose_y(label_path, data["pose_y"], strict=True)
    root = extract_root(data["pose_m"], data["joint_3d"], root_mode)
    trans_rel = T.astype(np.float32) - root.astype(np.float32)

    mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    if mesh.is_empty():
        raise ValueError(f"empty object mesh: {object_mesh_path}")
    mesh = normalize_mesh_unit_to_meter(mesh)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = trans_rel
    mesh.transform(M)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="try", choices=["try", "train", "test", "all"])
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hand_pose_mode", default="pca", choices=["axis_angle", "pca"])
    parser.add_argument("--root_mode", default="joint3d_root", choices=["joint3d_root", "pose_m_trans"])
    parser.add_argument("--sample_index", type=int, default=-1, help="指定 dataset index；默认随机")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_root = normalize_path(args.data_root)
    dataset = DexYCBDataset(data_root=data_root, split=args.split, transform=transform, T=1000, strict_pose_index=True)

    random.seed(args.seed)
    if args.sample_index >= 0:
        indices = [args.sample_index]
    else:
        indices = random.sample(range(len(dataset)), min(args.num_images, len(dataset)))

    for idx in indices:
        image_path, label_path, betas, camera_id = dataset.samples[idx]
        meta = load_sequence_meta_from_label(label_path)
        side = "right"
        if meta.get("mano_sides"):
            side = str(meta["mano_sides"][0]).lower()
        mano_layer = mano_layer_left if side == "left" else mano_layer_right

        # target object mesh path
        _, obj_name = get_target_object_id_and_name(label_path)
        mesh_path = dataset._find_object_mesh_path(obj_name)

        if args.debug:
            data = np.load(label_path, allow_pickle=True)
            root_joint3d = np.asarray(data["joint_3d"])[0, 0]
            pose_m_trans = np.asarray(data["pose_m"]).reshape(-1)[-3:]
            print("\nidx:", idx)
            print("image_path:", image_path)
            print("label_path:", label_path)
            print("object:", obj_name)
            print("mano_side:", side)
            print("pose_y.shape:", data["pose_y"].shape)
            print("pose_m_trans:", pose_m_trans)
            print("joint3d_root:", root_joint3d)
            print("root_diff:", np.linalg.norm(pose_m_trans - root_joint3d))
            print("hand_pose_mode:", args.hand_pose_mode, "root_mode:", args.root_mode)

        hand = make_gt_hand_mesh_from_label(label_path, mano_layer, args.hand_pose_mode, [0.7, 0.7, 0.7])
        obj = make_gt_object_mesh_from_label(label_path, mesh_path, args.root_mode, [0.2, 0.55, 0.85])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([frame, hand[0], obj], window_name=f"GT checked sample {idx}")


if __name__ == "__main__":
    main()

import os
import sys
import argparse
import random
sys.path.insert(0, ".")

import numpy as np
import torch
import open3d as o3d

from lib.config import device, transform, mano_layer_left, mano_layer_right
from lib.dataset import DexYCBDataset, load_sequence_meta_from_label, normalize_path
from lib.diffusion import (
    precompute_diffusion_coeffs,
    roi_crop,
    generate_hand_candidates,
    generate_object_candidates_handroot,
)
from lib.models import (
    FeatureExtractor,
    HeatmapPredictor,
    FeatureEncoder,
    HandleDiffusionModel,
    ObjectDiffusionModelWithHandRoot,
    HandRegressionHead,
    ObjectRegressionHead,
    pose96_to_axis_angle_torch,
)
from lib.rotation import pose_96_to_48, rotation_6d_to_matrix
from lib.aggregation_vpho import HandAggregator, ObjectAggregator


def parse_args():
    p = argparse.ArgumentParser("Visualize with VPHO-style regression candidates + aggregation")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["test", "try", "train", "all"])
    p.add_argument("--num_images", type=int, default=3)
    p.add_argument("--num_candidates", type=int, default=50)
    p.add_argument("--candidate_index", type=int, default=0)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cond_dim", type=int, default=1024)
    p.add_argument("--pose_dim_hand", type=int, default=96)
    p.add_argument("--pose_dim_obj", type=int, default=9)
    p.add_argument("--aggregate", action="store_true", help="使用 heatmap aggregation 选择/融合候选")
    p.add_argument("--topk_hand", type=int, default=10)
    p.add_argument("--topk_obj", type=int, default=5)
    p.add_argument("--include_reg_candidate", action="store_true")
    p.add_argument("--no_fuse", action="store_true", help="只选 top1，不做 top-k 融合")
    p.add_argument("--show_gt", action="store_true")
    p.add_argument("--only_gt", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--save_screenshot_dir", type=str, default="")
    return p.parse_args()


def to_device_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.detach().clone().to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)


def set_eval(*models):
    for m in models:
        m.eval()


def load_checkpoint(ckpt_path, modules):
    ckpt = torch.load(ckpt_path, map_location=device)
    for key, module in modules.items():
        if key in ckpt:
            try:
                module.load_state_dict(ckpt[key], strict=False)
                print(f"已加载 {key}")
            except Exception as e:
                print(f"加载 {key} 失败: {e}")
        else:
            print(f"checkpoint 中没有 {key}")
    return ckpt


def build_object_geom_condition(hand_heatmap, bbox_hand_norm, bbox_obj_norm):
    root_hm = hand_heatmap[:, 0]
    B, H, W = root_hm.shape
    flat_idx = torch.argmax(root_hm.reshape(B, -1), dim=1)
    y = (flat_idx // W).float()
    x = (flat_idx % W).float()
    rx = (x + 0.5) / float(W)
    ry = (y + 0.5) / float(H)

    x1, y1, x2, y2 = bbox_hand_norm[:, 0], bbox_hand_norm[:, 1], bbox_hand_norm[:, 2], bbox_hand_norm[:, 3]
    root_uv = torch.stack([x1 + rx * (x2 - x1), y1 + ry * (y2 - y1)], dim=-1)
    hand_center = 0.5 * (bbox_hand_norm[:, :2] + bbox_hand_norm[:, 2:4])
    obj_center = 0.5 * (bbox_obj_norm[:, :2] + bbox_obj_norm[:, 2:4])
    center_delta = obj_center - hand_center
    hand_size = (bbox_hand_norm[:, 2:4] - bbox_hand_norm[:, :2]).clamp(min=1e-4)
    obj_size = (bbox_obj_norm[:, 2:4] - bbox_obj_norm[:, :2]).clamp(min=1e-4)
    return torch.cat([root_uv, hand_center, obj_center, center_delta, hand_size, obj_size], dim=-1)


def normalize_mesh_unit_to_meter(mesh):
    verts = np.asarray(mesh.vertices)
    if verts.size > 0 and float(verts.max() - verts.min()) > 10.0:
        mesh.scale(1.0 / 1000.0, center=(0, 0, 0))
    return mesh


def make_hand_mesh_from_aa(pose_aa, betas, mano_layer, color):
    pose_aa = to_device_tensor(pose_aa)
    if pose_aa.dim() == 1:
        pose_aa = pose_aa.unsqueeze(0)
    betas = to_device_tensor(betas)
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)
    with torch.no_grad():
        verts, joints = mano_layer(pose_aa, betas)
    verts = verts[0].detach().cpu().numpy() / 1000.0
    joints = joints[0].detach().cpu().numpy() / 1000.0
    root = joints[0].copy()
    verts = verts - root
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(mano_layer.th_faces.detach().cpu().numpy())
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def make_hand_mesh(theta_96, betas, mano_layer, color):
    pose_aa = pose_96_to_48(to_device_tensor(theta_96).reshape(1, -1))[0]
    return make_hand_mesh_from_aa(pose_aa, betas, mano_layer, color)


def make_object_mesh(phi_9d, object_mesh_path, color):
    mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    if mesh.is_empty():
        raise ValueError(f"物体 mesh 为空: {object_mesh_path}")
    mesh = normalize_mesh_unit_to_meter(mesh)
    phi = to_device_tensor(phi_9d).reshape(-1).detach().cpu()
    R = rotation_6d_to_matrix(phi[:6].unsqueeze(0))[0].detach().cpu().numpy()
    trans = phi[6:9].numpy()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = trans
    mesh.transform(T)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def visualize_geometries(geometries, window_name, screenshot_path=None):
    if screenshot_path is None:
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
        return
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, visible=True)
    for g in geometries:
        vis.add_geometry(g)
    vis.poll_events(); vis.update_renderer()
    vis.capture_screen_image(screenshot_path)
    vis.destroy_window()
    print(f"截图已保存: {screenshot_path}")


def run_one_sample(dataset, idx, models, alpha, alpha_bar, args, ckpt_meta):
    (
        feature_extractor,
        heatmap_predictor,
        hand_encoder,
        obj_encoder,
        hand_model,
        object_model,
        hand_reg_head,
        obj_reg_head,
    ) = models

    sample = dataset[idx]
    (
        _theta_t, _phi_t, _t, image, theta_0, phi_0, _eps_hand, _eps_obj,
        betas, object_mesh_path, bbox_hand_norm, bbox_obj_norm,
    ) = sample
    image_path, label_path, _betas, camera_id = dataset.samples[idx]
    meta = load_sequence_meta_from_label(label_path)
    side = str(meta.get("mano_sides", ["right"])[0]).lower()
    mano_layer = mano_layer_left if side == "left" else mano_layer_right
    is_right = torch.tensor([side == "right"], device=device, dtype=torch.bool)

    label_data = np.load(label_path, allow_pickle=True)
    root_joint = torch.tensor(label_data["joint_3d"][0, 0], dtype=torch.float32, device=device).reshape(1, 3)
    K = dataset.intrinsics_cache.get(int(camera_id), None)
    if K is None:
        raise RuntimeError(f"找不到相机内参: {camera_id}")
    K = K.to(device=device, dtype=torch.float32).reshape(1, 3, 3)

    image_batch = image.unsqueeze(0).to(device, dtype=torch.float32)
    theta_0 = to_device_tensor(theta_0).reshape(-1)
    phi_0 = to_device_tensor(phi_0).reshape(-1)
    betas = to_device_tensor(betas).reshape(1, -1)
    bbox_hand_norm = to_device_tensor(bbox_hand_norm).reshape(1, 4)
    bbox_obj_norm = to_device_tensor(bbox_obj_norm).reshape(1, 4)
    bbox_hand_px = bbox_hand_norm * 256.0
    bbox_obj_px = bbox_obj_norm * 256.0

    if args.debug:
        print("\n" + "-" * 80)
        print("idx:", idx)
        print("image_path:", image_path)
        print("label_path:", label_path)
        print("camera_id:", camera_id, "mano_side:", side)
        print("bbox_hand_norm:", bbox_hand_norm.cpu().numpy().reshape(-1))
        print("bbox_obj_norm:", bbox_obj_norm.cpu().numpy().reshape(-1))
        print("GT phi_0:", phi_0.cpu().numpy())
        print("root_joint:", root_joint.cpu().numpy().reshape(-1))

    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])]

    if args.show_gt or args.only_gt:
        geometries.append(make_hand_mesh(theta_0, betas, mano_layer, [0.70, 0.70, 0.70]))
        geometries.append(make_object_mesh(phi_0, object_mesh_path, [0.20, 0.55, 0.85]))

    if not args.only_gt:
        with torch.no_grad():
            hand_feat, obj_feat = feature_extractor(image_batch)
            hand_roi = roi_crop(hand_feat, bbox_hand_norm, output_size=32)
            obj_roi = roi_crop(obj_feat, bbox_obj_norm, output_size=32)
            hand_hm, obj_hm = heatmap_predictor(hand_roi, obj_roi)
            psi_h = hand_encoder(hand_roi, hand_hm)
            psi_o = obj_encoder(obj_roi, obj_hm)
            geom_cond = build_object_geom_condition(hand_hm, bbox_hand_norm, bbox_obj_norm)

            hand_cands_96 = generate_hand_candidates(
                hand_model, psi_h, betas, args.T, device, alpha, alpha_bar,
                args.pose_dim_hand, args.num_candidates,
            ).reshape(1, args.num_candidates, args.pose_dim_hand)
            hand_cands_aa = pose96_to_axis_angle_torch(hand_cands_96.reshape(-1, 96)).reshape(1, args.num_candidates, 48)

            obj_cands = generate_object_candidates_handroot(
                object_model, psi_o, psi_h, geom_cond, args.T, device, alpha, alpha_bar,
                args.pose_dim_obj, args.num_candidates,
            )

            reg_pose_aa, reg_shape = hand_reg_head(psi_h)
            reg_phi = obj_reg_head(psi_o, psi_h, geom_cond)

        has_reg = bool(ckpt_meta.get("has_regression_heads", False)) or ("hand_reg_head_state_dict" in ckpt_meta)
        include_reg = args.include_reg_candidate and has_reg

        shape_pool = betas[:, None].repeat(1, hand_cands_aa.shape[1], 1)
        if include_reg:
            # VPHO style: add regression result as an anchor candidate.
            hand_cands_aa = torch.cat([hand_cands_aa, reg_pose_aa[:, None]], dim=1)
            shape_pool = torch.cat([shape_pool, reg_shape[:, None]], dim=1)
            obj_cands = torch.cat([obj_cands, reg_phi[:, None]], dim=1)

        if args.aggregate:
            hand_agg = HandAggregator(topk=args.topk_hand)
            obj_agg = ObjectAggregator(topk=args.topk_obj)
            hand_out = hand_agg.select_by_heatmap(
                pose_candidates_aa=hand_cands_aa,
                shape_candidates=shape_pool,
                is_right=is_right,
                root_joint=root_joint,
                cam_intrinsic=K,
                heatmap=hand_hm,
                bbox_px=bbox_hand_px,
                fuse=not args.no_fuse,
            )
            obj_out = obj_agg.select_by_heatmap(
                pose_candidates_9d=obj_cands,
                object_mesh_paths=[object_mesh_path],
                root_joint=root_joint,
                cam_intrinsic=K,
                heatmap=obj_hm,
                bbox_px=bbox_obj_px,
                fuse=not args.no_fuse,
            )
            pred_pose_aa = hand_out["agg_pose_aa"][0]
            pred_betas = hand_out["agg_shape"][0]
            pred_phi = obj_out["agg_pose_9d"][0]

            if args.debug:
                print("hand score min/mean/max:", hand_out["score"].min().item(), hand_out["score"].mean().item(), hand_out["score"].max().item())
                print("hand topk_idx:", hand_out["topk_idx"].detach().cpu().numpy())
                print("obj score min/mean/max:", obj_out["score"].min().item(), obj_out["score"].mean().item(), obj_out["score"].max().item())
                print("obj topk_idx:", obj_out["topk_idx"].detach().cpu().numpy())
                print("pred_phi aggregated:", pred_phi.detach().cpu().numpy())
        else:
            cand_idx = max(0, min(args.candidate_index, hand_cands_aa.shape[1] - 1))
            pred_pose_aa = hand_cands_aa[0, cand_idx]
            pred_betas = shape_pool[0, cand_idx]
            pred_phi = obj_cands[0, min(cand_idx, obj_cands.shape[1] - 1)]
            if args.debug:
                print("display candidate:", cand_idx)
                print("pred_phi:", pred_phi.detach().cpu().numpy())

        geometries.append(make_hand_mesh_from_aa(pred_pose_aa, pred_betas, mano_layer, [0.95, 0.25, 0.20]))
        geometries.append(make_object_mesh(pred_phi, object_mesh_path, [0.95, 0.65, 0.10]))

    screenshot_path = None
    if args.save_screenshot_dir:
        os.makedirs(args.save_screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(args.save_screenshot_dir, f"sample_{idx:06d}.png")
    visualize_geometries(geometries, f"DexYCB regagg sample {idx}", screenshot_path)


def main():
    args = parse_args()
    args.data_root = normalize_path(args.data_root)
    args.ckpt = normalize_path(args.ckpt)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DexYCBDataset(args.data_root, split=args.split, transform=transform, T=args.T, strict_pose_index=True)
    indices = random.sample(range(len(dataset)), min(args.num_images, len(dataset)))
    print(f"样本数: {len(dataset)}")
    print(f"选择样本: {indices}")

    feature_extractor = FeatureExtractor().to(device)
    heatmap_predictor = HeatmapPredictor().to(device)
    hand_encoder = FeatureEncoder(256, 21, args.cond_dim, 32).to(device)
    obj_encoder = FeatureEncoder(256, 27, args.cond_dim, 32).to(device)
    hand_model = HandleDiffusionModel(args.pose_dim_hand, args.cond_dim, 10, 256, 512).to(device)
    object_model = ObjectDiffusionModelWithHandRoot(args.cond_dim, args.pose_dim_obj, 12, 256, 512).to(device)
    hand_reg_head = HandRegressionHead(args.cond_dim, 512).to(device)
    obj_reg_head = ObjectRegressionHead(args.cond_dim * 2 + 12, 512, args.pose_dim_obj).to(device)

    modules = {
        "feature_extractor_state_dict": feature_extractor,
        "heatmap_predictor_state_dict": heatmap_predictor,
        "hand_encoder_state_dict": hand_encoder,
        "obj_encoder_state_dict": obj_encoder,
        "hand_model_state_dict": hand_model,
        "object_model_state_dict": object_model,
        "hand_reg_head_state_dict": hand_reg_head,
        "obj_reg_head_state_dict": obj_reg_head,
    }
    ckpt_meta = load_checkpoint(args.ckpt, modules)
    set_eval(*modules.values())

    _beta, alpha, alpha_bar = precompute_diffusion_coeffs(args.T, device=device)
    models = (feature_extractor, heatmap_predictor, hand_encoder, obj_encoder, hand_model, object_model, hand_reg_head, obj_reg_head)

    for idx in indices:
        run_one_sample(dataset, idx, models, alpha, alpha_bar, args, ckpt_meta)


if __name__ == "__main__":
    main()

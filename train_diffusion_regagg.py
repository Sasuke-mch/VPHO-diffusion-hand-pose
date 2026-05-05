import os
import sys
import argparse
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from lib.config import device, transform
from lib.models import (
    FeatureExtractor,
    HeatmapPredictor,
    FeatureEncoder,

    HandleDiffusionModel,
    ObjectDiffusionModelWithHandRoot,
    HandRegressionHead,
    ObjectRegressionHead,
)
from lib.dataset import DexYCBDataset, load_sequence_meta_from_label, normalize_path
from lib.diffusion import roi_crop
from lib.geometry import get_hand_bbox_from_joints
from lib.utils import setup_logger


class DexYCBDatasetRegAgg(DexYCBDataset):
    """Second-stage dataset wrapper that appends side/paths metadata."""

    @staticmethod
    def _safe_tensor(x, dtype=torch.float32):
        if torch.is_tensor(x):
            return x.detach().clone().to(dtype=dtype)
        return torch.tensor(x, dtype=dtype)

    def _fixed_hand_bbox_norm(self, label_path: str) -> torch.Tensor:
        label_data = np.load(label_path, allow_pickle=True)
        joints_2d = label_data["joint_2d"][0].astype(np.float32)
        joints_2d_256 = joints_2d.copy()
        joints_2d_256[:, 0] *= 256.0 / 640.0
        joints_2d_256[:, 1] *= 256.0 / 480.0
        bbox_hand = get_hand_bbox_from_joints(joints_2d_256)
        if bbox_hand[2] <= bbox_hand[0] or bbox_hand[3] <= bbox_hand[1]:
            bbox_hand = np.array([64, 64, 192, 192], dtype=np.float32)
        return torch.tensor(bbox_hand / 256.0, dtype=torch.float32)

    def __getitem__(self, idx: int):
        base = super().__getitem__(idx)
        (
            theta_t,
            phi_t,
            t,
            image,
            theta_0,
            phi_0,
            eps_hand,
            eps_obj,
            betas,
            object_mesh_path,
            _bbox_hand_norm_old,
            bbox_obj_norm,
        ) = base

        image_path, label_path, _betas, camera_id = self.samples[idx]
        bbox_hand_norm = self._fixed_hand_bbox_norm(label_path)
        bbox_obj_norm = self._safe_tensor(bbox_obj_norm)

        meta = load_sequence_meta_from_label(label_path)
        mano_side = "right"
        if meta.get("mano_sides", None):
            mano_side = str(meta["mano_sides"][0]).lower()
        is_right = torch.tensor(mano_side == "right", dtype=torch.bool)

        return (
            theta_t,
            phi_t,
            t,
            image,
            theta_0,
            phi_0,
            eps_hand,
            eps_obj,
            betas,
            object_mesh_path,
            bbox_hand_norm,
            bbox_obj_norm,
            is_right,
            label_path,
            str(camera_id),
        )


def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def heatmap_root_uv_from_bbox(hand_heatmap: torch.Tensor, bbox_hand_norm: torch.Tensor) -> torch.Tensor:
    root_hm = hand_heatmap[:, 0]
    B, H, W = root_hm.shape
    flat_idx = torch.argmax(root_hm.reshape(B, -1), dim=1)
    y = (flat_idx // W).float()
    x = (flat_idx % W).float()
    rx = (x + 0.5) / float(W)
    ry = (y + 0.5) / float(H)
    x1, y1, x2, y2 = bbox_hand_norm[:, 0], bbox_hand_norm[:, 1], bbox_hand_norm[:, 2], bbox_hand_norm[:, 3]
    u = x1 + rx * (x2 - x1)
    v = y1 + ry * (y2 - y1)
    return torch.stack([u, v], dim=-1)


def build_object_geom_condition(hand_heatmap, bbox_hand_norm, bbox_obj_norm):
    root_uv = heatmap_root_uv_from_bbox(hand_heatmap, bbox_hand_norm)
    hand_center = 0.5 * (bbox_hand_norm[:, :2] + bbox_hand_norm[:, 2:4])
    obj_center = 0.5 * (bbox_obj_norm[:, :2] + bbox_obj_norm[:, 2:4])
    center_delta = obj_center - hand_center
    hand_size = (bbox_hand_norm[:, 2:4] - bbox_hand_norm[:, :2]).clamp(min=1e-4)
    obj_size = (bbox_obj_norm[:, 2:4] - bbox_obj_norm[:, :2]).clamp(min=1e-4)
    return torch.cat([root_uv, hand_center, obj_center, center_delta, hand_size, obj_size], dim=-1)


def load_heatmap_checkpoint(heatmap_predictor, feature_extractor, ckpt_path, logger):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"没有找到热图 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        heatmap_predictor.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("已从 model_state_dict 加载 HeatmapPredictor。")
    elif "heatmap_predictor_state_dict" in ckpt:
        heatmap_predictor.load_state_dict(ckpt["heatmap_predictor_state_dict"], strict=False)
        logger.info("已从 heatmap_predictor_state_dict 加载 HeatmapPredictor。")
    else:
        raise KeyError("heatmap checkpoint 中没有 model_state_dict 或 heatmap_predictor_state_dict")

    if "feature_extractor_state_dict" in ckpt:
        feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"], strict=False)
        logger.info("已加载 feature_extractor_state_dict。")


def load_any_checkpoint(path, modules, optimizer=None, logger=None):
    """Load compatible keys from old/new checkpoints."""
    if not path:
        return 0
    if not os.path.exists(path):
        if logger:
            logger.warning(f"resume checkpoint 不存在: {path}")
        return 0
    ckpt = torch.load(path, map_location=device)
    if logger:
        logger.info(f"加载 checkpoint: {path}")
        logger.info(f"checkpoint keys: {list(ckpt.keys())}")
    for key, module in modules.items():
        if key in ckpt:
            try:
                module.load_state_dict(ckpt[key], strict=False)
                if logger:
                    logger.info(f"已加载 {key}")
            except Exception as e:
                if logger:
                    logger.warning(f"加载 {key} 失败: {e}")
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if logger:
                logger.info("已加载 optimizer_state_dict")
        except Exception as e:
            if logger:
                logger.warning(f"optimizer 不兼容，跳过: {e}")
    return int(ckpt.get("epoch", 0))


def save_checkpoint(path, epoch, batch_idx, modules, optimizer, meta):
    ckpt = {"epoch": epoch, "batch_idx": batch_idx, "optimizer_state_dict": optimizer.state_dict()}
    ckpt.update(meta)
    for key, module in modules.items():
        ckpt[key] = module.state_dict()
    torch.save(ckpt, path)


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(args.log_file)
    logger.info(f"Using device: {device}")
    logger.info("开始第二阶段：diffusion + VPHO-style regression heads")

    dataset = DexYCBDatasetRegAgg(args.data_root, split="train", transform=transform, T=args.T)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    logger.info(f"训练样本数: {len(dataset)}")
    logger.info(f"每个 epoch 的 batch 数: {len(loader)}")

    feature_extractor = FeatureExtractor().to(device)
    heatmap_predictor = HeatmapPredictor().to(device)
    hand_encoder = FeatureEncoder(256, 21, args.cond_dim, 32).to(device)
    obj_encoder = FeatureEncoder(256, 27, args.cond_dim, 32).to(device)
    hand_model = HandleDiffusionModel(
        args.pose_dim_hand, args.cond_dim, 10, 256, 512
    ).to(device)
    object_model = ObjectDiffusionModelWithHandRoot(
        args.cond_dim, args.pose_dim_obj, 12, 256, 512
    ).to(device)
    hand_reg_head = HandRegressionHead(args.cond_dim, 512).to(device)
    obj_reg_head = ObjectRegressionHead(args.cond_dim * 2 + 12, 512, args.pose_dim_obj).to(device)

    load_heatmap_checkpoint(heatmap_predictor, feature_extractor, args.heatmap_ckpt, logger)
    set_requires_grad(feature_extractor, False)
    set_requires_grad(heatmap_predictor, False)
    feature_extractor.eval()
    heatmap_predictor.eval()

    train_modules = {
        "hand_encoder_state_dict": hand_encoder,
        "obj_encoder_state_dict": obj_encoder,
        "hand_model_state_dict": hand_model,
        "object_model_state_dict": object_model,
        "hand_reg_head_state_dict": hand_reg_head,
        "obj_reg_head_state_dict": obj_reg_head,
    }
    all_save_modules = {
        **train_modules,
        "feature_extractor_state_dict": feature_extractor,
        "heatmap_predictor_state_dict": heatmap_predictor,
    }

    optimizer = optim.Adam(
        list(hand_encoder.parameters())
        + list(obj_encoder.parameters())
        + list(hand_model.parameters())
        + list(object_model.parameters())
        + list(hand_reg_head.parameters())
        + list(obj_reg_head.parameters()),
        lr=args.lr,
    )

    start_epoch = load_any_checkpoint(args.resume, train_modules, optimizer=None if args.no_resume_optimizer else optimizer, logger=logger)
    criterion = nn.MSELoss()
    best_loss = float("inf")

    meta = {
        "T": args.T,
        "cond_dim": args.cond_dim,
        "pose_dim_hand": args.pose_dim_hand,
        "pose_dim_obj": args.pose_dim_obj,
        "object_model_variant": "with_handroot_geom_v1",
        "has_regression_heads": True,
        "geom_dim": 12,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    for epoch in range(start_epoch, start_epoch + args.epochs):
        hand_encoder.train(); obj_encoder.train(); hand_model.train(); object_model.train()
        hand_reg_head.train(); obj_reg_head.train()
        feature_extractor.eval(); heatmap_predictor.eval()

        sums = {"hand_diff": 0.0, "obj_diff": 0.0, "hand_reg": 0.0, "obj_reg": 0.0, "total": 0.0}

        for batch_idx, batch in enumerate(loader):
            (
                theta_t, phi_t, t, image, theta_0, phi_0, eps_hand, eps_obj,
                betas, object_mesh_path, bbox_hand_norm, bbox_obj_norm,
                is_right, label_path, camera_id,
            ) = batch

            theta_t = theta_t.to(device, dtype=torch.float32, non_blocking=True)
            phi_t = phi_t.to(device, dtype=torch.float32, non_blocking=True)
            t = t.to(device, dtype=torch.long, non_blocking=True)
            image = image.to(device, dtype=torch.float32, non_blocking=True)
            theta_0 = theta_0.to(device, dtype=torch.float32, non_blocking=True)
            phi_0 = phi_0.to(device, dtype=torch.float32, non_blocking=True)
            eps_hand = eps_hand.to(device, dtype=torch.float32, non_blocking=True)
            eps_obj = eps_obj.to(device, dtype=torch.float32, non_blocking=True)
            betas = betas.to(device, dtype=torch.float32, non_blocking=True)
            bbox_hand_norm = bbox_hand_norm.to(device, dtype=torch.float32, non_blocking=True)
            bbox_obj_norm = bbox_obj_norm.to(device, dtype=torch.float32, non_blocking=True)
            is_right = is_right.to(device, non_blocking=True)

            with torch.no_grad():
                hand_feat, obj_feat = feature_extractor(image)
                hand_roi = roi_crop(hand_feat, bbox_hand_norm, output_size=32)
                obj_roi = roi_crop(obj_feat, bbox_obj_norm, output_size=32)
                hand_heatmap, obj_heatmap = heatmap_predictor(hand_roi, obj_roi)

            psi_h = hand_encoder(hand_roi, hand_heatmap)
            psi_o = obj_encoder(obj_roi, obj_heatmap)
            geom_cond = build_object_geom_condition(hand_heatmap, bbox_hand_norm, bbox_obj_norm)

            eps_hand_pred = hand_model(theta_t, t, psi_h, betas)
            eps_obj_pred = object_model(phi_t, t, psi_o, psi_h, geom_cond)
            loss_hand_diff = criterion(eps_hand_pred, eps_hand)
            loss_obj_diff = criterion(eps_obj_pred, eps_obj)

            reg_pose_aa, reg_shape = hand_reg_head(psi_h)
            hand_reg_losses = hand_reg_head.get_loss(reg_pose_aa, reg_shape, theta_0, betas, is_right)
            loss_hand_reg = (
                args.w_mano_pose * hand_reg_losses["mano_pose_loss"]
                + args.w_mano_shape * hand_reg_losses["mano_shape_loss"]
                + args.w_mano_joint * hand_reg_losses["joint_loss"]
                + args.w_mano_vert * hand_reg_losses["vert_loss"]
            )

            reg_phi = obj_reg_head(psi_o, psi_h, geom_cond)
            obj_reg_losses = obj_reg_head.get_loss(reg_phi, phi_0, rot_weight=args.w_obj_rot, trans_weight=args.w_obj_trans)
            loss_obj_reg = obj_reg_losses["obj_reg_total_loss"]

            loss_total = (
                args.w_hand_diff * loss_hand_diff
                + args.w_obj_diff * loss_obj_diff
                + loss_hand_reg
                + loss_obj_reg
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(hand_encoder.parameters()) + list(obj_encoder.parameters())
                + list(hand_model.parameters()) + list(object_model.parameters())
                + list(hand_reg_head.parameters()) + list(obj_reg_head.parameters()),
                max_norm=args.grad_clip,
            )
            optimizer.step()

            sums["hand_diff"] += loss_hand_diff.item()
            sums["obj_diff"] += loss_obj_diff.item()
            sums["hand_reg"] += loss_hand_reg.item()
            sums["obj_reg"] += loss_obj_reg.item()
            sums["total"] += loss_total.item()

            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{start_epoch + args.epochs}] "
                    f"Batch [{batch_idx}/{len(loader)}] "
                    f"hand_diff={loss_hand_diff.item():.6f} "
                    f"obj_diff={loss_obj_diff.item():.6f} "
                    f"hand_reg={loss_hand_reg.item():.6f} "
                    f"obj_reg={loss_obj_reg.item():.6f} "
                    f"total={loss_total.item():.6f} "
                    f"mano_pose={hand_reg_losses['mano_pose_loss'].item():.6f} "
                    f"joint={hand_reg_losses['joint_loss'].item():.6f} "
                    f"obj_trans={obj_reg_losses['obj_reg_trans_loss'].item():.6f}"
                )

            if batch_idx > 0 and batch_idx % args.save_interval == 0:
                latest = os.path.join(args.save_dir, "diffusion_regagg_latest.pth")
                save_checkpoint(latest, epoch + 1, batch_idx, all_save_modules, optimizer, meta)
                logger.info(f"已保存中间 checkpoint: {latest}")

        avg = {k: v / len(loader) for k, v in sums.items()}
        logger.info(
            f"Epoch [{epoch + 1}/{start_epoch + args.epochs}] 完成 | "
            f"hand_diff={avg['hand_diff']:.6f}, obj_diff={avg['obj_diff']:.6f}, "
            f"hand_reg={avg['hand_reg']:.6f}, obj_reg={avg['obj_reg']:.6f}, total={avg['total']:.6f}"
        )

        epoch_path = os.path.join(args.save_dir, f"diffusion_regagg_epoch_{epoch + 1}.pth")
        save_checkpoint(epoch_path, epoch + 1, -1, all_save_modules, optimizer, {**meta, **{f"avg_{k}": v for k, v in avg.items()}})
        save_checkpoint(os.path.join(args.save_dir, "diffusion_regagg_latest.pth"), epoch + 1, -1, all_save_modules, optimizer, {**meta, **{f"avg_{k}": v for k, v in avg.items()}})
        logger.info(f"已保存 epoch checkpoint: {epoch_path}")

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            best_path = os.path.join(args.save_dir, "diffusion_regagg_best.pth")
            save_checkpoint(best_path, epoch + 1, -1, all_save_modules, optimizer, {**meta, "best_loss": best_loss})
            logger.info(f"已保存 best checkpoint: {best_path}")

    logger.info("训练完成。")


def parse_args():
    p = argparse.ArgumentParser("Train diffusion + VPHO-style regression heads")
    p.add_argument("--data_root", type=str, default="/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb")
    p.add_argument("--heatmap_ckpt", type=str, default="checkpoints/heatmap_epoch_4.pth")
    p.add_argument("--resume", type=str, default="", help="可加载 diffusion_handroot_latest 或 diffusion_regagg_latest")
    p.add_argument("--no_resume_optimizer", action="store_true")
    p.add_argument("--save_dir", type=str, default="checkpoints_regagg")
    p.add_argument("--log_file", type=str, default="train_diffusion_regagg.log")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--pose_dim_hand", type=int, default=96)
    p.add_argument("--pose_dim_obj", type=int, default=9)
    p.add_argument("--cond_dim", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--w_hand_diff", type=float, default=1.0)
    p.add_argument("--w_obj_diff", type=float, default=1.0)
    p.add_argument("--w_mano_pose", type=float, default=1.0)
    p.add_argument("--w_mano_shape", type=float, default=0.1)
    p.add_argument("--w_mano_joint", type=float, default=10.0)
    p.add_argument("--w_mano_vert", type=float, default=1.0)
    p.add_argument("--w_obj_rot", type=float, default=1.0)
    p.add_argument("--w_obj_trans", type=float, default=10.0)
    args = p.parse_args()
    args.data_root = normalize_path(args.data_root)
    args.heatmap_ckpt = normalize_path(args.heatmap_ckpt)
    args.resume = normalize_path(args.resume) if args.resume else ""
    return args


if __name__ == "__main__":
    train(parse_args())

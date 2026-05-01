import os
import sys
sys.path.insert(0, ".")

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
    ObjectDiffusionModel,
)
from lib.dataset import DexYCBDataset
from lib.diffusion import roi_crop
from lib.utils import setup_logger


def set_requires_grad(model, flag: bool):
    for param in model.parameters():
        param.requires_grad = flag


def load_heatmap_checkpoint(heatmap_predictor, feature_extractor, ckpt_path, logger):
    if not os.path.exists(ckpt_path):
        logger.warning(f"没有找到热图 checkpoint: {ckpt_path}")
        logger.warning("将使用随机初始化的 HeatmapPredictor，这不建议用于正式训练第二阶段。")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in checkpoint:
        heatmap_predictor.load_state_dict(checkpoint["model_state_dict"])
        logger.info("已从 model_state_dict 加载 HeatmapPredictor。")
    elif "heatmap_predictor_state_dict" in checkpoint:
        heatmap_predictor.load_state_dict(checkpoint["heatmap_predictor_state_dict"])
        logger.info("已从 heatmap_predictor_state_dict 加载 HeatmapPredictor。")
    else:
        raise KeyError(
            f"{ckpt_path} 中没有 model_state_dict 或 heatmap_predictor_state_dict。"
        )

    if "feature_extractor_state_dict" in checkpoint:
        feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        logger.info("已加载 feature_extractor_state_dict。")
    else:
        logger.info("checkpoint 中没有 feature_extractor_state_dict，将使用当前 FeatureExtractor 初始化。")


def train_diffusion(
    data_root,
    heatmap_ckpt_path="checkpoints/heatmap_predictor_complete.pth",
    save_dir="checkpoints",
    batch_size=32,
    epochs=3,
    lr=1e-4,
    T=1000,
    pose_dim_hand=96,
    pose_dim_obj=9,
    cond_dim=1024,
    num_workers=0,
    log_interval=50,
    save_interval=1000,
):
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger("train_diffusion.log")
    logger.info(f"Using device: {device}")
    logger.info("开始第二阶段：训练 FeatureEncoder + Diffusion Models")

    dataset_train = DexYCBDataset(
        data_root=data_root,
        split="train",
        transform=transform,
        T=T,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(f"训练样本数: {len(dataset_train)}")
    logger.info(f"每个 epoch 的 batch 数: {len(dataloader_train)}")


    # ===== 模型 =====
    feature_extractor = FeatureExtractor().to(device)
    heatmap_predictor = HeatmapPredictor().to(device)

    hand_encoder = FeatureEncoder(
        input_channels=256,
        num_heatmaps=21,
        output_dim=cond_dim,
        roi_size=32,
    ).to(device)

    obj_encoder = FeatureEncoder(
        input_channels=256,
        num_heatmaps=27,
        output_dim=cond_dim,
        roi_size=32,
    ).to(device)

    hand_model = HandleDiffusionModel(
        pose_dim=pose_dim_hand,
        cond_dim=cond_dim,
        beta_dim=10,
        time_dim=256,
        hidden_dim=512,
    ).to(device)

    object_model = ObjectDiffusionModel(
        cond_dim=cond_dim,
        pose_dim=pose_dim_obj,
        time_dim=256,
        hidden_dim=512,
    ).to(device)

    # ===== 加载阶段1热图权重 =====
    load_heatmap_checkpoint(
        heatmap_predictor=heatmap_predictor,
        feature_extractor=feature_extractor,
        ckpt_path=heatmap_ckpt_path,
        logger=logger,
    )

    # ===== 冻结 FPN + HeatmapPredictor =====
    set_requires_grad(feature_extractor, False)
    set_requires_grad(heatmap_predictor, False)

    feature_extractor.eval()
    heatmap_predictor.eval()

    # ===== 第二阶段训练 Encoder + Diffusion =====
    set_requires_grad(hand_encoder, True)
    set_requires_grad(obj_encoder, True)
    set_requires_grad(hand_model, True)
    set_requires_grad(object_model, True)

    hand_encoder.train()
    obj_encoder.train()
    hand_model.train()
    object_model.train()

    optimizer = optim.Adam(
        list(hand_encoder.parameters())
        + list(obj_encoder.parameters())
        + list(hand_model.parameters())
        + list(object_model.parameters()),
        lr=lr,
    )

    criterion = nn.MSELoss()

    best_avg_loss = float("inf")

    for epoch in range(epochs):
        hand_encoder.train()
        obj_encoder.train()
        hand_model.train()
        object_model.train()

        epoch_hand_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader_train):
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
                _object_mesh_path,
                bbox_hand_norm,
                bbox_obj_norm,
            ) = batch

            theta_t = theta_t.to(device, dtype=torch.float32, non_blocking=True)
            phi_t = phi_t.to(device, dtype=torch.float32, non_blocking=True)
            t = t.to(device, dtype=torch.long, non_blocking=True)

            image = image.to(device, dtype=torch.float32, non_blocking=True)
            eps_hand = eps_hand.to(device, dtype=torch.float32, non_blocking=True)
            eps_obj = eps_obj.to(device, dtype=torch.float32, non_blocking=True)
            betas = betas.to(device, dtype=torch.float32, non_blocking=True)

            bbox_hand_norm = bbox_hand_norm.to(device, dtype=torch.float32, non_blocking=True)
            bbox_obj_norm = bbox_obj_norm.to(device, dtype=torch.float32, non_blocking=True)

            # 冻结模块只负责提供视觉特征和热图条件，不参与反向传播
            with torch.no_grad():
                hand_feat, obj_feat = feature_extractor(image)

                hand_roi = roi_crop(
                    hand_feat,
                    bbox_hand_norm,
                    output_size=32,
                )

                obj_roi = roi_crop(
                    obj_feat,
                    bbox_obj_norm,
                    output_size=32,
                )

                hand_heatmap, obj_heatmap = heatmap_predictor(
                    hand_roi,
                    obj_roi,
                )

            # FeatureEncoder 是第二阶段要训练的，所以不能放进 no_grad
            psi_h = hand_encoder(hand_roi, hand_heatmap)
            psi_o = obj_encoder(obj_roi, obj_heatmap)

            # Diffusion models 也是第二阶段要训练的，所以不能放进 no_grad
            eps_hand_pred = hand_model(theta_t, t, psi_h, betas)
            eps_obj_pred = object_model(phi_t, t, psi_o)

            loss_hand = criterion(eps_hand_pred, eps_hand)
            loss_obj = criterion(eps_obj_pred, eps_obj)
            loss_total = loss_hand + loss_obj

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                list(hand_encoder.parameters())
                + list(obj_encoder.parameters())
                + list(hand_model.parameters())
                + list(object_model.parameters()),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_hand_loss += loss_hand.item()
            epoch_obj_loss += loss_obj.item()
            epoch_total_loss += loss_total.item()

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader_train)}] "
                    f"Hand Diff Loss: {loss_hand.item():.6f}, "
                    f"Obj Diff Loss: {loss_obj.item():.6f}, "
                    f"Total: {loss_total.item():.6f}"
                )

            if batch_idx > 0 and batch_idx % save_interval == 0:
                latest_path = os.path.join(save_dir, "diffusion_latest.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "batch_idx": batch_idx,
                        "hand_encoder_state_dict": hand_encoder.state_dict(),
                        "obj_encoder_state_dict": obj_encoder.state_dict(),
                        "hand_model_state_dict": hand_model.state_dict(),
                        "object_model_state_dict": object_model.state_dict(),
                        "heatmap_predictor_state_dict": heatmap_predictor.state_dict(),
                        "feature_extractor_state_dict": feature_extractor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_hand": loss_hand.item(),
                        "loss_obj": loss_obj.item(),
                        "loss_total": loss_total.item(),
                        "T": T,
                        "cond_dim": cond_dim,
                        "pose_dim_hand": pose_dim_hand,
                        "pose_dim_obj": pose_dim_obj,
                    },
                    latest_path,
                )
                logger.info(f"已保存中间 checkpoint: {latest_path}")

        avg_hand_loss = epoch_hand_loss / len(dataloader_train)
        avg_obj_loss = epoch_obj_loss / len(dataloader_train)
        avg_total_loss = epoch_total_loss / len(dataloader_train)

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] 完成 | "
            f"Avg Hand Diff Loss: {avg_hand_loss:.6f}, "
            f"Avg Obj Diff Loss: {avg_obj_loss:.6f}, "
            f"Avg Total: {avg_total_loss:.6f}"
        )

        epoch_path = os.path.join(save_dir, f"diffusion_epoch_{epoch + 1}.pth")
        checkpoint = {
            "epoch": epoch + 1,
            "hand_encoder_state_dict": hand_encoder.state_dict(),
            "obj_encoder_state_dict": obj_encoder.state_dict(),
            "hand_model_state_dict": hand_model.state_dict(),
            "object_model_state_dict": object_model.state_dict(),
            "heatmap_predictor_state_dict": heatmap_predictor.state_dict(),
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_hand_loss": avg_hand_loss,
            "avg_obj_loss": avg_obj_loss,
            "avg_total_loss": avg_total_loss,
            "T": T,
            "cond_dim": cond_dim,
            "pose_dim_hand": pose_dim_hand,
            "pose_dim_obj": pose_dim_obj,
            "batch_size": batch_size,
            "lr": lr,
        }

        torch.save(checkpoint, epoch_path)
        logger.info(f"已保存 epoch checkpoint: {epoch_path}")

        latest_path = os.path.join(save_dir, "diffusion_latest.pth")
        torch.save(checkpoint, latest_path)
        logger.info(f"已更新 latest checkpoint: {latest_path}")

        if avg_total_loss < best_avg_loss:
            best_avg_loss = avg_total_loss
            best_path = os.path.join(save_dir, "diffusion_best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"已保存 best checkpoint: {best_path}")

    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(
        {
            "hand_encoder_state_dict": hand_encoder.state_dict(),
            "obj_encoder_state_dict": obj_encoder.state_dict(),
            "hand_model_state_dict": hand_model.state_dict(),
            "object_model_state_dict": object_model.state_dict(),
            "heatmap_predictor_state_dict": heatmap_predictor.state_dict(),
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "T": T,
            "cond_dim": cond_dim,
            "pose_dim_hand": pose_dim_hand,
            "pose_dim_obj": pose_dim_obj,
            "best_avg_loss": best_avg_loss,
        },
        final_path,
    )

    logger.info(f"第二阶段训练完成，最终模型已保存到: {final_path}")


if __name__ == "__main__":
    data_root = "/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb"

    train_diffusion(
        data_root=data_root,
        heatmap_ckpt_path="checkpoints/heatmap_predictor_complete.pth",
        save_dir="checkpoints",
        batch_size=32,
        epochs=3,
        lr=1e-4,
        T=1000,
        pose_dim_hand=96,
        pose_dim_obj=9,
        cond_dim=1024,
        num_workers=0,
        log_interval=50,
        save_interval=1000,
    )
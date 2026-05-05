import os
import sys
import math
import argparse
import logging
from typing import Dict, Tuple

sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from lib.config import device, transform
from lib.dataset import DexYCBDataset
from lib.models import (
    FeatureExtractor,
    HeatmapPredictor,
    FeatureEncoder,
)
from lib.diffusion import roi_crop


# ============================================================
# 1. Logger
# ============================================================

def build_logger(log_path: str):
    logger = logging.getLogger("train_score")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# 2. SDE functions
# ============================================================

def ve_marginal_prob(x: torch.Tensor, t: torch.Tensor,
                     sigma_min: float = 0.01,
                     sigma_max: float = 50.0):
    """
    VE SDE marginal probability.

    x: [B, D]
    t: [B, 1], continuous time in (eps, 1)
    return:
        mean: [B, D]
        std:  [B, 1]
    """
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std


def ve_sde(t: torch.Tensor,
           sigma_min: float = 0.01,
           sigma_max: float = 50.0):
    """
    VE SDE:
        drift = 0
        diffusion = sigma(t) * sqrt(2 * log(sigma_max / sigma_min))
    """
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift = torch.zeros_like(t)
    diffusion = sigma * torch.sqrt(
        torch.tensor(
            2.0 * (math.log(sigma_max) - math.log(sigma_min)),
            device=t.device,
            dtype=t.dtype,
        )
    )
    return drift, diffusion


def ve_prior(shape, device, sigma_min: float = 0.01,
             sigma_max: float = 50.0,
             T: float = 1.0):
    """
    VE prior at time T.
    """
    sigma_T = sigma_min * (sigma_max / sigma_min) ** T
    return torch.randn(*shape, device=device) * sigma_T


def vp_marginal_prob(x: torch.Tensor, t: torch.Tensor,
                     beta_0: float = 0.1,
                     beta_1: float = 20.0):
    """
    VP SDE marginal probability.

    x: [B, D]
    t: [B, 1]
    """
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = torch.exp(log_mean_coeff) * x
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    return mean, std


def vp_sde(t: torch.Tensor,
           beta_0: float = 0.1,
           beta_1: float = 20.0):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift = -0.5 * beta_t
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion


def vp_prior(shape, device):
    return torch.randn(*shape, device=device)


def get_sde_functions(sde_mode: str):
    """
    返回:
        marginal_prob_fn
        sde_fn
        prior_fn
        eps
        T
    """
    if sde_mode == "ve":
        eps = 1e-5
        T = 1.0

        def marginal_prob_fn(x, t):
            return ve_marginal_prob(x, t)

        def sde_fn(t):
            return ve_sde(t)

        def prior_fn(shape, device):
            return ve_prior(shape, device=device, T=T)

        return marginal_prob_fn, sde_fn, prior_fn, eps, T

    if sde_mode == "vp":
        eps = 1e-3
        T = 1.0

        def marginal_prob_fn(x, t):
            return vp_marginal_prob(x, t)

        def sde_fn(t):
            return vp_sde(t)

        def prior_fn(shape, device):
            return vp_prior(shape, device=device)

        return marginal_prob_fn, sde_fn, prior_fn, eps, T

    raise ValueError(f"Unsupported sde_mode: {sde_mode}. Use 've' or 'vp'.")


# ============================================================
# 3. Score Denoiser
# ============================================================

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier features for continuous time t.

    输入:
        t: [B]
    输出:
        time embedding: [B, embed_dim]
    """

    def __init__(self, embed_dim: int = 128, scale: float = 30.0):
        super().__init__()
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale)

    def forward(self, t: torch.Tensor):
        x_proj = t[:, None] * self.W[None, :] * 2.0 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreDenoiser(nn.Module):
    """
    通用 score denoiser。

    它学习:
        score(x_t, t, condition) ≈ ∇_x log p_t(x_t | condition)

    训练时输入:
        data["feat"]         条件特征 [B, cond_dim]
        data["sampled_pose"] 带噪姿态 [B, pose_dim]
        data["t"]            连续时间 [B, 1]

    输出:
        estimated_score [B, pose_dim]
    """

    def __init__(
        self,
        pose_dim: int,
        cond_dim: int,
        marginal_prob_fn,
        time_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.pose_dim = pose_dim
        self.cond_dim = cond_dim
        self.out_dim = pose_dim
        self.marginal_prob_fn = marginal_prob_fn

        self.act = nn.ReLU(inplace=True)

        self.time_encoder = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_dim),
            nn.Linear(time_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
        )

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
        )

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            self.act,
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.act,
            zero_module(nn.Linear(hidden_dim, pose_dim)),
        )

    def forward(self, data: Dict[str, torch.Tensor]):
        feat = data["feat"]
        sampled_pose = data["sampled_pose"]
        t = data["t"]

        if t.dim() == 2:
            t_flat = t.squeeze(-1)
        else:
            t_flat = t

        t_feat = self.time_encoder(t_flat)
        pose_feat = self.pose_encoder(sampled_pose)
        cond_feat = self.cond_encoder(feat)

        total_feat = torch.cat([t_feat, pose_feat, cond_feat], dim=-1)

        _, std = self.marginal_prob_fn(sampled_pose, t)
        score = self.net(total_feat)

        # 和 VPHO BaseDenoiser 保持一致：输出除以 std
        score = score / (std + 1e-7)
        return score


def zero_module(module: nn.Module):
    """
    把最后一层初始化为 0。
    官方 denoiser 的 head 也使用 zero_module，这样初始输出更稳定。
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# ============================================================
# 4. Score loss
# ============================================================

def score_matching_loss(
    model: nn.Module,
    feat: torch.Tensor,
    gt_pose: torch.Tensor,
    marginal_prob_fn,
    eps: float,
    repeat_num: int = 1,
):
    """
    Score matching loss.

    gt_pose: [B, D]
    feat:    [B, cond_dim]

    每次:
        1. 随机采连续时间 t
        2. 用 SDE marginal_prob 得到 mean/std
        3. 加噪: x_t = mean + std * z
        4. 模型预测 estimated_score
        5. 监督 target_score = -z / std
    """
    total_loss = 0.0
    B = gt_pose.shape[0]

    for _ in range(repeat_num):
        random_t = torch.rand(B, device=gt_pose.device) * (1.0 - eps) + eps
        random_t = random_t.unsqueeze(-1)

        mu, std = marginal_prob_fn(gt_pose, random_t)
        std = std.view(B, 1)

        z = torch.randn_like(gt_pose)
        perturbed_x = mu + z * std

        data = {
            "feat": feat,
            "sampled_pose": perturbed_x,
            "t": random_t,
        }

        estimated_score = model(data)
        target_score = -z / (std + 1e-7)

        loss_weighting = std ** 2
        loss = torch.mean(
            torch.sum(
                (loss_weighting * (estimated_score - target_score) ** 2).view(B, -1),
                dim=-1,
            )
        )

        total_loss = total_loss + loss

    return total_loss / float(repeat_num)


# ============================================================
# 5. Geometry condition for object
# ============================================================

def heatmap_root_uv_from_bbox(
    hand_heatmap: torch.Tensor,
    bbox_hand_norm: torch.Tensor,
) -> torch.Tensor:
    """
    从 hand heatmap 的 root channel 估计 root 2D 位置。

    hand_heatmap: [B, 21, 64, 64]
    bbox_hand_norm: [B, 4], 归一化到 0~1
    return: [B, 2]
    """
    root_hm = hand_heatmap[:, 0]
    B, H, W = root_hm.shape

    flat_idx = torch.argmax(root_hm.reshape(B, -1), dim=1)
    y = (flat_idx // W).float()
    x = (flat_idx % W).float()

    rx = (x + 0.5) / float(W)
    ry = (y + 0.5) / float(H)

    x1 = bbox_hand_norm[:, 0]
    y1 = bbox_hand_norm[:, 1]
    x2 = bbox_hand_norm[:, 2]
    y2 = bbox_hand_norm[:, 3]

    u = x1 + rx * (x2 - x1)
    v = y1 + ry * (y2 - y1)

    return torch.stack([u, v], dim=-1)


def build_object_geom_condition(
    hand_heatmap: torch.Tensor,
    bbox_hand_norm: torch.Tensor,
    bbox_obj_norm: torch.Tensor,
) -> torch.Tensor:
    """
    构造 object score model 的几何条件，共 12 维。

    root_uv      [2]
    hand_center  [2]
    obj_center   [2]
    center_delta [2]
    hand_size    [2]
    obj_size     [2]
    """
    root_uv = heatmap_root_uv_from_bbox(hand_heatmap, bbox_hand_norm)

    hand_center = 0.5 * (bbox_hand_norm[:, :2] + bbox_hand_norm[:, 2:4])
    obj_center = 0.5 * (bbox_obj_norm[:, :2] + bbox_obj_norm[:, 2:4])

    center_delta = obj_center - hand_center

    hand_size = (bbox_hand_norm[:, 2:4] - bbox_hand_norm[:, :2]).clamp(min=1e-4)
    obj_size = (bbox_obj_norm[:, 2:4] - bbox_obj_norm[:, :2]).clamp(min=1e-4)

    return torch.cat(
        [
            root_uv,
            hand_center,
            obj_center,
            center_delta,
            hand_size,
            obj_size,
        ],
        dim=-1,
    )


class ObjectConditionProjector(nn.Module):
    """
    物体 score model 使用更强条件:
        psi_o + psi_h + geom_cond

    输入:
        psi_o: [B, 1024]
        psi_h: [B, 1024]
        geom_cond: [B, 12]

    输出:
        obj_score_feat: [B, cond_dim]
    """

    def __init__(self, cond_dim=1024, geom_dim=12, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim * 2 + geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, psi_o, psi_h, geom_cond):
        x = torch.cat([psi_o, psi_h, geom_cond], dim=-1)
        return self.net(x)


# ============================================================
# 6. EMA
# ============================================================

class EMA:
    """
    Exponential Moving Average（指数滑动平均）。
    保存 score model 参数的平滑版本，通常采样更稳定。
    """

    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow = []
        self.backup = []
        for p in parameters:
            if p.requires_grad:
                self.shadow.append(p.detach().clone())

    @torch.no_grad()
    def update(self, parameters):
        i = 0
        for p in parameters:
            if not p.requires_grad:
                continue
            self.shadow[i].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            i += 1

    @torch.no_grad()
    def copy_to(self, parameters):
        i = 0
        for p in parameters:
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[i].data)
            i += 1

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ============================================================
# 7. Checkpoint loading
# ============================================================

def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def load_heatmap_checkpoint(
    heatmap_predictor,
    feature_extractor,
    ckpt_path,
    logger,
):
    if not ckpt_path:
        raise ValueError("必须提供 --heatmap_ckpt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 heatmap checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        heatmap_predictor.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info("已从 model_state_dict 加载 HeatmapPredictor。")
    elif "heatmap_predictor_state_dict" in ckpt:
        heatmap_predictor.load_state_dict(ckpt["heatmap_predictor_state_dict"], strict=False)
        logger.info("已从 heatmap_predictor_state_dict 加载 HeatmapPredictor。")
    else:
        raise KeyError("heatmap checkpoint 里没有 model_state_dict 或 heatmap_predictor_state_dict")

    if "feature_extractor_state_dict" in ckpt:
        feature_extractor.load_state_dict(ckpt["feature_extractor_state_dict"], strict=False)
        logger.info("已加载 feature_extractor_state_dict。")
    else:
        logger.warning("heatmap checkpoint 中没有 feature_extractor_state_dict。")


def load_visual_init(
    init_ckpt_path,
    hand_encoder,
    obj_encoder,
    obj_cond_projector,
    logger,
):
    """
    从已有 DDPM / regagg checkpoint 里加载 encoder。
    不加载 hand_model/object_model，因为 DDPM 模型预测 epsilon，
    score model 预测 score，语义不同。
    """
    if not init_ckpt_path:
        logger.info("未提供 --init_ckpt，不加载已有 encoder。")
        return

    if not os.path.exists(init_ckpt_path):
        logger.warning(f"--init_ckpt 不存在: {init_ckpt_path}")
        return

    ckpt = torch.load(init_ckpt_path, map_location=device)
    logger.info(f"尝试从 init checkpoint 加载视觉条件模块: {init_ckpt_path}")

    if "hand_encoder_state_dict" in ckpt:
        hand_encoder.load_state_dict(ckpt["hand_encoder_state_dict"], strict=False)
        logger.info("已加载 hand_encoder_state_dict。")

    if "obj_encoder_state_dict" in ckpt:
        obj_encoder.load_state_dict(ckpt["obj_encoder_state_dict"], strict=False)
        logger.info("已加载 obj_encoder_state_dict。")

    if "obj_cond_projector_state_dict" in ckpt:
        obj_cond_projector.load_state_dict(ckpt["obj_cond_projector_state_dict"], strict=False)
        logger.info("已加载 obj_cond_projector_state_dict。")

    logger.info("注意：不会加载 DDPM hand_model/object_model 到 score model。")


def resume_score_checkpoint(
    resume_path,
    hand_encoder,
    obj_encoder,
    obj_cond_projector,
    score_hand,
    score_obj,
    optimizer,
    ema_hand,
    ema_obj,
    logger,
):
    if not resume_path:
        return 0, float("inf")

    if not os.path.exists(resume_path):
        logger.warning(f"resume checkpoint 不存在: {resume_path}")
        return 0, float("inf")

    ckpt = torch.load(resume_path, map_location=device)
    logger.info(f"恢复 score checkpoint: {resume_path}")

    hand_encoder.load_state_dict(ckpt["hand_encoder_state_dict"], strict=False)
    obj_encoder.load_state_dict(ckpt["obj_encoder_state_dict"], strict=False)
    obj_cond_projector.load_state_dict(ckpt["obj_cond_projector_state_dict"], strict=False)
    score_hand.load_state_dict(ckpt["score_hand_state_dict"], strict=True)
    score_obj.load_state_dict(ckpt["score_obj_state_dict"], strict=True)

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("已恢复 optimizer。")

    if "ema_hand_state_dict" in ckpt:
        ema_hand.load_state_dict(ckpt["ema_hand_state_dict"])
        logger.info("已恢复 ema_hand。")

    if "ema_obj_state_dict" in ckpt:
        ema_obj.load_state_dict(ckpt["ema_obj_state_dict"])
        logger.info("已恢复 ema_obj。")

    start_epoch = int(ckpt.get("epoch", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))

    logger.info(f"恢复完成: start_epoch={start_epoch}, best_loss={best_loss:.6f}")
    return start_epoch, best_loss


def save_score_checkpoint(
    path,
    epoch,
    hand_encoder,
    obj_encoder,
    obj_cond_projector,
    score_hand,
    score_obj,
    heatmap_predictor,
    feature_extractor,
    optimizer,
    ema_hand,
    ema_obj,
    best_loss,
    args,
    avg_hand_loss=None,
    avg_obj_loss=None,
    avg_total_loss=None,
):
    ckpt = {
        "epoch": epoch,
        "hand_encoder_state_dict": hand_encoder.state_dict(),
        "obj_encoder_state_dict": obj_encoder.state_dict(),
        "obj_cond_projector_state_dict": obj_cond_projector.state_dict(),
        "score_hand_state_dict": score_hand.state_dict(),
        "score_obj_state_dict": score_obj.state_dict(),
        "heatmap_predictor_state_dict": heatmap_predictor.state_dict(),
        "feature_extractor_state_dict": feature_extractor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_hand_state_dict": ema_hand.state_dict(),
        "ema_obj_state_dict": ema_obj.state_dict(),
        "best_loss": best_loss,
        "sde_mode": args.sde_mode,
        "pose_dim_hand": args.pose_dim_hand,
        "pose_dim_obj": args.pose_dim_obj,
        "cond_dim": args.cond_dim,
        "score_model_type": "score_sde_v1",
    }

    if avg_hand_loss is not None:
        ckpt["avg_hand_loss"] = float(avg_hand_loss)
    if avg_obj_loss is not None:
        ckpt["avg_obj_loss"] = float(avg_obj_loss)
    if avg_total_loss is not None:
        ckpt["avg_total_loss"] = float(avg_total_loss)

    torch.save(ckpt, path)


# ============================================================
# 8. Training
# ============================================================

def train_score(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logger = build_logger(os.path.join(args.save_dir, "train_score.log"))

    logger.info(f"Using device: {device}")
    logger.info("开始训练 Score Model，不是 DDPM epsilon model。")
    logger.info(f"sde_mode={args.sde_mode}, repeat_num={args.repeat_num}")

    marginal_prob_fn, sde_fn, prior_fn, eps, T_sde = get_sde_functions(args.sde_mode)

    dataset = DexYCBDataset(
        data_root=args.data_root,
        split="train",
        transform=transform,
        T=args.T,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(f"训练样本数: {len(dataset)}")
    logger.info(f"每个 epoch batch 数: {len(dataloader)}")

    feature_extractor = FeatureExtractor().to(device)
    heatmap_predictor = HeatmapPredictor().to(device)

    hand_encoder = FeatureEncoder(
        input_channels=256,
        num_heatmaps=21,
        output_dim=args.cond_dim,
        roi_size=32,
    ).to(device)

    obj_encoder = FeatureEncoder(
        input_channels=256,
        num_heatmaps=27,
        output_dim=args.cond_dim,
        roi_size=32,
    ).to(device)

    obj_cond_projector = ObjectConditionProjector(
        cond_dim=args.cond_dim,
        geom_dim=12,
        hidden_dim=args.cond_dim,
    ).to(device)

    score_hand = ScoreDenoiser(
        pose_dim=args.pose_dim_hand,
        cond_dim=args.cond_dim,
        marginal_prob_fn=marginal_prob_fn,
        hidden_dim=args.hidden_dim,
    ).to(device)

    score_obj = ScoreDenoiser(
        pose_dim=args.pose_dim_obj,
        cond_dim=args.cond_dim,
        marginal_prob_fn=marginal_prob_fn,
        hidden_dim=args.hidden_dim,
    ).to(device)

    load_heatmap_checkpoint(
        heatmap_predictor=heatmap_predictor,
        feature_extractor=feature_extractor,
        ckpt_path=args.heatmap_ckpt,
        logger=logger,
    )

    set_requires_grad(feature_extractor, False)
    set_requires_grad(heatmap_predictor, False)
    feature_extractor.eval()
    heatmap_predictor.eval()

    load_visual_init(
        init_ckpt_path=args.init_ckpt,
        hand_encoder=hand_encoder,
        obj_encoder=obj_encoder,
        obj_cond_projector=obj_cond_projector,
        logger=logger,
    )

    trainable_modules = [
        hand_encoder,
        obj_encoder,
        obj_cond_projector,
        score_hand,
        score_obj,
    ]

    if args.freeze_encoders:
        set_requires_grad(hand_encoder, False)
        set_requires_grad(obj_encoder, False)
        logger.info("已冻结 hand_encoder / obj_encoder，只训练 score models 和 object condition projector。")
        trainable_modules = [
            obj_cond_projector,
            score_hand,
            score_obj,
        ]

    optimizer = optim.AdamW(
        [p for m in trainable_modules for p in m.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ema_hand = EMA(score_hand.parameters(), decay=args.ema_decay)
    ema_obj = EMA(score_obj.parameters(), decay=args.ema_decay)

    start_epoch, best_loss = resume_score_checkpoint(
        resume_path=args.resume,
        hand_encoder=hand_encoder,
        obj_encoder=obj_encoder,
        obj_cond_projector=obj_cond_projector,
        score_hand=score_hand,
        score_obj=score_obj,
        optimizer=optimizer,
        ema_hand=ema_hand,
        ema_obj=ema_obj,
        logger=logger,
    )

    end_epoch = start_epoch + args.epochs
    logger.info(f"训练 epoch 范围: {start_epoch + 1} -> {end_epoch}")

    for epoch in range(start_epoch, end_epoch):
        if args.freeze_encoders:
            hand_encoder.eval()
            obj_encoder.eval()
        else:
            hand_encoder.train()
            obj_encoder.train()

        obj_cond_projector.train()
        score_hand.train()
        score_obj.train()
        feature_extractor.eval()
        heatmap_predictor.eval()

        epoch_hand_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            (
                theta_t_unused,
                phi_t_unused,
                t_unused,
                image,
                theta_0,
                phi_0,
                eps_hand_unused,
                eps_obj_unused,
                betas_unused,
                object_mesh_path_unused,
                bbox_hand_norm,
                bbox_obj_norm,
            ) = batch

            image = image.to(device, dtype=torch.float32, non_blocking=True)
            theta_0 = theta_0.to(device, dtype=torch.float32, non_blocking=True)
            phi_0 = phi_0.to(device, dtype=torch.float32, non_blocking=True)
            bbox_hand_norm = bbox_hand_norm.to(device, dtype=torch.float32, non_blocking=True)
            bbox_obj_norm = bbox_obj_norm.to(device, dtype=torch.float32, non_blocking=True)

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

            if args.freeze_encoders:
                with torch.no_grad():
                    psi_h = hand_encoder(hand_roi, hand_heatmap)
                    psi_o = obj_encoder(obj_roi, obj_heatmap)
            else:
                psi_h = hand_encoder(hand_roi, hand_heatmap)
                psi_o = obj_encoder(obj_roi, obj_heatmap)

            geom_cond = build_object_geom_condition(
                hand_heatmap=hand_heatmap,
                bbox_hand_norm=bbox_hand_norm,
                bbox_obj_norm=bbox_obj_norm,
            )

            obj_score_feat = obj_cond_projector(psi_o, psi_h, geom_cond)

            loss_hand = score_matching_loss(
                model=score_hand,
                feat=psi_h,
                gt_pose=theta_0,
                marginal_prob_fn=marginal_prob_fn,
                eps=eps,
                repeat_num=args.repeat_num,
            )

            loss_obj = score_matching_loss(
                model=score_obj,
                feat=obj_score_feat,
                gt_pose=phi_0,
                marginal_prob_fn=marginal_prob_fn,
                eps=eps,
                repeat_num=args.repeat_num,
            )

            loss_total = args.weight_hand * loss_hand + args.weight_obj * loss_obj

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for m in trainable_modules for p in m.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )

            optimizer.step()

            ema_hand.update(score_hand.parameters())
            ema_obj.update(score_obj.parameters())

            epoch_hand_loss += float(loss_hand.item())
            epoch_obj_loss += float(loss_obj.item())
            epoch_total_loss += float(loss_total.item())

            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{end_epoch}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"score_hand={loss_hand.item():.6f}, "
                    f"score_obj={loss_obj.item():.6f}, "
                    f"total={loss_total.item():.6f}, "
                    f"theta0_mean/std={theta_0.mean().item():.4f}/{theta_0.std().item():.4f}, "
                    f"phi0_mean/std={phi_0.mean().item():.4f}/{phi_0.std().item():.4f}"
                )

            if batch_idx > 0 and batch_idx % args.save_interval == 0:
                latest_path = os.path.join(args.save_dir, "score_latest.pth")
                save_score_checkpoint(
                    path=latest_path,
                    epoch=epoch + 1,
                    hand_encoder=hand_encoder,
                    obj_encoder=obj_encoder,
                    obj_cond_projector=obj_cond_projector,
                    score_hand=score_hand,
                    score_obj=score_obj,
                    heatmap_predictor=heatmap_predictor,
                    feature_extractor=feature_extractor,
                    optimizer=optimizer,
                    ema_hand=ema_hand,
                    ema_obj=ema_obj,
                    best_loss=best_loss,
                    args=args,
                )
                logger.info(f"已保存中间 checkpoint: {latest_path}")

        avg_hand_loss = epoch_hand_loss / max(len(dataloader), 1)
        avg_obj_loss = epoch_obj_loss / max(len(dataloader), 1)
        avg_total_loss = epoch_total_loss / max(len(dataloader), 1)

        logger.info(
            f"Epoch [{epoch + 1}/{end_epoch}] 完成 | "
            f"Avg hand={avg_hand_loss:.6f}, "
            f"Avg obj={avg_obj_loss:.6f}, "
            f"Avg total={avg_total_loss:.6f}"
        )

        epoch_path = os.path.join(args.save_dir, f"score_epoch_{epoch + 1}.pth")
        latest_path = os.path.join(args.save_dir, "score_latest.pth")

        save_score_checkpoint(
            path=epoch_path,
            epoch=epoch + 1,
            hand_encoder=hand_encoder,
            obj_encoder=obj_encoder,
            obj_cond_projector=obj_cond_projector,
            score_hand=score_hand,
            score_obj=score_obj,
            heatmap_predictor=heatmap_predictor,
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            ema_hand=ema_hand,
            ema_obj=ema_obj,
            best_loss=best_loss,
            args=args,
            avg_hand_loss=avg_hand_loss,
            avg_obj_loss=avg_obj_loss,
            avg_total_loss=avg_total_loss,
        )

        save_score_checkpoint(
            path=latest_path,
            epoch=epoch + 1,
            hand_encoder=hand_encoder,
            obj_encoder=obj_encoder,
            obj_cond_projector=obj_cond_projector,
            score_hand=score_hand,
            score_obj=score_obj,
            heatmap_predictor=heatmap_predictor,
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            ema_hand=ema_hand,
            ema_obj=ema_obj,
            best_loss=best_loss,
            args=args,
            avg_hand_loss=avg_hand_loss,
            avg_obj_loss=avg_obj_loss,
            avg_total_loss=avg_total_loss,
        )

        logger.info(f"已保存 epoch checkpoint: {epoch_path}")
        logger.info(f"已更新 latest checkpoint: {latest_path}")

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_path = os.path.join(args.save_dir, "score_best.pth")

            save_score_checkpoint(
                path=best_path,
                epoch=epoch + 1,
                hand_encoder=hand_encoder,
                obj_encoder=obj_encoder,
                obj_cond_projector=obj_cond_projector,
                score_hand=score_hand,
                score_obj=score_obj,
                heatmap_predictor=heatmap_predictor,
                feature_extractor=feature_extractor,
                optimizer=optimizer,
                ema_hand=ema_hand,
                ema_obj=ema_obj,
                best_loss=best_loss,
                args=args,
                avg_hand_loss=avg_hand_loss,
                avg_obj_loss=avg_obj_loss,
                avg_total_loss=avg_total_loss,
            )

            logger.info(f"已保存 best checkpoint: {best_path}")

    logger.info("Score model 训练完成。")


# ============================================================
# 9. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("Train score-based SDE models for hand/object pose")

    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb",
    )

    parser.add_argument(
        "--heatmap_ckpt",
        type=str,
        default="checkpoints/heatmap_epoch_1.pth",
        help="第一阶段 heatmap checkpoint",
    )

    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="可选：从已有 diffusion/regagg checkpoint 加载 hand_encoder / obj_encoder",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="可选：恢复 score checkpoint，例如 checkpoints_score/score_latest.pth",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_score",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument(
        "--sde_mode",
        type=str,
        default="ve",
        choices=["ve", "vp"],
        help="官方常用 score SDE 思路；先建议 ve",
    )

    parser.add_argument(
        "--repeat_num",
        type=int,
        default=1,
        help="每个 batch 重复采样多少个 random_t。显存够可以设为 2 或 4。",
    )

    parser.add_argument("--T", type=int, default=1000, help="Dataset 兼容参数；score 训练本身不用离散 T")
    parser.add_argument("--cond_dim", type=int, default=1024)
    parser.add_argument("--pose_dim_hand", type=int, default=96)
    parser.add_argument("--pose_dim_obj", type=int, default=9)
    parser.add_argument("--hidden_dim", type=int, default=512)

    parser.add_argument("--weight_hand", type=float, default=1.0)
    parser.add_argument("--weight_obj", type=float, default=1.0)

    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)

    parser.add_argument(
        "--freeze_encoders",
        action="store_true",
        help="冻结 hand_encoder / obj_encoder，只训练 score denoiser。默认不冻结。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_score(args)
import torch
from torchvision.ops import roi_align

def precompute_diffusion_coeffs(T=1000, beta_start=1e-4, beta_end=0.02, device=None):
    """ 预计算高斯扩散过程的系数 (betas, alphas, alpha_bars) """
    beta = torch.linspace(beta_start, beta_end, T, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

def reverse_diffusion(model, psi, T, device, alpha, alpha_bar, pose_dim, beta=None):
    """
    反向扩散去噪过程 (Reverse Diffusion Process)：
    从纯噪声开始，借助条件特征 psi，迭代去噪得到最终的姿态表示。

    Args:
        model: 扩散去噪网络模型（手部或物体）
        psi: [B, cond_dim] 条件特征向量
        T: int, 扩散总步数
        device: 运行设备
        alpha: [T] 扩散系数alpha
        alpha_bar: [T] 累积的alpha_bar
        pose_dim: int, 姿态维度（手部96，物体9）
        beta: [B, 10] （可选）手部形状参数，用于手部模型

    Returns:
        theta_t: [B, pose_dim] 最终生成的姿态参数
    """
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    B = psi.shape[0]
    theta_t = torch.randn(B, pose_dim).to(device)

    for t in reversed(range(1, T + 1)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model(theta_t, t_tensor, psi, beta)
        alpha_t = alpha[t - 1]
        alpha_bar_t = alpha_bar[t - 1]
        beta_t = 1 - alpha_t

        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        theta_det = coeff1 * (theta_t - coeff2 * eps_pred)

        if t > 1:
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn_like(theta_t)
            theta_t = theta_det + sigma_t * z
        else:
            theta_t = theta_det

    return theta_t

def generate_hand_candidates(model, psi, beta, T, device, alpha, alpha_bar, pose_dim, num_candidates=100):
    candidates = []
    for _ in range(num_candidates):
        theta = reverse_diffusion(model, psi, T, device, alpha, alpha_bar, pose_dim, beta)
        candidates.append(theta)
    return torch.cat(candidates, dim=0)

def generate_object_candidates(model, psi, T, device, alpha, alpha_bar, pose_dim=9, num_candidates=100):
    model.eval()
    candidates = []

    for _ in range(num_candidates):
        phi_t = torch.randn(1, pose_dim).to(device)
        for t_step in reversed(range(1, T + 1)):
            t_tensor = torch.full((1,), t_step, device=device, dtype=torch.long)
            eps_pred = model(phi_t, t_tensor, psi)
            alpha_t = alpha[t_step - 1]
            alpha_bar_t = alpha_bar[t_step - 1]
            beta_t = 1 - alpha_t

            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            phi_det = coeff1 * (phi_t - coeff2 * eps_pred)

            if t_step > 1:
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(phi_t)
                phi_t = phi_det + sigma_t * z
            else:
                phi_t = phi_det
        candidates.append(phi_t)

    return torch.cat(candidates, dim=0)

def roi_crop(feat, bboxes, output_size=32):
    # feat: [B, 256, 64, 64]
    # bboxes: [B, 4]，归一化坐标 [0,1]
    B = feat.shape[0]
    spatial_scale = 64.0 / 256.0   # 特征图是原图 1/4
    # 转为 roi_align 需要的格式 [batch_idx, x1, y1, x2, y2]（像素坐标）
    b_idx = torch.arange(B, device=feat.device).float()
    bboxes_pixel = bboxes * 256.0
    rois = torch.cat([b_idx[:, None], bboxes_pixel], dim=1)
    return roi_align(feat, rois, output_size=(output_size, output_size),
                     spatial_scale=spatial_scale, aligned=True)
@torch.no_grad()
def generate_object_candidates_handroot(
    model,
    psi_o,
    psi_h,
    geom_cond,
    T,
    device,
    alpha,
    alpha_bar,
    pose_dim=9,
    num_candidates=100,
):
    """Generate object candidates for ObjectDiffusionModelWithHandRoot."""
    model.eval()
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    candidates = []
    B = psi_o.shape[0]

    for _ in range(num_candidates):
        phi_t = torch.randn(B, pose_dim, device=device)
        for t_step in reversed(range(1, T + 1)):
            t_tensor = torch.full((B,), t_step, device=device, dtype=torch.long)
            eps_pred = model(phi_t, t_tensor, psi_o, psi_h, geom_cond)
            alpha_t = alpha[t_step - 1]
            alpha_bar_t = alpha_bar[t_step - 1]
            beta_t = 1 - alpha_t

            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            phi_det = coeff1 * (phi_t - coeff2 * eps_pred)

            if t_step > 1:
                phi_t = phi_det + torch.sqrt(beta_t) * torch.randn_like(phi_t)
            else:
                phi_t = phi_det
        candidates.append(phi_t.detach())

    # [num_candidates, B, D] -> [B, num_candidates, D]
    return torch.stack(candidates, dim=1)

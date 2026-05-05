import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

def _debug_tensor_summary(name: str, x, max_items: int = 12):
    """打印张量/数组的简要统计，便于推理阶段排查单位和数值是否异常。"""
    if torch.is_tensor(x):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    flat = x_np.reshape(-1)
    finite_mask = np.isfinite(flat)
    finite_ratio = float(finite_mask.mean()) if flat.size > 0 else 1.0
    if flat.size == 0:
        print(f"[{name}] empty")
        return

    finite_vals = flat[finite_mask]
    if finite_vals.size > 0:
        vmin = float(finite_vals.min())
        vmax = float(finite_vals.max())
        vmean = float(finite_vals.mean())
    else:
        vmin = vmax = vmean = float('nan')

    preview = flat[:max_items]
    print(f"[{name}] shape={x_np.shape}, finite_ratio={finite_ratio:.3f}, min={vmin:.6f}, max={vmax:.6f}, mean={vmean:.6f}")
    print(f"[{name}] preview={np.array2string(preview, precision=5, separator=', ')}")


def _debug_pose_block(tag: str, pose, is_hand: bool = True):
    """打印手/物体候选的关键片段：旋转、平移、范围。"""
    if torch.is_tensor(pose):
        pose_np = pose.detach().cpu().numpy().reshape(-1)
    else:
        pose_np = np.asarray(pose).reshape(-1)

    print(f"\n--- {tag} ---")
    _debug_tensor_summary(f"{tag}.pose", pose_np)
    if is_hand:
        print(f"[{tag}] root/first 6D: {pose_np[:6]}")
    else:
        print(f"[{tag}] r6d: {pose_np[:6]}")
        print(f"[{tag}] trans(m): {pose_np[6:9]}")

def setup_logger(log_file_path='training.log'):
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def plot_loss_curve(losses: list, save_path: str = "train_loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, linestyle='-', color='b', label='Train Loss')
    plt.title("Training Loss Over Batches")
    plt.xlabel("Batch Iterations")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


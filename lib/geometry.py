import torch
import numpy as np
def project_3d_to_2d(joints_3d, intrinsic_matrix):
    """
    将3D关节投影到2D图像平面
    """
    # 统一转换为 float32
    joints_3d = joints_3d.to(torch.float32)
    # 确保在同一设备且数据类型一致
    intrinsic_matrix = intrinsic_matrix.to(device=joints_3d.device, dtype=torch.float32)

    # 直接用内参矩阵乘以 3D 点的转置
    # joints_3d: [N, 3] → 转置 → [3, N] → 内参 [3,3] @ [3,N] → [3,N]
    points_2d_hom = intrinsic_matrix @ joints_3d.T

    # 归一化并转置回 [N, 2]
    joints_2d = (points_2d_hom[:2, :] / (points_2d_hom[2:3, :] + 1e-8)).T

    return joints_2d

def compute_object_keypoints_3d(object_mesh):
    """
    计算物体的27个3D关键点（物体坐标系）
    包括：8个角点 + 12个边中点 + 6个面中点 + 1个中心点 = 27
    """
    # 从顶点判断单位
    verts = np.asarray(object_mesh.vertices)
    vert_range = verts.max() - verts.min()
    if vert_range > 10:  # 毫米
        scale = 1.0 / 1000.0
    else:  # 米
        scale = 1.0

    bbox = object_mesh.get_axis_aligned_bounding_box()
    min_p = bbox.get_min_bound() * scale
    max_p = bbox.get_max_bound() * scale
    center = (min_p + max_p) / 2.0

    # 8个角点
    corners = np.array([
        [min_p[0], min_p[1], min_p[2]],
        [min_p[0], min_p[1], max_p[2]],
        [min_p[0], max_p[1], min_p[2]],
        [min_p[0], max_p[1], max_p[2]],
        [max_p[0], min_p[1], min_p[2]],
        [max_p[0], min_p[1], max_p[2]],
        [max_p[0], max_p[1], min_p[2]],
        [max_p[0], max_p[1], max_p[2]],
    ], dtype=np.float32)

    # 12个边中点（每条棱的中点）
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5),
        (2, 3), (2, 6), (3, 7), (4, 5), (4, 6),
        (5, 7), (6, 7)
    ]
    edge_midpoints = [(corners[i] + corners[j]) / 2.0 for i, j in edges]
    edge_midpoints = np.array(edge_midpoints, dtype=np.float32)

    # 6个面中点
    face_centers = np.array([
        (corners[0] + corners[1] + corners[2] + corners[3]) / 4.0,  # 前
        (corners[4] + corners[5] + corners[6] + corners[7]) / 4.0,  # 后
        (corners[0] + corners[1] + corners[4] + corners[5]) / 4.0,  # 右
        (corners[2] + corners[3] + corners[6] + corners[7]) / 4.0,  # 左
        (corners[0] + corners[2] + corners[4] + corners[6]) / 4.0,  # 下
        (corners[1] + corners[3] + corners[5] + corners[7]) / 4.0,  # 上
    ], dtype=np.float32)

    # 中心点
    center = center.reshape(1, 3)

    # 拼接：8+12+6+1 = 27
    keypoints = np.vstack([corners, edge_midpoints, face_centers, center])
    assert keypoints.shape == (27, 3)
    return keypoints

def get_hand_bbox_from_joints(joints_2d, image_size=(256, 256), expand_ratio=1.2):
    """从2D关节点估计手部bbox"""
    x_min, y_min = joints_2d.min(axis=0)
    x_max, y_max = joints_2d.max(axis=0)

    w, h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    # 扩展bbox并确保正方形
    size = max(w, h) * expand_ratio
    size = min(size, image_size[0])

    x1 = max(0, cx - size / 2)
    y1 = max(0, cy - size / 2)
    x2 = min(image_size[0], cx + size / 2)
    y2 = min(image_size[1], cy + size / 2)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def get_object_bbox_from_2d_points(points_2d, image_size=(256, 256), expand_ratio=1.3):
    """从物体2D投影点估计物体bbox"""
    valid = (points_2d[:, 0] > 0) & (points_2d[:, 1] > 0)
    if valid.sum() < 4:
        return np.array([64, 64, 192, 192], dtype=np.float32)

    valid_points = points_2d[valid]
    x_min, y_min = valid_points.min(axis=0)
    x_max, y_max = valid_points.max(axis=0)

    w, h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

    size = max(w, h) * expand_ratio
    size = min(size, image_size[0])

    x1 = max(0, cx - size / 2)
    y1 = max(0, cy - size / 2)
    x2 = min(image_size[0], cx + size / 2)
    y2 = min(image_size[1], cy + size / 2)

    return np.array([x1, y1, x2, y2], dtype=np.float32)

def generate_roi_heatmap_from_joints(joints_2d, bbox, heatmap_size=64, sigma=2.0):
    """
    根据 ROI/bbox 内坐标生成 heatmap。

    Args:
        joints_2d: np.ndarray, shape [J, 2]
            关键点坐标，必须和 bbox 在同一个图像坐标系下。
            建议统一使用 256x256 图像坐标。

        bbox: np.ndarray, shape [4]
            [x1, y1, x2, y2]，同样是 256x256 图像坐标。

        heatmap_size: int
            输出 heatmap 尺寸，默认 64。

        sigma: float
            高斯核标准差。

    Returns:
        heatmap: np.ndarray, shape [J, 64, 64]
    """
    joints_2d = np.asarray(joints_2d, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32)

    J = joints_2d.shape[0]
    heatmap = np.zeros((J, heatmap_size, heatmap_size), dtype=np.float32)

    x1, y1, x2, y2 = bbox
    bw = max(float(x2 - x1), 1e-6)
    bh = max(float(y2 - y1), 1e-6)

    yy, xx = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
        indexing="ij"
    )

    for j in range(J):
        x, y = joints_2d[j]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        # 从图像坐标变成 ROI 内 heatmap 坐标
        x_hm = (x - x1) / bw * (heatmap_size - 1)
        y_hm = (y - y1) / bh * (heatmap_size - 1)

        # 如果关键点不在 ROI 内，可以跳过
        if x_hm < 0 or x_hm >= heatmap_size or y_hm < 0 or y_hm >= heatmap_size:
            continue

        heatmap[j] = np.exp(-((xx - x_hm) ** 2 + (yy - y_hm) ** 2) / (2 * sigma ** 2))

    return heatmap
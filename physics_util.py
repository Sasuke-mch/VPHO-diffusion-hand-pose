from hand_obj_model import (
    mano_layer_left, mano_layer_right,
    DexYCBDataset, transform,
    pose_96_to_48, pca_to_axis_angle,
    rotation_6d_to_matrix, rotation_matrix_to_6d
)
import os
import yaml
import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
import copy
import hashlib

YCB_CLASSES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box', 4: '005_tomato_soup_can',
    5: '006_mustard_bottle', 6: '007_tuna_fish_can', 7: '008_pudding_box', 8: '009_gelatin_box',
    9: '010_potted_meat_can', 10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block', 17: '037_scissors',
    18: '040_large_marker', 19: '051_large_clamp', 20: '052_extra_large_clamp', 21: '061_foam_brick'
}

MESH_CACHE_MAX = 8
SCENE_CACHE_MAX = 32

mesh_cache = OrderedDict()
scene_cache = OrderedDict()


def _lru_get(cache: OrderedDict, key):
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]

def _lru_put(cache: OrderedDict, key, value, max_size: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)

def _pose_key_from_transform(T: np.ndarray, decimals: int = 5) -> str:
    # 位姿量化，避免浮点微小抖动导致缓存失效
    Tq = np.round(T.astype(np.float64), decimals=decimals)
    return hashlib.md5(Tq.tobytes()).hexdigest()

def load_mesh_cached(obj_mesh_path: str):
    m = _lru_get(mesh_cache, obj_mesh_path)
    if m is not None:
        return m
    m = o3d.io.read_triangle_mesh(obj_mesh_path)
    _lru_put(mesh_cache, obj_mesh_path, m, MESH_CACHE_MAX)
    return m

def get_scene_cached(obj_mesh_path: str, transform_mat: np.ndarray):
    pkey = _pose_key_from_transform(transform_mat)
    key = (obj_mesh_path, pkey)
    cached = _lru_get(scene_cache, key)
    if cached is not None:
        return cached  # (scene, obj_center_np)

    # 从 mesh cache 拿 base mesh，并拷贝后做位姿变换，避免污染缓存
    base_mesh = load_mesh_cached(obj_mesh_path)
    mesh_inst = copy.deepcopy(base_mesh)
    mesh_inst.transform(transform_mat)

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_inst)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    obj_center_np = np.asarray(mesh_inst.get_center(), dtype=np.float32)
    value = (scene, obj_center_np)
    _lru_put(scene_cache, key, value, SCENE_CACHE_MAX)
    return value

def select_anchor_points(mano_faces, mano_vertices, num_anchors=32):
    """
    从 MANO 模型的面中随机选择 num_anchors 个三角形，返回锚点（重心）及其所属三角形的顶点索引和权重。
    返回：anchor_triangles (N,3), anchor_weights (N,3), anchor_positions (N,3)
    """
    np.random.seed(42)
    selected = np.random.choice(len(mano_faces), num_anchors, replace=False)
    triangles = mano_faces[selected]          # (N,3) 顶点索引
    verts = mano_vertices[triangles]          # (N,3,3)
    weights = np.ones((num_anchors,3)) / 3.0  # 重心权重
    anchor_pos = verts.mean(axis=1)           # 锚点坐标
    return triangles, weights, anchor_pos

def build_friction_cone_basis(Nv=12, mu=1.0):
    """ 生成摩擦锥的基向量 v_j，公式 (2) """
    angles = 2 * np.pi * np.arange(Nv) / Nv
    v = np.stack([mu * np.sin(angles), mu * np.cos(angles), np.ones(Nv)], axis=1)
    return torch.tensor(v, dtype=torch.float32)  # (Nv, 3)

def local_to_global_force(F_local, triangle_verts):
    """
    triangle_verts: (3,3) 三个顶点坐标
    F_local: (3,) 局部力
    """
    p1, p2, p3 = triangle_verts
    x = F.normalize(p2 - p1, dim=-1)
    z = F.normalize(torch.cross(p2 - p1, p3 - p1, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=0)  # (3,3)
    return R @ F_local

def compute_physics_losses(forces_global, anchor_positions, object_center, gravity, contact_distances):
    """
    forces_global: (N,3) N个锚点的全局力
    anchor_positions: (N,3) 锚点全局坐标
    object_center: (3,) 物体质心
    gravity: (3,) 重力向量
    contact_distances: (N,) 每个锚点到物体表面的距离
    """
    # 力平衡 (5)
    loss_force = torch.norm(forces_global.sum(dim=0) + gravity) ** 2

    # 力矩平衡 (6)（力矩臂 = anchor_position - object_center）
    r = anchor_positions - object_center
    torques = torch.cross(forces_global, r, dim=-1)
    loss_torque = torch.norm(torques.sum(dim=0)) ** 2

    # 接触-力关系 (7)
    loss_contact = (torch.norm(forces_global, dim=-1) * torch.abs(contact_distances)).sum()

    return loss_force, loss_torque, loss_contact

class ForcePredictionModule(nn.Module):
    """ 输入手部特征、物体特征、重力方向，输出力参数 """
    def __init__(self, input_dim=2048, hidden_dim=512, num_anchors=32, Nv=12):
        super().__init__()
        self.num_anchors = num_anchors
        self.Nv = Nv
        self.friction_basis = build_friction_cone_basis(Nv)  # 注册为 buffer
        self.register_buffer('v_basis', self.friction_basis)

        # 特征编码
        self.encoder = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 分支：权重矩阵 w
        self.fc_w = nn.Linear(hidden_dim, num_anchors * Nv)
        # 分支：缩放因子 s
        self.fc_s = nn.Linear(hidden_dim, num_anchors)
        # 分支：物体质心偏移（其实可以固定为物体中心）
        self.fc_center = nn.Linear(hidden_dim, 3)

    def forward(self, psi_h, psi_o, gravity):
        """
        psi_h: (B, 2048) 手部特征向量
        psi_o: (B, 2048) 物体特征向量
        gravity: (B, 3) 重力向量（训练时朝下，推理时相机y轴）
        """
        B = psi_h.shape[0]
        x = torch.cat([psi_h, psi_o], dim=-1)
        feat = self.encoder(x)

        # 权重 (B, 32, 12)
        w_logits = self.fc_w(feat).view(B, self.num_anchors, self.Nv)
        w = F.softmax(w_logits, dim=-1)          # 每锚点权重和为1

        # 缩放 (B, 32)
        s = F.softplus(self.fc_s(feat))          # 保证非负

        # 物体质心
        center_offset = self.fc_center(feat)     # (B,3)

        # 计算局部力
        local_forces = s.unsqueeze(-1) * (w @ self.v_basis)  # (B,32,3)

        return local_forces, w, s, center_offset

# 伪力标签生成函数
def generate_pseudo_force_labels(
    mano_vertices, mano_faces,
    anchor_triangles, anchor_weights,
    gravity, obj_center_np,  obj_verts_tensor, device='cuda',
    stage1_steps=80, stage2_steps=400
):
    """
    对单个样本优化伪力标签。
    mano_vertices: torch.Tensor (N_verts, 3) 手部顶点（全局坐标系）
    mano_faces: np.array (N_faces, 3) 手部三角面索引
    obj_mesh: open3d TriangleMesh，物体网格（已在全局坐标系，米单位）
    anchor_triangles: (32,3) 预选的三角形顶点索引
    anchor_weights: (32,3) 重心权重
    gravity: (3,) 重力向量
    """
    Nv = 12
    mu = 1.0
    friction_basis = build_friction_cone_basis(Nv, mu).to(device)

    # 初始化优化变量
    w_logits = torch.zeros(32, Nv, device=device, requires_grad=True)
    s_raw = torch.full((32,), 0.1, device=device, requires_grad=True)

    # 计算锚点全局位置
    tri_verts = mano_vertices[anchor_triangles]          # (32,3,3)
    anchor_pos = (anchor_weights.unsqueeze(-1) * tri_verts).sum(dim=1)  # (32,3)

    # 不再调用 Open3D，直接计算 32 个锚点到物体数千个顶点的最小距离
    def compute_approx_dist_gpu(points, mesh_verts):
        # points: [32, 3], mesh_verts: [V, 3]
        # cdist 计算两两点之间的欧式距离 [32, V]
        dists = torch.cdist(points.unsqueeze(0), mesh_verts.unsqueeze(0))
        # 取每个锚点到物体的最近距离 [32]
        min_dists, _ = torch.min(dists.squeeze(0), dim=1)
        return min_dists

    # 使用绝对距离，保证表面内外的距离偏差都被视作大于0的标量
    contact_dist = compute_approx_dist_gpu(anchor_pos, obj_verts_tensor)
    obj_center = torch.tensor(obj_center_np, dtype=torch.float32, device=device)

    early_stop_patience = 30
    early_stop_min_delta = 1e-5
    early_stop_warmup = 80

    # 第一阶段：只优化 w，使合力平衡重力
    optimizer1 = torch.optim.Adam([w_logits], lr=1e-2)
    for step in range(stage1_steps):
        optimizer1.zero_grad()
        w = F.softmax(w_logits, dim=-1)
        s = F.softplus(s_raw)  # 使用 softplus梯度更平滑
        local_F = (s.unsqueeze(-1) * (w @ friction_basis))  # (32,3)

        glob_F = []
        for k in range(32):
            tri_v = tri_verts[k]
            Fk = local_to_global_force(local_F[k], tri_v)
            glob_F.append(Fk)
        glob_F = torch.stack(glob_F)

        net_force = glob_F.sum(dim=0) + gravity
        loss =  0.01 * (net_force ** 2).sum()
        loss.backward()
        optimizer1.step()

    best_loss = float('inf')
    bad_count = 0
    # 第二阶段：联合优化 w, s，加入力矩和接触约束
    optimizer2 = torch.optim.Adam([w_logits, s_raw], lr=1e-3)
    for step in range(stage2_steps):
        optimizer2.zero_grad()
        w = F.softmax(w_logits, dim=-1)
        s = F.softplus(s_raw)
        local_F = (s.unsqueeze(-1) * (w @ friction_basis))

        glob_F = []
        for k in range(32):
            tri_v = tri_verts[k]
            Fk = local_to_global_force(local_F[k], tri_v)
            glob_F.append(Fk)
        glob_F = torch.stack(glob_F)

        # 力平衡
        loss_force = torch.norm(glob_F.sum(dim=0) + gravity) ** 2

        # 力矩平衡（相对物体质心）
        obj_center = torch.tensor(obj_center_np, dtype=torch.float32, device=device)
        r = anchor_pos - obj_center
        torques = torch.cross(glob_F, r, dim=-1)
        loss_torque = torch.norm(torques.sum(dim=0)) ** 2

        # 接触-力约束：距离近的锚点力大，远的力小
        contact_prob = torch.sigmoid(-100 * (contact_dist - 0.02))
        loss_contact = ((1 - contact_prob) * torch.norm(glob_F, dim=-1)).mean()

        # 建议的平衡权重
        w_force = 0.01
        w_torque = 30.0
        w_contact = 0.1

        loss = w_force * loss_force + w_torque * loss_torque + w_contact * loss_contact
        loss.backward()
        optimizer2.step()


        # early stop
        cur = float(loss.item())
        if step >= early_stop_warmup:
            if best_loss - cur > early_stop_min_delta:
                best_loss = cur
                bad_count = 0
            else:
                bad_count += 1
                if bad_count >= early_stop_patience:
                    break
        else:
            if cur < best_loss:
                best_loss = cur

    with torch.no_grad():
        w_opt = F.softmax(w_logits, dim=-1)
        s_opt = F.softplus(s_raw)
        local_F_opt = (s_opt.unsqueeze(-1) * (w_opt @ friction_basis))

        glob_F_opt = []
        for k in range(32):
            Fk = local_to_global_force(local_F_opt[k], tri_verts[k])
            glob_F_opt.append(Fk)
        glob_F_opt = torch.stack(glob_F_opt)

        final_force = torch.norm(glob_F_opt.sum(dim=0) + gravity).item()
        final_torque = torch.norm(torch.cross(glob_F_opt, (anchor_pos - obj_center), dim=-1).sum(dim=0)).item()
        mean_contact = contact_dist.mean().item()

    return w_opt.cpu(), s_opt.cpu(), local_F_opt.cpu(), final_force, final_torque, mean_contact

def _safe_int(x):
    """
    尽量把 numpy/python 标量转 int，失败返回 None
    """
    try:
        return int(x)
    except Exception:
        return None


def _resolve_cam_index_from_meta(label_path, camera_serial):
    """
    根据 label_path 回溯到 sequence 的 meta.yml，尝试把相机 serial 映射到 pose_y 的下标。
    返回 int 或 None（映射失败）。
    """
    cam_dir = os.path.dirname(label_path)
    seq_dir = os.path.dirname(cam_dir)
    meta_path = os.path.join(seq_dir, "meta.yml")

    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)
    except Exception:
        return None

    candidate_keys = ["cam_serials", "camera_serials", "serials", "cams", "cameras"]
    serial_list = None
    for k in candidate_keys:
        if isinstance(meta, dict) and k in meta and meta[k] is not None:
            serial_list = meta[k]
            break

    if serial_list is None:
        return None

    # 统一成 int 列表
    serial_norm = []
    for s in serial_list:
        si = _safe_int(s)
        if si is not None:
            serial_norm.append(si)
        else:
            # 如果不是纯数字，保留原字符串
            serial_norm.append(str(s))

    cam_i = _safe_int(camera_serial)
    if cam_i is not None and cam_i in serial_norm:
        return serial_norm.index(cam_i)

    # 字符串兜底
    cam_s = str(camera_serial)
    if cam_s in serial_norm:
        return serial_norm.index(cam_s)

    return None

# 数据集加载与处理
class PseudoForceDataset(DexYCBDataset):
    """
    在原数据集基础上，增加手性信息读取和物体模型路径回传。
    """
    def __init__(self, data_root, split='train'):
        super().__init__(data_root, split, transform, T=1000)

    def __getitem__(self, idx):
        # self.samples 元素结构来自父类 _load_samples:
        # (rgb_path, label_path, betas, camera_id)
        image_path, label_path, betas, camera_id = self.samples[idx]

        # 1) 手姿态：复用父类工具函数
        theta_0 = self._load_pose_from_label(label_path)  # np[96]

        # 2) 物体姿态：自己读，避免把 camera serial 当数组下标
        data = np.load(label_path, allow_pickle=True)
        pose_y = data["pose_y"]  # 通常 [Nc,4,4]

        # 优先用 meta.yml 做 serial->index 映射；失败则回退0
        cam_idx = _resolve_cam_index_from_meta(label_path, camera_id)
        if cam_idx is None:
            cam_idx = 0

        # 边界保护
        if cam_idx < 0 or cam_idx >= pose_y.shape[0]:
            cam_idx = 0

        transform_mat = pose_y[cam_idx]
        R = transform_mat[:3, :3]
        trans = transform_mat[:3, 3]
        R_tensor = torch.tensor(R, dtype=torch.float32)
        r6d = rotation_matrix_to_6d(R_tensor).numpy()
        phi_0 = np.concatenate([r6d, trans.astype(np.float32)]).astype(np.float32)

        # 3) 物体模型路径：和 hand_obj_model.py 一样，从 meta.yml 按 grasp 物体解析
        seq_dir = os.path.dirname(os.path.dirname(label_path))
        meta_path = os.path.join(seq_dir, "meta.yml")
        meta = None
        object_name = "002_master_chef_can"  # fallback
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f) or {}
                ycb_ids = meta.get("ycb_ids", [])
                ycb_grasp_ind = meta.get("ycb_grasp_ind", 0)
                if 0 <= ycb_grasp_ind < len(ycb_ids):
                    object_name = YCB_CLASSES.get(ycb_ids[ycb_grasp_ind], object_name)
            except Exception:
                meta = None
        dexycb_path = "/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb/models"
        obj_mesh_path = os.path.join(dexycb_path, str(object_name), "textured.obj")

        # 4) 手性读取
        is_left = False
        if meta is None and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)
        if meta and "mano_sides" in meta and len(meta["mano_sides"]) > 0:
            is_left = (meta["mano_sides"][0] == "left")

        # 返回 torch
        theta_0 = torch.tensor(theta_0, dtype=torch.float32)
        phi_0 = torch.tensor(phi_0, dtype=torch.float32)
        betas = torch.tensor(betas, dtype=torch.float32)

        return theta_0, phi_0, betas, obj_mesh_path, is_left, idx

def get_gravity(device, axis='y', sign=-1.0, g=9.81):
    v = torch.zeros(3, device=device)
    ax = {'x':0, 'y':1, 'z':2}[axis]
    v[ax] = sign * g
    return v


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #加载锚点数据
    anchor_left = np.load('anchor_data_left.npz')
    anchor_right = np.load('anchor_data_right.npz')

    ANCHOR_TRIANGLES_LEFT = torch.tensor(anchor_left['triangles'], dtype=torch.long)
    ANCHOR_WEIGHTS_LEFT = torch.tensor(anchor_left['weights'], dtype=torch.float32)
    ANCHOR_TRIANGLES_RIGHT = torch.tensor(anchor_right['triangles'], dtype=torch.long)
    ANCHOR_WEIGHTS_RIGHT = torch.tensor(anchor_right['weights'], dtype=torch.float32)

    data_root = "/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb"
    save_dir = "pseudo_force_labels"
    os.makedirs(save_dir, exist_ok=True)

    dataset = PseudoForceDataset(data_root, split='train')
    gravity = get_gravity(device=device, axis='y', sign=-1.0, g=9.81)

    for idx in tqdm(range(len(dataset))):
        theta_0_96, phi_0, betas, obj_mesh_path, is_left, sample_idx = dataset[idx]

        # 获取物体平移 (Trans)
        trans = phi_0[6:9]
        # DexYCB 坐标系 Y 轴向下，如果平移 Y 比较大，说明物体还在桌子上或较低位置
        # 我们只训练“抓取并举起”的样本，这里可以根据实际观察设定阈值
        # 比如 trans[1] < 0.3 (数值越小代表物体越高)
        if trans[1].item() > 0.45:  # 假设 0.45m 以下被认为是在桌面附近
            continue

        # 4.GPU 顶点变换
        base_mesh = load_mesh_cached(obj_mesh_path)
        obj_v_np = np.asarray(base_mesh.vertices)
        # 如果物体顶点是毫米，这里除以 1000
        if obj_v_np.max() > 10:
            obj_v_np /= 1000.0

        # 选择对应手部模型和锚点
        if is_left:
            mano_layer = mano_layer_left
            anchor_tri = ANCHOR_TRIANGLES_LEFT
            anchor_w = ANCHOR_WEIGHTS_LEFT
        else:
            mano_layer = mano_layer_right
            anchor_tri = ANCHOR_TRIANGLES_RIGHT
            anchor_w = ANCHOR_WEIGHTS_RIGHT

        # 计算手部顶点（全局坐标系，米）
        theta_48_pca = pose_96_to_48(theta_0_96.unsqueeze(0))  # PCA 48维
        theta_48_aa = pca_to_axis_angle(theta_48_pca.numpy(), mano_layer)
        theta_tensor = torch.tensor(theta_48_aa).float().to(device)
        beta_tensor = betas.detach().clone().unsqueeze(0).float().to(device)

        with torch.no_grad():
            verts, _ = mano_layer(theta_tensor, beta_tensor)

        # MANO输出按毫米处理，统一转米级单位
        verts = verts / 1000.0
        verts_np = verts[0]


        # 物体真值姿态
        r6d = phi_0[:6]
        trans = phi_0[6:9]
        R = rotation_6d_to_matrix(r6d.unsqueeze(0)).cpu().numpy()[0]  # 3x3
        T = trans.cpu().numpy()
        transform_mat = np.eye(4)
        transform_mat[:3, :3] = R
        transform_mat[:3, 3] = T


        try:
            scene, obj_center_np = get_scene_cached(obj_mesh_path, transform_mat)
        except Exception as e:
            print(f"构建/读取场景失败: {obj_mesh_path}, 错误: {e}")
            continue

        # 预加载物体顶点到 GPU (配合上面的提速)
        base_mesh = load_mesh_cached(obj_mesh_path)
        obj_v_tensor = torch.from_numpy(obj_v_np).float().to(device)
        # 应用当前 R 和 T
        obj_v_tensor = (torch.from_numpy(R).float().to(device) @ obj_v_tensor.T).T + \
                       torch.from_numpy(T).float().to(device)
        # 生成伪力标签
        try:
            w, s, local_F, lf, lt, dc = generate_pseudo_force_labels(
                verts_np, mano_layer.th_faces,
                anchor_tri.to(device), anchor_w.to(device),
                gravity,
                obj_center_np=obj_center_np,
                device=device,
                stage1_steps=80,
                stage2_steps=400,
                obj_verts_tensor=obj_v_tensor
            )
        except Exception as e:
            print(f"生成标签失败 (idx {idx}): {e}")
            continue

        # 保存
        save_path = os.path.join(save_dir, f'{idx:06d}.npz')
        np.savez(save_path,
                 local_F=local_F.numpy(),
                 w=w.numpy(),
                 s=s.numpy(),
                 is_left=is_left,
                 loss_force=lf,
                 loss_torque=lt,
                 mean_contact_dist=dc)
    print(f"完成！标签已保存至 {save_dir}")
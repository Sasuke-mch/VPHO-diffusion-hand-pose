
import os
import re
import cv2
import yaml
import pickle
import numpy as np
import torch
import open3d as o3d

from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Optional

from lib.geometry import (
    compute_object_keypoints_3d,
    project_3d_to_2d,
    get_hand_bbox_from_joints,
    get_object_bbox_from_2d_points,
    generate_roi_heatmap_from_joints,
)
from lib.config import YCB_CLASSES, mano_layer_left, mano_layer_right
from lib.rotation import pose_48_to_96, rotation_matrix_to_6d, pca_to_axis_angle
from lib.diffusion import precompute_diffusion_coeffs


def normalize_path(path: str) -> str:
    """
    同时兼容 Windows Python 和 WSL。
    如果用户在 WSL 中传入 D:/xxx，会自动尝试转换为 /mnt/d/xxx。
    """
    if path is None:
        return path

    path = str(path).replace("\\", "/")
    if os.path.exists(path):
        return path

    m = re.match(r"^([A-Za-z]):/(.*)$", path)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        wsl_path = f"/mnt/{drive}/{rest}"
        if os.path.exists(wsl_path):
            return wsl_path

    return path


def parse_camera_serial_from_path(label_path: str):
    """
    OpenDataLab DexYCB:
        .../20200709_143008/932122060861/labels_000054.npz

    labels 文件的父目录名就是当前相机 serial。
    这个 serial 不用于索引 pose_y；只用于找内参。
    """
    cam_dir = os.path.basename(os.path.dirname(label_path))
    if cam_dir.startswith("cam_"):
        cam_dir = cam_dir[len("cam_"):]
    try:
        return int(cam_dir)
    except Exception:
        return None


def load_sequence_meta_from_label(label_path: str) -> dict:
    seq_dir = os.path.dirname(os.path.dirname(label_path))
    meta_path = os.path.join(seq_dir, "meta.yml")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_object_pose_index(label_path: str, pose_y=None, strict: bool = True) -> int:
    """
    重要修正：
    你的 OpenDataLab DexYCB 结构中，labels_*.npz 已经位于具体相机目录下，
    因此 pose_y 不应该再用相机 serials 索引。

    已由你的样本验证：
        pose_y.shape = (5, 3, 4)
        meta["ycb_ids"] = [4, 7, 14, 16, 17]
    即：
        pose_y[i] 对应 ycb_ids[i]

    目标抓取物体使用：
        obj_idx = meta["ycb_grasp_ind"]

    兼容少数可能有占位位姿的情况：
        pose_y.shape[0] == len(ycb_ids) + 1 时，使用 ycb_grasp_ind + 1。
    """
    meta = load_sequence_meta_from_label(label_path)
    ycb_ids = meta.get("ycb_ids", [])
    ycb_grasp_ind = int(meta.get("ycb_grasp_ind", 0))

    if pose_y is None:
        return ycb_grasp_ind

    pose_y_arr = np.asarray(pose_y)

    # 单个物体姿态，例如 [3,4] 或 [4,4]
    if pose_y_arr.ndim == 2:
        return 0

    if pose_y_arr.ndim != 3:
        msg = f"Unexpected pose_y shape: {pose_y_arr.shape}, label_path={label_path}"
        if strict:
            raise RuntimeError(msg)
        print("警告:", msg)
        return 0

    n_pose = pose_y_arr.shape[0]
    n_obj = len(ycb_ids)

    if n_pose == n_obj:
        obj_idx = ycb_grasp_ind
    elif n_pose == n_obj + 1:
        obj_idx = ycb_grasp_ind + 1
    else:
        msg = (
            "无法唯一确定 pose_y 的物体索引。\n"
            f"pose_y.shape={pose_y_arr.shape}\n"
            f"len(ycb_ids)={n_obj}\n"
            f"ycb_ids={ycb_ids}\n"
            f"ycb_grasp_ind={ycb_grasp_ind}\n"
            f"label_path={label_path}\n"
            "请先用 check_pose_y_index.py 验证这个序列的 pose_y 顺序。"
        )
        if strict:
            raise RuntimeError(msg)
        print("警告:", msg)
        obj_idx = min(max(ycb_grasp_ind, 0), n_pose - 1)

    if obj_idx < 0 or obj_idx >= n_pose:
        msg = (
            f"object pose index 越界: obj_idx={obj_idx}, pose_y.shape={pose_y_arr.shape}, "
            f"label_path={label_path}"
        )
        if strict:
            raise RuntimeError(msg)
        print("警告:", msg)
        obj_idx = min(max(obj_idx, 0), n_pose - 1)

    return int(obj_idx)


def get_target_object_id_and_name(label_path: str):
    meta = load_sequence_meta_from_label(label_path)
    ycb_ids = meta.get("ycb_ids", [])
    ycb_grasp_ind = int(meta.get("ycb_grasp_ind", 0))

    if len(ycb_ids) > ycb_grasp_ind:
        obj_id = int(ycb_ids[ycb_grasp_ind])
    elif len(ycb_ids) > 0:
        obj_id = int(ycb_ids[0])
    else:
        obj_id = 1

    obj_name = YCB_CLASSES.get(obj_id, "002_master_chef_can")
    return obj_id, obj_name


def get_mano_layer_from_label(label_path: str):
    meta = load_sequence_meta_from_label(label_path)
    side = "right"
    if meta.get("mano_sides"):
        side = str(meta["mano_sides"][0]).lower()
    return mano_layer_left if side == "left" else mano_layer_right


def extract_root_joint_from_label_data(label_data) -> np.ndarray:
    """
    关键修正：
    诊断结果显示 pose_m[-3:] 与 joint_3d wrist 相差约 9.8 cm。
    OpenDataLab DexYCB 中用于 root-relative 手物统一坐标时，
    应该使用 label_data["joint_3d"][0, 0] 作为手腕根节点。
    """
    if "joint_3d" in label_data:
        joint_3d = np.asarray(label_data["joint_3d"])
        if joint_3d.ndim == 3 and joint_3d.shape[1] >= 1:
            return joint_3d[0, 0].astype(np.float32)
        if joint_3d.ndim == 2 and joint_3d.shape[0] >= 1:
            return joint_3d[0].astype(np.float32)

    # 兜底，不推荐
    return extract_root_joint_from_pose_m(label_data["pose_m"])


def pose_m_pca_to_axis_angle(label_path: str, pose_48_pca: np.ndarray) -> np.ndarray:
    """
    关键修正：
    诊断结果显示 pose_m[:48] 是 MANO PCA 表示，不是 16 个关节的 axis-angle。
    需要先 pca_to_axis_angle，再转换为 6D 旋转训练目标。
    """
    mano_layer = get_mano_layer_from_label(label_path)
    aa = pca_to_axis_angle(pose_48_pca.reshape(1, 48), mano_layer)
    aa = np.asarray(aa, dtype=np.float32)
    if aa.ndim == 2:
        aa = aa[0]
    return aa.reshape(48).astype(np.float32)


def select_transform_from_pose_y(label_path: str, pose_y, strict: bool = True):
    pose_y_arr = np.asarray(pose_y)
    if pose_y_arr.ndim == 2:
        transform = pose_y_arr
        obj_idx = 0
    else:
        obj_idx = resolve_object_pose_index(label_path, pose_y_arr, strict=strict)
        transform = pose_y_arr[obj_idx]

    if transform.shape == (3, 4):
        R = transform[:3, :3]
        T = transform[:3, 3]
    elif transform.shape == (4, 4):
        R = transform[:3, :3]
        T = transform[:3, 3]
    else:
        raise RuntimeError(f"Unexpected transform shape: {transform.shape}, label_path={label_path}")

    return R.astype(np.float32), T.astype(np.float32), obj_idx


def extract_root_joint_from_pose_m(pose_m) -> np.ndarray:
    pose_m = np.asarray(pose_m)
    if pose_m.shape == (1, 51):
        return pose_m[0, -3:].astype(np.float32)
    if pose_m.shape == (51,):
        return pose_m[-3:].astype(np.float32)
    return pose_m.reshape(-1)[-3:].astype(np.float32)


def extract_pose48_from_pose_m(pose_m) -> np.ndarray:
    pose_m = np.asarray(pose_m)
    if pose_m.shape == (1, 51):
        return pose_m[0, :48].astype(np.float32)
    if pose_m.shape == (51,):
        return pose_m[:48].astype(np.float32)
    return pose_m.reshape(-1)[:48].astype(np.float32)


class DexYCBDataset(Dataset):
    """
    OpenDataLab DexYCB 数据集加载类。

    关键修正：
    1. labels 文件已经位于具体相机 serial 目录下，不再用 serials.index(camera) 索引 pose_y。
    2. pose_y 第一维对应物体，按 meta.yml 的 ycb_grasp_ind 选择目标物体。
    3. hand bbox 用 joint_2d 从 640x480 缩放到 256x256 后再计算。
    4. 物体平移仍保持你的设计：T_obj - root_joint，即相对手腕。
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        T: int = 1000,
        strict_pose_index: bool = True,
    ):
        self.data_root = normalize_path(data_root)
        self.transform = transform
        self.T = T
        self.strict_pose_index = strict_pose_index

        self.beta, self.alpha, self.alpha_bar = precompute_diffusion_coeffs(T)
        self.samples = self._load_samples(split)

        print("预加载物体3D关键点...")
        self.obj_kpt_cache = {}
        self.mesh_path_cache = {}
        models_dir = os.path.join(self.data_root, "models")

        for obj_id, obj_name in YCB_CLASSES.items():
            mesh_path = self._find_object_mesh_path(obj_name)
            if mesh_path is not None:
                try:
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    self.obj_kpt_cache[obj_name] = compute_object_keypoints_3d(mesh)
                    self.mesh_path_cache[obj_name] = mesh_path
                except Exception as e:
                    print(f"  警告: 无法加载 {obj_name}: {e}")

        print(f"已缓存 {len(self.obj_kpt_cache)} 个物体的3D关键点")

        print("预加载相机内参...")
        self.intrinsics_cache = {}
        self._load_intrinsics_cache()
        print(f"已缓存 {len(self.intrinsics_cache)} 个相机的内参")
        print(f"Loaded {len(self.samples)} samples from DexYCB ({split} split)")

    def _find_object_mesh_path(self, obj_name: str):
        models_dir = os.path.join(self.data_root, "models")
        candidates = [
            os.path.join(models_dir, obj_name, "textured.obj"),
            os.path.join(models_dir, obj_name, "textured_simple.obj"),
            os.path.join(models_dir, obj_name, "textured.ply"),
            os.path.join(models_dir, obj_name, "textured_simple.ply"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _load_intrinsics_cache(self):
        calib_dir = os.path.join(self.data_root, "calibration", "intrinsics")
        if not os.path.exists(calib_dir):
            print(f"  警告: 内参目录不存在 {calib_dir}")
            return

        for cam_file in os.listdir(calib_dir):
            if not cam_file.endswith("_640x480.yml"):
                continue

            try:
                cam_name = int(cam_file.replace("_640x480.yml", ""))
            except Exception:
                continue

            file_path = os.path.join(calib_dir, cam_file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    calib_data = yaml.safe_load(f) or {}
            except Exception:
                with open(file_path, "r", encoding="utf-8") as f:
                    calib_data = yaml.unsafe_load(f) or {}

            K = None
            if "color" in calib_data:
                color = calib_data["color"]
                fx = color.get("fx", color.get("fx_color", None))
                fy = color.get("fy", color.get("fy_color", None))
                ppx = color.get("ppx", color.get("cx", None))
                ppy = color.get("ppy", color.get("cy", None))
                if fx is not None and fy is not None and ppx is not None and ppy is not None:
                    K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float32)
            elif "camera_matrix" in calib_data:
                K = np.array(calib_data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
            elif "K" in calib_data:
                raw = calib_data["K"]
                if isinstance(raw, dict) and "data" in raw:
                    K = np.array(raw["data"], dtype=np.float32).reshape(3, 3)
                else:
                    K = np.array(raw, dtype=np.float32).reshape(3, 3)

            if K is None:
                continue

            scale_x = 256.0 / 640.0
            scale_y = 256.0 / 480.0
            K_scaled = K.copy()
            K_scaled[0, :] *= scale_x
            K_scaled[1, :] *= scale_y
            K_scaled[2, 2] = 1.0

            self.intrinsics_cache[cam_name] = torch.tensor(K_scaled, dtype=torch.float32)
            print(f"  ✓ 已缓存: {cam_name}")

    def _load_samples(self, split: str) -> List[Tuple[str, str, np.ndarray, int]]:
        cache_path = os.path.join(self.data_root, f"cache_{split}_samples_corrected_v3.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                samples = pickle.load(f)
            print(f"从缓存加载 {len(samples)} 个样本 ({split} split)")
            return samples

        samples = []
        if not os.path.exists(self.data_root):
            print(f"数据根目录不存在: {self.data_root}")
            return samples

        all_subjects = [
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d)) and "subject-" in d
        ]

        if split == "train":
            subjects = [d for d in all_subjects if any(f"subject-{i:02d}" in d for i in range(1, 9))]
        elif split == "test":
            subjects = [d for d in all_subjects if "subject-09" in d or "subject-10" in d]
        elif split == "try":
            subjects = [d for d in all_subjects if "subject-01" in d]
        else:
            subjects = all_subjects

        subjects = sorted(subjects)
        print(f"共找到 {len(subjects)} 个 {split} 集 subject")

        stop_after_first_seq = split == "try"

        for subject_idx, subject_dir in enumerate(subjects):
            print(f"\n[{subject_idx + 1}/{len(subjects)}] 正在加载 subject: {subject_dir}")
            subject_path = os.path.join(self.data_root, subject_dir)
            sequences = sorted([
                d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d))
            ])
            print(f"  └─ 找到 {len(sequences)} 个序列")

            for seq_idx, seq_dir in enumerate(sequences):
                if seq_idx % 10 == 0:
                    print(f"    处理序列进度: {seq_idx + 1}/{len(sequences)}")

                seq_path = os.path.join(subject_path, seq_dir)
                betas = self._load_sequence_betas(seq_path)

                # 相机目录是 serial 数字字符串，例如 932122060861
                cam_dirs = sorted([
                    d for d in os.listdir(seq_path)
                    if os.path.isdir(os.path.join(seq_path, d)) and parse_camera_dir_name(d) is not None
                ])

                for cam_dir in cam_dirs:
                    camera_id = parse_camera_dir_name(cam_dir)
                    cam_path = os.path.join(seq_path, cam_dir)

                    npz_files = sorted([
                        f for f in os.listdir(cam_path)
                        if f.endswith(".npz") and f.startswith("labels_")
                    ])

                    for file in npz_files:
                        frame_id = file.replace("labels_", "").replace(".npz", "")
                        label_path = os.path.join(cam_path, file)
                        rgb_path = None
                        for ext in [".jpg", ".png", ".jpeg"]:
                            candidate = os.path.join(cam_path, f"color_{frame_id}{ext}")
                            if os.path.exists(candidate):
                                rgb_path = candidate
                                break

                        if rgb_path is not None:
                            samples.append((rgb_path, label_path, betas, camera_id))

                if stop_after_first_seq:
                    break

            print(f"  └─ {subject_dir} 完成，累计已加载 {len(samples)} 个样本")
            if split == "try":
                break

        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"缓存已保存到 {cache_path}")
        print(f"\n加载完成！共加载 {len(samples)} 个样本")
        return samples

    def _load_sequence_betas(self, seq_path: str) -> np.ndarray:
        betas = np.zeros(10, dtype=np.float32)

        # OpenDataLab 序列下常见 pose.npz，尽量从里面找 betas / mano_shape。
        pose_npz = os.path.join(seq_path, "pose.npz")
        candidate_files = []
        if os.path.exists(pose_npz):
            candidate_files.append(pose_npz)

        for file in os.listdir(seq_path):
            if file.startswith("info") and (file.endswith(".pkl") or file.endswith(".npz")):
                candidate_files.append(os.path.join(seq_path, file))

        for info_path in candidate_files:
            try:
                if info_path.endswith(".pkl"):
                    with open(info_path, "rb") as f:
                        info = pickle.load(f, encoding="latin1")
                    for key in ["betas", "mano_shape", "shape"]:
                        if key in info:
                            arr = np.asarray(info[key]).reshape(-1)
                            if arr.size >= 10:
                                return arr[:10].astype(np.float32)
                else:
                    info = np.load(info_path, allow_pickle=True)
                    for key in ["betas", "mano_shape", "shape"]:
                        if key in info:
                            arr = np.asarray(info[key]).reshape(-1)
                            if arr.size >= 10:
                                return arr[:10].astype(np.float32)
            except Exception:
                continue

        return betas

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_pose_from_label(self, label_path: str) -> np.ndarray:
        data = np.load(label_path, allow_pickle=True)

        # pose_m[:48] 是 MANO PCA 表示；先解码为 48D axis-angle，
        # 再转换为 96D 6D rotation，作为真正的手部扩散训练目标。
        pose_48_pca = extract_pose48_from_pose_m(data["pose_m"])
        pose_48_aa = pose_m_pca_to_axis_angle(label_path, pose_48_pca)
        pose_96 = pose_48_to_96(pose_48_aa)
        return pose_96.astype(np.float32)

    def _load_object_pose_from_label(self, label_path: str, camera_id: int = None) -> np.ndarray:
        data = np.load(label_path, allow_pickle=True)
        pose_y = data["pose_y"]

        R, T, obj_idx = select_transform_from_pose_y(
            label_path=label_path,
            pose_y=pose_y,
            strict=self.strict_pose_index,
        )

        root_joint = extract_root_joint_from_label_data(data)

        # 你的设计选择：物体平移相对手腕；root 使用 joint_3d[0,0]，不是 pose_m[-3:]
        trans_rel = T.astype(np.float32) - root_joint.astype(np.float32)

        r6d = rotation_matrix_to_6d(torch.tensor(R, dtype=torch.float32)).detach().cpu().numpy()
        return np.concatenate([r6d.astype(np.float32), trans_rel.astype(np.float32)])

    def _make_noisy_pair(self, clean_pose: np.ndarray, pose_dim: int):
        clean_pose = torch.tensor(clean_pose, dtype=torch.float32)
        t = torch.randint(1, self.T + 1, (1,), dtype=torch.long).item()
        eps = torch.randn_like(clean_pose)

        alpha_bar_t = self.alpha_bar[t - 1]
        noisy = torch.sqrt(alpha_bar_t) * clean_pose + torch.sqrt(1.0 - alpha_bar_t) * eps
        return noisy, t, eps

    def __getitem__(self, idx: int):
        image_path, label_path, betas, camera_id = self.samples[idx]

        image = self._load_image(image_path)
        if self.transform:
            image = self.transform(image)

        label_data = np.load(label_path, allow_pickle=True)

        theta_0 = self._load_pose_from_label(label_path)
        phi_0 = self._load_object_pose_from_label(label_path, camera_id)

        theta_t, t_hand, eps_hand = self._make_noisy_pair(theta_0, 96)
        phi_t, t_obj, eps_obj = self._make_noisy_pair(phi_0, 9)

        # 使用同一个 t 训练手和物体，便于 batch 对齐；这里沿用手部 t
        t = torch.tensor(t_hand, dtype=torch.long)

        _, object_name = get_target_object_id_and_name(label_path)
        object_mesh_path = self._find_object_mesh_path(object_name)
        if object_mesh_path is None:
            object_mesh_path = os.path.join(self.data_root, "models", object_name, "textured.obj")

        # hand bbox：joint_2d 原始是 640x480，先缩放到 256x256
        hand_joints_2d = label_data["joint_2d"][0].astype(np.float32)
        hand_joints_2d_256 = hand_joints_2d.copy()
        hand_joints_2d_256[:, 0] *= 256.0 / 640.0
        hand_joints_2d_256[:, 1] *= 256.0 / 480.0
        bbox_hand = get_hand_bbox_from_joints(hand_joints_2d_256)
        if bbox_hand[2] <= bbox_hand[0] or bbox_hand[3] <= bbox_hand[1]:
            bbox_hand = np.array([64, 64, 192, 192], dtype=np.float32)
        bbox_hand_norm = bbox_hand / 256.0

        # object bbox：使用目标物体 pose_y[obj_idx] 投影关键点
        pose_y = label_data["pose_y"]
        R_obj, T_obj, obj_idx = select_transform_from_pose_y(
            label_path=label_path,
            pose_y=pose_y,
            strict=self.strict_pose_index,
        )

        keypoints_3d_object = self.obj_kpt_cache.get(object_name, None)
        intrinsic_matrix = self.intrinsics_cache.get(camera_id, None)

        if keypoints_3d_object is not None and intrinsic_matrix is not None:
            keypoints_3d_cam = (R_obj @ keypoints_3d_object.T).T + T_obj
            keypoints_2d = project_3d_to_2d(
                torch.tensor(keypoints_3d_cam, dtype=torch.float32),
                intrinsic_matrix,
            ).detach().cpu().numpy()
            bbox_obj = get_object_bbox_from_2d_points(keypoints_2d)
        else:
            bbox_obj = np.array([64, 64, 192, 192], dtype=np.float32)

        if bbox_obj[2] <= bbox_obj[0] or bbox_obj[3] <= bbox_obj[1]:
            bbox_obj = np.array([64, 64, 192, 192], dtype=np.float32)

        bbox_obj_norm = bbox_obj / 256.0

        return (
            theta_t,
            phi_t,
            t,
            image,
            torch.tensor(theta_0, dtype=torch.float32),
            torch.tensor(phi_0, dtype=torch.float32),
            eps_hand,
            eps_obj,
            torch.tensor(betas, dtype=torch.float32),
            object_mesh_path,
            torch.tensor(bbox_hand_norm, dtype=torch.float32),
            torch.tensor(bbox_obj_norm, dtype=torch.float32),
        )


def parse_camera_dir_name(name: str):
    name = str(name)
    if name.startswith("cam_"):
        name = name[len("cam_"):]
    try:
        return int(name)
    except Exception:
        return None


class HeatmapDataset(DexYCBDataset):
    """
    第一阶段热图训练数据集。

    关键修正：
    object heatmap 使用 pose_y[obj_idx]，而不是 pose_y[cam_idx]。
    """

    def __getitem__(self, idx):
        image_path, label_path, betas, camera_id = self.samples[idx]

        image = self._load_image(image_path)
        if self.transform:
            image = self.transform(image)

        label_data = np.load(label_path, allow_pickle=True)

        # hand heatmap
        hand_joints_2d = label_data["joint_2d"][0].astype(np.float32)
        hand_joints_2d_256 = hand_joints_2d.copy()
        hand_joints_2d_256[:, 0] *= 256.0 / 640.0
        hand_joints_2d_256[:, 1] *= 256.0 / 480.0

        bbox_hand = get_hand_bbox_from_joints(hand_joints_2d_256)
        if bbox_hand[2] <= bbox_hand[0] or bbox_hand[3] <= bbox_hand[1]:
            bbox_hand = np.array([64, 64, 192, 192], dtype=np.float32)

        hand_heatmap_gt = generate_roi_heatmap_from_joints(
            hand_joints_2d_256,
            bbox_hand,
            heatmap_size=64,
            sigma=2.0,
        )

        bbox_hand_norm = bbox_hand / 256.0

        # object heatmap
        _, object_name = get_target_object_id_and_name(label_path)
        pose_y = label_data["pose_y"]
        R_obj, T_obj, obj_idx = select_transform_from_pose_y(
            label_path=label_path,
            pose_y=pose_y,
            strict=self.strict_pose_index,
        )

        keypoints_3d_object = self.obj_kpt_cache.get(object_name, None)
        intrinsic_matrix = self.intrinsics_cache.get(camera_id, None)

        if keypoints_3d_object is not None and intrinsic_matrix is not None:
            keypoints_3d_cam = (R_obj @ keypoints_3d_object.T).T + T_obj
            keypoints_2d = project_3d_to_2d(
                torch.tensor(keypoints_3d_cam, dtype=torch.float32),
                intrinsic_matrix,
            ).detach().cpu().numpy()
            bbox_obj = get_object_bbox_from_2d_points(keypoints_2d)
        else:
            keypoints_2d = np.zeros((27, 2), dtype=np.float32)
            bbox_obj = np.array([64, 64, 192, 192], dtype=np.float32)

        if bbox_obj[2] <= bbox_obj[0] or bbox_obj[3] <= bbox_obj[1]:
            bbox_obj = np.array([64, 64, 192, 192], dtype=np.float32)

        bbox_obj_norm = bbox_obj / 256.0

        object_heatmap_gt = generate_roi_heatmap_from_joints(
            keypoints_2d,
            bbox_obj,
            heatmap_size=64,
            sigma=2.0,
        )

        return (
            image,
            torch.tensor(hand_heatmap_gt, dtype=torch.float32),
            torch.tensor(object_heatmap_gt, dtype=torch.float32),
            torch.tensor(bbox_hand_norm, dtype=torch.float32),
            torch.tensor(bbox_obj_norm, dtype=torch.float32),
        )

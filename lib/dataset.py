import os
import cv2
import open3d as o3d
import yaml
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Optional
from lib.geometry import *
from lib.config import YCB_CLASSES
from lib.rotation import *
from lib.diffusion import precompute_diffusion_coeffs

class DexYCBDataset(Dataset):
    """
    DexYCB 数据集加载类，负责读取RGB图像及手部/物体的真值标签，
    并在前向过程中(Forward Diffusion)为数据添加随机噪声。
    """

    def __init__(self, data_root: str, split: str = 'train', transform: Optional[transforms.Compose] = None,
                 T: int = 1000):
        self.data_root = data_root
        self.transform = transform
        self.T = T
        self.beta, self.alpha, self.alpha_bar = precompute_diffusion_coeffs(T)
        self.samples = self._load_samples(split)

        print("预加载物体3D关键点...")
        self.obj_kpt_cache = {}
        models_dir = os.path.join(data_root, "models")
        for obj_id, obj_name in YCB_CLASSES.items():
            mesh_path = os.path.join(models_dir, obj_name, "textured.obj")
            if os.path.exists(mesh_path):
                try:
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    self.obj_kpt_cache[obj_name] = compute_object_keypoints_3d(mesh)
                except Exception as e:
                    print(f"  警告: 无法加载 {obj_name}: {e}")
        print(f"已缓存 {len(self.obj_kpt_cache)} 个物体的3D关键点")

        # 预加载相机内参缓存
        print("预加载相机内参...")
        self.intrinsics_cache = {}  # key: camera_id (int), value: torch.Tensor [3,3]

        calib_dir = os.path.join(data_root, 'calibration', 'intrinsics')
        if os.path.exists(calib_dir):
            import yaml
            for cam_file in os.listdir(calib_dir):
                if cam_file.endswith('_640x480.yml'):
                    cam_name = int(cam_file.replace('_640x480.yml', ''))
                    file_path = os.path.join(calib_dir, cam_file)
                    with open(file_path, 'r') as f:
                        calib_data = yaml.unsafe_load(f)

                    if 'color' in calib_data:
                        color = calib_data['color']
                        fx = color['fx']
                        fy = color['fy']
                        ppx = color['ppx']
                        ppy = color['ppy']
                        K = np.array([
                            [fx, 0, ppx],
                            [0, fy, ppy],
                            [0, 0, 1]
                        ], dtype=np.float32)
                    else:
                        continue

                    scale_x = 256.0 / 640.0
                    scale_y = 256.0 / 480.0
                    K_scaled = K.copy()
                    K_scaled[0, :] *= scale_x
                    K_scaled[1, :] *= scale_y
                    K_scaled[2, 2] = 1.0
                    self.intrinsics_cache[cam_name] = torch.tensor(K_scaled, dtype=torch.float32)
                    print(f"  ✓ 已缓存: cam_{cam_name}")
        else:
            print(f"  警告: 内参目录不存在 {calib_dir}")
        print(f"已缓存 {len(self.intrinsics_cache)} 个相机的内参")
        print(f"Loaded {len(self.samples)} samples from DexYCB ({split} split)")

    def _load_samples(self, split: str) -> List[Tuple[str, str, np.ndarray, int]]:
        # 检查缓存是否存在
        cache_path = os.path.join(self.data_root, f"cache_{split}_samples.pkl")
        if os.path.exists(cache_path):
            import pickle
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            print(f"从缓存加载 {len(samples)} 个样本 ({split} split)")
            return samples

        samples = []
        if not os.path.exists(self.data_root):
            return samples

        all_subjects = [d for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d))]
        if split == 'train':
            subjects = [d for d in all_subjects if any(f'subject-{i:02d}' in d for i in range(1, 9))]
        elif split == 'test':
            subjects = [d for d in all_subjects if 'subject-09' in d or 'subject-10' in d]
        elif split == 'try':
            subjects = [d for d in all_subjects if 'subject-01' in d]
        else:
            subjects = all_subjects

        total_subjects = len(subjects)
        print(f"共找到 {total_subjects} 个 {split} 集 subject")

        for subject_idx, subject_dir in enumerate(subjects):
            print(f"\n[{subject_idx + 1}/{total_subjects}] 正在加载 subject: {subject_dir}")
            subject_path = os.path.join(self.data_root, subject_dir)
            sequences = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
            print(f"  └─ 找到 {len(sequences)} 个序列")

            for seq_idx, seq_dir in enumerate(sequences):
                # 每5个序列打印一次进度，避免刷屏
                if seq_idx % 10 == 0:
                    print(f"    处理序列进度: {seq_idx + 1}/{len(sequences)}")

                seq_path = os.path.join(subject_path, seq_dir)
                betas = self._load_sequence_betas(seq_path)
                cam_dirs = [d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))]

                for cam_dir in cam_dirs:
                    camera_id = int(cam_dir.replace('cam_', ''))
                    cam_path = os.path.join(seq_path, cam_dir)
                    npz_files = [f for f in os.listdir(cam_path) if f.endswith('.npz') and 'labels_' in f]

                    for file in npz_files:
                        frame_id = file.replace('labels_', '').replace('.npz', '')
                        label_path = os.path.join(cam_path, file)
                        rgb_path = None
                        for ext in ['.jpg', '.png']:
                            candidate = os.path.join(cam_path, f'color_{frame_id}{ext}')
                            if os.path.exists(candidate):
                                rgb_path = candidate
                                break
                        if rgb_path:
                            samples.append((rgb_path, label_path, betas, camera_id))
                if split == 'try':
                    break

            print(f"  └─ {subject_dir} 完成，累计已加载 {len(samples)} 个样本")
            # ========== 3. 保存缓存 ==========
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"缓存已保存到 {cache_path}")
        print(f"\n加载完成！共加载 {len(samples)} 个样本")
        return samples

    def _load_sequence_betas(self, seq_path: str) -> np.ndarray:
        betas = np.zeros(10, dtype=np.float32)
        for file in os.listdir(seq_path):
            if file.startswith('info') and (file.endswith('.pkl') or file.endswith('.npz')):
                info_path = os.path.join(seq_path, file)
                try:
                    if info_path.endswith('.pkl'):
                        with open(info_path, 'rb') as f:
                            info = pickle.load(f, encoding='latin1')
                            if 'betas' in info:
                                betas = np.array(info['betas'][:10], dtype=np.float32)
                            elif 'mano_shape' in info:
                                betas = np.array(info['mano_shape'][:10], dtype=np.float32)
                    elif info_path.endswith('.npz'):
                        info = np.load(info_path, allow_pickle=True)
                        if 'betas' in info:
                            betas = info['betas'][:10].astype(np.float32)
                        elif 'mano_shape' in info:
                            betas = info['mano_shape'][:10].astype(np.float32)
                except Exception:
                    pass
                break
        return betas

    def __getitem__(self, idx: int):
        image_path, label_path, betas, camera_id = self.samples[idx]
        image = self._load_image(image_path)
        if self.transform:
            image = self.transform(image)

        theta_0 = self._load_pose_from_label(label_path)
        phi_0 = self._load_object_pose_from_label(label_path, camera_id)

        # ========== 从 meta.yml 解析物体名称 ==========
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
                    YCB_CLASSES = {
                        1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box', 4: '005_tomato_soup_can',
                        5: '006_mustard_bottle', 6: '007_tuna_fish_can', 7: '008_pudding_box', 8: '009_gelatin_box',
                        9: '010_potted_meat_can', 10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
                        13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block', 17: '037_scissors',
                        18: '040_large_marker', 19: '051_large_clamp', 20: '052_extra_large_clamp', 21: '061_foam_brick'
                    }
                    object_name = YCB_CLASSES.get(ycb_ids[ycb_grasp_ind], object_name)
            except Exception:
                pass
        # =============================================

        dexycb_path = "/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb/models"
        object_mesh_path = os.path.join(dexycb_path, str(object_name), "textured.obj")

        # ========== 计算 bbox ==========
        label_data = np.load(label_path, allow_pickle=True)

        # 手部 bbox（和之前一样）
        hand_joints_2d = label_data['joint_2d'][0]
        bbox_hand = get_hand_bbox_from_joints(hand_joints_2d)
        bbox_hand_norm = bbox_hand / 256.0

        # 物体 bbox（用缓存）
        cam_idx = 0
        seq_dir = os.path.dirname(os.path.dirname(label_path))
        meta_path = os.path.join(seq_dir, 'meta.yml')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta_cam = yaml.safe_load(f)
                cam_serials = meta_cam.get('cam_serials', [])
                if camera_id in cam_serials:
                    cam_idx = cam_serials.index(camera_id)
            except Exception:
                pass

        pose_y = label_data['pose_y']
        transform_mat = pose_y[cam_idx]
        R_obj = transform_mat[:3, :3]
        T_obj = transform_mat[:3, 3]

        # 物体3D关键点 —— 直接从缓存拿
        kpt3d_obj = self.obj_kpt_cache.get(object_name)
        if kpt3d_obj is not None:
            kpt3d_cam = (R_obj @ kpt3d_obj.T).T + T_obj

            # 内参 —— 直接从缓存拿
            K = self.intrinsics_cache.get(camera_id)
            if K is None:
                # 兜底：用默认内参
                K_default = np.array([[617, 0, 320], [0, 617, 240], [0, 0, 1]], dtype=np.float32)
                scale_x, scale_y = 256 / 640, 256 / 480
                K_default[0, :] *= scale_x
                K_default[1, :] *= scale_y
                K_default[2, 2] = 1.0
                K = torch.tensor(K_default, dtype=torch.float32)

            kpt2d_obj = project_3d_to_2d(
                torch.tensor(kpt3d_cam, dtype=torch.float32),
                K
            ).numpy()

            bbox_obj = get_object_bbox_from_2d_points(kpt2d_obj)
        else:
            # 没有关键点缓存，用默认 bbox
            bbox_obj = np.array([64, 64, 192, 192], dtype=np.float32)

        bbox_obj_norm = bbox_obj / 256.0
        # =================================

        t = torch.randint(1, self.T + 1, (1,)).item()
        alpha_bar_t = self.alpha_bar[t - 1]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)

        eps_hand = torch.randn(96)
        theta_t = sqrt_alpha_bar_t * theta_0 + sqrt_one_minus * eps_hand

        eps_obj = torch.randn(9)
        phi_t = sqrt_alpha_bar_t * phi_0 + sqrt_one_minus * eps_obj

        return (theta_t, phi_t, torch.tensor(t, dtype=torch.long), image,
                theta_0, phi_0, eps_hand, eps_obj, betas, object_mesh_path,
                torch.tensor(bbox_hand_norm, dtype=torch.float32),
                torch.tensor(bbox_obj_norm, dtype=torch.float32))

    def _parse_path_info(self, image_path):
        """返回 (subject, seq)"""
        parts = image_path.replace('\\', '/').split('/')
        # 路径格式: .../subject_dir/seq_dir/cam_dir/color_xxx.jpg
        return parts[-4], parts[-3]

    def _load_object_pose_from_label(self, label_path: str, camera_id: int = 0) -> np.ndarray:
        data = np.load(label_path, allow_pickle=True)
        pose_y = data['pose_y']

        # ========== 修复：相机序列号 → pose_y 索引 ==========
        import yaml
        seq_dir = os.path.dirname(os.path.dirname(label_path))
        meta_path = os.path.join(seq_dir, 'meta.yml')

        cam_idx = 0  # 默认值
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = yaml.safe_load(f)
                cam_serials = meta.get('cam_serials', [])
                if camera_id in cam_serials:
                    cam_idx = cam_serials.index(camera_id)
            except Exception:
                pass
        # ===================================================

        transform = pose_y[cam_idx]
        R = transform[:3, :3]
        trans = transform[:3, 3]

        # 转为相对于手腕的平移
        pose_m = data['pose_m']
        if pose_m.shape == (1, 51):
            root_joint = pose_m[0][-3:]
        elif pose_m.shape == (51,):
            root_joint = pose_m[-3:]
        else:
            root_joint = pose_m.flatten()[-3:]
        trans = trans - root_joint

        R_tensor = torch.tensor(R, dtype=torch.float32)
        r6d = rotation_matrix_to_6d(R_tensor).numpy()
        return np.concatenate([r6d, trans.astype(np.float32)])

    def _load_pose_from_label(self, label_path: str) -> np.ndarray:
        data = np.load(label_path, allow_pickle=True)
        pose_m = data['pose_m']
        if pose_m.shape == (1, 51):
            pose_48 = pose_m[0][:48]
        elif pose_m.shape == (51,):
            pose_48 = pose_m[:48]
        else:
            pose_48 = pose_m.flatten()[:48]
        pose_96 = pose_48_to_96(pose_48)
        return pose_96.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



class HeatmapDataset(DexYCBDataset):
    """
    为物体热图训练扩展数据集，预加载物体网格和相机内参
    """

    def __init__(self, data_root: str, split: str = 'train',
                 transform=None, T: int = 1000):
        super().__init__(data_root, split, transform, T)

        # ========== 预加载所有物体的3D关键点 ==========
        print("预加载物体3D关键点...")
        self.object_keypoints_cache = {}
        models_dir = os.path.join(data_root, 'models')

        if os.path.exists(models_dir):
            for obj_name in os.listdir(models_dir):
                obj_path = os.path.join(models_dir, obj_name, 'textured_simple.obj')
                if os.path.exists(obj_path):
                    try:
                        mesh = o3d.io.read_triangle_mesh(obj_path)
                        keypoints_3d = compute_object_keypoints_3d(mesh)
                        self.object_keypoints_cache[obj_name] = keypoints_3d
                        print(f"  ✓ 预加载物体: {obj_name}")
                    except Exception as e:
                        print(f"  ✗ 加载失败: {obj_name}, {e}")
                else:
                    obj_path = os.path.join(models_dir, obj_name, 'textured.obj')
                    if os.path.exists(obj_path):
                        try:
                            mesh = o3d.io.read_triangle_mesh(obj_path)
                            keypoints_3d = compute_object_keypoints_3d(mesh)
                            self.object_keypoints_cache[obj_name] = keypoints_3d
                            print(f"  ✓ 预加载物体: {obj_name}")
                        except Exception as e:
                            print(f"  ✗ 加载失败: {obj_name}, {e}")

        # ========== 预加载相机内参 ==========
        print("预加载相机内参...")
        self.intrinsics_cache = {}
        calib_dir = os.path.join(data_root, 'calibration', 'intrinsics')

        if os.path.exists(calib_dir):
            for cam_file in os.listdir(calib_dir):
                if cam_file.endswith('_640x480.yml'):
                    cam_name = int(cam_file.replace('_640x480.yml', ''))
                    import yaml
                    with open(os.path.join(calib_dir, cam_file), 'r') as f:
                        calib_data = yaml.load(f, Loader=yaml.FullLoader)

                    if 'camera_matrix' in calib_data:
                        K_flat = calib_data['camera_matrix']['data']
                        K = np.array(K_flat, dtype=np.float32).reshape(3, 3)
                    elif 'K' in calib_data:
                        K = np.array(calib_data['K']['data'], dtype=np.float32).reshape(3, 3)
                    else:
                        continue

                    scale_x = 256.0 / 640.0
                    scale_y = 256.0 / 480.0
                    K_scaled = K.copy()
                    K_scaled[0, :] *= scale_x
                    K_scaled[1, :] *= scale_y
                    K_scaled[2, 2] = 1.0

                    self.intrinsics_cache[cam_name] = torch.tensor(K_scaled, dtype=torch.float32)
                    print(f"  ✓ 预加载相机内参: cam_{cam_name}")
        else:
            print("  警告: calibration/intrinsics 目录不存在，将使用默认内参")

    # ===== 修改：完整的 __getitem__，包含 bbox 计算 =====
    def __getitem__(self, idx):
        image_path, label_path, betas, camera_id = self.samples[idx]

        image = self._load_image(image_path)
        if self.transform:
            image = self.transform(image)

        label_data = np.load(label_path, allow_pickle=True)

        # ========== 解析物体名称 ==========
        import yaml
        sequence_dir = os.path.dirname(os.path.dirname(label_path))
        meta_path = os.path.join(sequence_dir, 'meta.yml')

        object_name = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta_data = yaml.safe_load(f)
                ycb_ids = meta_data.get('ycb_ids', [])
                ycb_grasp_ind = meta_data.get('ycb_grasp_ind', 0)
                if len(ycb_ids) > ycb_grasp_ind:
                    obj_id = ycb_ids[ycb_grasp_ind]
                    object_name = YCB_CLASSES.get(obj_id)
        if object_name is None:
            object_name = "unknown"

        # ========== 手部热图真值 + 手部 bbox ==========
        hand_joints_2d = label_data['joint_2d'][0].astype(np.float32)

        # DexYCB 原始标注通常是 640x480 坐标；你的输入图像 resize 到 256x256
        hand_joints_2d_256 = hand_joints_2d.copy()
        hand_joints_2d_256[:, 0] *= 256.0 / 640.0
        hand_joints_2d_256[:, 1] *= 256.0 / 480.0

        # 用 256 坐标算 bbox
        bbox_hand = get_hand_bbox_from_joints(hand_joints_2d_256)

        # 生成 ROI 内 heatmap
        hand_heatmap_gt = generate_roi_heatmap_from_joints(
            hand_joints_2d_256,
            bbox_hand,
            heatmap_size=64,
            sigma=2.0
        )
        hand_heatmap_gt = torch.tensor(hand_heatmap_gt, dtype=torch.float32)

        bbox_hand_norm = bbox_hand / 256.0

        # ========== 获取相机索引 ==========
        image_dir = os.path.dirname(image_path)
        cam_serial = int(os.path.basename(image_dir).replace('cam_', ''))

        cam_idx = 0
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta_cam = yaml.safe_load(f)
                cam_serials = meta_cam.get('cam_serials', [])
                if cam_serial in cam_serials:
                    cam_idx = cam_serials.index(cam_serial)
            except Exception:
                pass

        pose_y = label_data['pose_y']
        transform_mat = pose_y[cam_idx]
        R_obj = transform_mat[:3, :3]
        T_obj = transform_mat[:3, 3]

        # ========== 物体热图真值 + 物体 bbox ==========
        keypoints_3d_object = self.object_keypoints_cache.get(object_name)

        if keypoints_3d_object is not None:
            keypoints_3d_cam = (R_obj @ keypoints_3d_object.T).T + T_obj

            intrinsic_matrix = self.intrinsics_cache.get(cam_serial)
            if intrinsic_matrix is None:
                K_default = np.array([[617, 0, 320], [0, 617, 240], [0, 0, 1]], dtype=np.float32)
                scale_x, scale_y = 256/640, 256/480
                K_default[0, :] *= scale_x
                K_default[1, :] *= scale_y
                K_default[2, 2] = 1.0
                intrinsic_matrix = torch.tensor(K_default, dtype=torch.float32)

            keypoints_2d = project_3d_to_2d(
                torch.tensor(keypoints_3d_cam, device='cpu'),
                intrinsic_matrix
            ).numpy()

            bbox_obj = get_object_bbox_from_2d_points(keypoints_2d)
        else:
            keypoints_2d = np.zeros((27, 2), dtype=np.float32)
            bbox_obj = np.array([64, 64, 192, 192], dtype=np.float32)

        bbox_obj_norm = bbox_obj / 256.0

        object_heatmap_gt = generate_roi_heatmap_from_joints(
            keypoints_2d,
            bbox_obj,
            heatmap_size=64,
            sigma=2.0
        )
        object_heatmap_gt = torch.tensor(object_heatmap_gt, dtype=torch.float32)

        return (image, hand_heatmap_gt, object_heatmap_gt,
                torch.tensor(bbox_hand_norm, dtype=torch.float32),
                torch.tensor(bbox_obj_norm, dtype=torch.float32))
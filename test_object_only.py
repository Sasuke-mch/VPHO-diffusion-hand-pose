import torch
import numpy as np
import open3d as o3d
from torchvision import transforms
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_diffusion_coeffs(T=1000, beta_start=1e-4, beta_end=0.02, device=None):
    beta = torch.linspace(beta_start, beta_end, T, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

from torch import nn
from torchvision import models

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class DualBranchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.hand_refinement = nn.Sequential(
            BottleneckBlock(2048, planes=512),
            BottleneckBlock(2048, planes=512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.object_refinement = nn.Sequential(
            BottleneckBlock(2048, planes=512),
            BottleneckBlock(2048, planes=512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, image):
        x = self.backbone(image)
        psi_h = self.hand_refinement(x).view(x.size(0), -1)
        psi_o = self.object_refinement(x).view(x.size(0), -1)
        return psi_h, psi_o

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        device = t.device
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ObjectDiffusionModel(nn.Module):
    def __init__(self, cond_dim=2048, pose_dim=9, time_dim=256, hidden_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(pose_dim + hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def forward(self, phi_t, t, psi_obj):
        t_emb = self.time_mlp(t)
        cond_emb = self.cond_mlp(psi_obj)
        inp = torch.cat([phi_t, t_emb, cond_emb], dim=-1)
        eps_pred = self.net(inp)
        return eps_pred

def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    6D连续表示 → 旋转矩阵 (3x3)
    """
    if r6d.dim() == 1:
        r6d = r6d.unsqueeze(0)
        single = True
    else:
        single = False

    a1 = r6d[:, :3]
    a2 = r6d[:, 3:6]

    b1 = a1 / torch.norm(a1, dim=-1, keepdim=True)
    b2 = a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=1)

    if single:
        R = R[0]
    return R

def reverse_diffusion_object(model, psi_o, T, device, alpha, alpha_bar, pose_dim=9):
    """
    反向扩散生成物体姿态候选
    """
    B = psi_o.shape[0]
    phi_t = torch.randn(B, pose_dim).to(device)

    for t in reversed(range(1, T + 1)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model(phi_t, t_tensor, psi_o)

        alpha_t = alpha[t - 1]
        alpha_bar_t = alpha_bar[t - 1]
        beta_t = 1 - alpha_t

        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

        phi_det = coeff1 * (phi_t - coeff2 * eps_pred)

        if t > 1:
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn_like(phi_t)
            phi_t = phi_det + sigma_t * z
        else:
            phi_t = phi_det

    return phi_t

def visualize_single_object(obj_pose_9d, object_mesh_path):
    geometries = []

    # 添加原点坐标系 (世界系)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(world_frame)

    if os.path.exists(object_mesh_path):
        obj_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
        if obj_mesh.has_vertices():

            if torch.is_tensor(obj_pose_9d):
                obj_pose = obj_pose_9d.detach().cpu().numpy()
            else:
                obj_pose = obj_pose_9d

            obj_pose = obj_pose.reshape(-1)

            r6d = obj_pose[:6]
            trans = obj_pose[6:9]

            print(f"Predicted Object Translation: {trans}")

            r6d_tensor = torch.tensor(r6d, dtype=torch.float32)
            R = rotation_6d_to_matrix(r6d_tensor).numpy()

            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = trans

            obj_mesh.transform(transform_matrix)
            obj_mesh.compute_vertex_normals()
            obj_mesh.paint_uniform_color([0.2, 0.6, 0.8])  # 蓝色

            geometries.append(obj_mesh)

            # 添加物体自身的局部坐标系
            obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            obj_frame.transform(transform_matrix)
            geometries.append(obj_frame)

            print(f"已加载物体模型: {object_mesh_path}")
        else:
            print(f"物体模型文件无效: {object_mesh_path}")
    else:
        print(f"物体模型不存在: {object_mesh_path}")

    if len(geometries) > 1:
        all_pts = []
        for g in geometries:
            all_pts.append(np.asarray(g.vertices))
        all_pts = np.concatenate(all_pts, axis=0)
        center = np.mean(all_pts, axis=0)

        print(f"将整体场景向中心锚点平移 {-center} 以自动居中视野")
        for g in geometries:
            g.translate(-center)

    print(f"显示物体，共 {len(geometries)} 个几何体")
    o3d.visualization.draw_geometries(geometries, window_name="Predict Object Pose Only")


def test_single_image_object_only(image_path, object_mesh_path):
    print(f"Using device: {device}")
    T = 1000
    beta, alpha, alpha_bar = precompute_diffusion_coeffs(T, device=device)

    feature_extractor = DualBranchFeatureExtractor().to(device)
    object_model = ObjectDiffusionModel(
        cond_dim=2048,
        pose_dim=9,
        time_dim=256,
        hidden_dim=512
    ).to(device)

    try:
        object_model.load_state_dict(torch.load("diffusion_model_object_model.pth", map_location=device))
        print("已加载 diffusion_model_object_model.pth")
    except Exception as e:
        print(f"未能加载物体模型权重，请检查路径。错误: {e}")
        return

    feature_extractor.eval()
    object_model.eval()

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"无法读取图片: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        _, psi_o = feature_extractor(image_tensor)

    print("生成物体候选扩散过程...")
    with torch.no_grad():
        obj_pose_9d = reverse_diffusion_object(object_model, psi_o, T, device, alpha, alpha_bar, pose_dim=9)

    # 6. 可视化
    visualize_single_object(obj_pose_9d[0], object_mesh_path)

if __name__ == "__main__":
    test_image_path = "D:/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb/20201015-subject-09/20201015_142601/839512060362/color_000049.jpg"
    test_obj_mesh = "D:/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb/models/002_master_chef_can/textured.obj"

    # 显示真实图像
    image_bgr = cv2.imread(test_image_path)
    cv2.imshow("The picture", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    test_single_image_object_only(test_image_path, test_obj_mesh)


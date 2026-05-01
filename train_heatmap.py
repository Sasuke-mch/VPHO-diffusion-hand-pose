import sys
sys.path.insert(0, '.')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
from lib.config import device, transform
from lib.models import FeatureExtractor, HeatmapPredictor
from lib.diffusion import roi_crop
from lib.utils import setup_logger, plot_loss_curve
from lib.dataset import HeatmapDataset

def train_heatmap(data_root, batch_size=72, epochs=10, lr=1e-4):
    # Dataset
    dataset = HeatmapDataset(data_root, split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 模型
    feature_extractor = FeatureExtractor().to(device)
    heatmap_predictor = HeatmapPredictor().to(device)

    # 冻结 FPN
    for param in feature_extractor.fpn.parameters():
        param.requires_grad = False
    feature_extractor.eval()

    optimizer = optim.Adam(heatmap_predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger = setup_logger("train_heatmap.log")
    logger.info("开始训练 heatmap")

    all_losses = []

    for epoch in range(epochs):
        heatmap_predictor.train()
        total_loss = 0
        for batch_idx, (images, hand_heatmaps_gt, object_heatmaps_gt,
                        bbox_hand_norm, bbox_obj_norm) in enumerate(dataloader):

            images = images.to(device)
            hand_heatmaps_gt = hand_heatmaps_gt.to(device)
            object_heatmaps_gt = object_heatmaps_gt.to(device)
            bbox_hand_norm = bbox_hand_norm.to(device)
            bbox_obj_norm = bbox_obj_norm.to(device)

            with torch.no_grad():
                # FPN
                hand_feat, obj_feat = feature_extractor(images)

                # ROI crop
                hand_roi = roi_crop(hand_feat, bbox_hand_norm)
                obj_roi = roi_crop(obj_feat, bbox_obj_norm)

            # Forward
            pred_hand_heatmap, pred_obj_heatmap = heatmap_predictor(hand_roi, obj_roi)

            loss_hand = criterion(pred_hand_heatmap, hand_heatmaps_gt)
            loss_obj = criterion(pred_obj_heatmap, object_heatmaps_gt)
            loss = loss_hand + loss_obj

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        all_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}")

        # 保存权重
        torch.save({
            'model_state_dict': heatmap_predictor.state_dict(),
            'feature_extractor_state_dict': feature_extractor.state_dict()
        }, f"checkpoints/heatmap_epoch_{epoch+1}.pth")

    # 绘制 loss 曲线
    plot_loss_curve(all_losses, save_path="heatmap_train_loss.png")
    logger.info("热图预测器训练完成")

if __name__ == "__main__":
    data_root = "/mnt/d/deep_learning/dataset/Dex_YCB/OpenDataLab___DexYCB/raw/dex_ycb"
    os.makedirs("checkpoints", exist_ok=True)
    train_heatmap(data_root, batch_size=72, epochs=10, lr=1e-4)



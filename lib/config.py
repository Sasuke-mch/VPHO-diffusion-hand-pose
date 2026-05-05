import torch
from torchvision import transforms
from manopth.manolayer import ManoLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mano_layer_right = ManoLayer(
    mano_root='./mano', use_pca=False, ncomps=6,
    flat_hand_mean=True, side='right').to(device)
mano_layer_left = ManoLayer(
    mano_root='./mano', use_pca=False, ncomps=6,
    flat_hand_mean=True, side='left').to(device)
mano_layer = mano_layer_right  # fallback

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

YCB_CLASSES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box', 4: '005_tomato_soup_can',
    5: '006_mustard_bottle', 6: '007_tuna_fish_can', 7: '008_pudding_box', 8: '009_gelatin_box',
    9: '010_potted_meat_can', 10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block', 17: '037_scissors',
    18: '040_large_marker', 19: '051_large_clamp', 20: '052_extra_large_clamp', 21: '061_foam_brick'
}



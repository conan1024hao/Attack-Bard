import torch
from surrogates import (
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
    VisionTransformerFeatureExtractor,
)
from utils import get_list_image, save_list_images, save_image
from tqdm import tqdm
from attacks import SpectrumSimulationAttack, SSA_CommonWeakness

import os
import argparse

parser = argparse.ArgumentParser(description='Attack image encoder with misdescription')
parser.add_argument('--original_dir', type=str, default="/workspace/multimodal-sae-steering/images/LLaVA-Instruct-150K/original/dev", help='Directory containing original images')
parser.add_argument('--output_dir', type=str, default="/workspace/multimodal-sae-steering/images/LLaVA-Instruct-150K/attacked/SSA-CWA/dev", help='Directory to save attacked images')
args = parser.parse_args()

original_dir = args.original_dir
output_dir = args.output_dir
images = get_list_image(original_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = [i.unsqueeze(0) for i in images]

blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
vit = VisionTransformerFeatureExtractor().eval().cuda().requires_grad_(False)
models = [vit, blip, clip]


def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max = ssa_N * num_models
    count = count % max
    count = count // ssa_N
    return count


ssa_cw_loss = EnsembleFeatureLoss(models, ssa_cw_count_to_index, feature_loss=torch.nn.MSELoss())


attacker = SSA_CommonWeakness(
    models,
    epsilon=16 / 255,
    step_size=1 / 255,
    total_step=500,
    criterion=ssa_cw_loss,
)


id = 0
for i, x in enumerate(tqdm(images)):
    
    if os.path.exists(os.path.join(output_dir, f"{id}.png")):
        print(f"Image {id} already exists, skipping...")
        id += x.shape[0]
        continue
    
    x = x.cuda()
    ssa_cw_loss.set_ground_truth(x)
    adv_x = attacker(x, None)
    save_list_images(adv_x, output_dir, begin_id=id)
    id += x.shape[0]

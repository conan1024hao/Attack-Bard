import torch

from utils import get_list_image, save_list_images, save_image
from tqdm import tqdm

import os

original_dir = "/workspace/multimodal-sae-steering/images/Medical-Multimodal-Eval/original/train"
images = get_list_image(original_dir)
dir = "/workspace/multimodal-sae-steering/images/Medical-Multimodal-Eval/attacked/SSA-CWA/train"
if not os.path.exists(dir):
    os.makedirs(dir)

for i, img in enumerate(images):
    save_image(img, os.path.join(original_dir, f"{i}.png"))

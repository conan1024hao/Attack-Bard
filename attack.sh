conda activate minigptv310

python attack_img_encoder_misdescription.py \
    --original_dir /workspace/multimodal-sae-steering/images/LLaVA-Instruct-150K/original/dev \
    --output_dir /workspace/multimodal-sae-steering/images/LLaVA-Instruct-150K/attacked/SSA-CWA/dev

python attack_img_encoder_misdescription.py \
    --original_dir /workspace/multimodal-sae-steering/images/LLaVA-Instruct-150K/original/test \
    --output_dir /workspace/multimodal-sae-steering/images/LLaVA-Instruct-150K/attacked/SSA-CWA/test

echo "done"
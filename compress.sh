#!/bin/bash
#注意：推断带有平铺的高分辨率图像时建议进行颜色修正（如 DIV2K、CLIC 2020）。

CUDA_VISIBLE_DEVICES=1 python src/compress.py \
    --sd_path="stabilityai/sd-turbo" \
    --elic_path="weight/elic_official.pth" \
    --img_path="data/test/div2k/" \
    --rec_path="output/ft32/div2k/rec/" \
    --bin_path="output/ft32/div2k/bin/" \
    --codec_path="weight/stablecodec_ft32.pkl" \
    --color_fix 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import BytesIO
import torch
import os
from diffusers import AutoencoderKL, AutoPipelineForImage2Image
from diffusers.utils import load_image
from PIL import Image
from pathlib import Path
import random
from huggingface_hub import snapshot_download


def download_models():
    # Ignore files that we don't need to speed up download time.
    ignore = [
        "*.bin",
        "*.onnx_data",
        "*/diffusion_pytorch_model.safetensors",
    ]
    snapshot_download("stabilityai/sdxl-turbo", ignore_patterns=ignore)
    # https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl_turbo#speed-up-sdxl-turbo-even-more
    # vae is used for a inference speedup
    snapshot_download("madebyollin/sdxl-vae-fp16-fix", ignore_patterns=ignore)


download_models()

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    device_map="auto",
    variant="fp16",
    vae=AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
        device_map="auto",
    ),
)


def inference(image, prompt, num_iterations):
    init_image = Image.open(image).resize((512, 512))
    # based on trial and error we saw the best results with 3 inference steps
    # it had better generation results than 4,5,6 even though it's faster
    num_inference_steps = int(num_iterations)
    # note: anything under 0.5 strength gives blurry results
    strength = 0.999 if num_iterations == 2 else 0.65
    assert num_inference_steps * strength >= 1

    image = pipe(
        prompt,
        image=init_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=0.0,
        seed=42,
    ).images[0]

    image_name = os.path.basename(image)
    base_name, ext = os.path.splitext(image_name)
    prompt_words = prompt.split()[:5]
    prompt_part = "_".join(prompt_words)
    seed = random.randint(0, 10000)

    output_image = f"{prompt_part}_{base_name}_{seed}.png"
    image.save(output_image)


inference('frontend/src/lib/assets/abstract.png', 'person on the beach', 2)
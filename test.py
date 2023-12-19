#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from io import BytesIO
import torch
from diffusers import AutoencoderKL, AutoPipelineForImage2Image
from diffusers.utils import load_image
from PIL import Image
from pathlib import Path
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

# We execute a blank inference since there are objects that are lazily loaded that
# we want to start loading before an actual user query
pipe(
            "blank",
            image=Image.new("RGB", (800, 1280), (255, 255, 255)),
            num_inference_steps=1,
            strength=1,
            guidance_scale=0.0,
            seed=42,
        )


def inference(image, prompt, num_iterations):
        img_data_in = image.read()

        init_image = load_image(Image.open(
            BytesIO(img_data_in))).resize((512, 512))
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

        byte_stream = BytesIO()
        image.save(byte_stream, format="jpeg")
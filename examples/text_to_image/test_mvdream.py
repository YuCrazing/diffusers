from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import make_image_grid

import torch

model_path = "mvdream"
pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=True).to("cuda")

ckp = ""
if ckp != "":
    unet = UNet2DConditionModel.from_pretrained(
        model_path+f"/checkpoint-{ckp}", subfolder="unet"
    )
    pipeline.unet = unet


pipeline.to("cuda")

prompt = "yoda"
generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]
images = pipeline(prompt, generator=generator, num_images_per_prompt=4).images

# make_image_grid(images, rows=2, cols=2).show()

import os
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i in range(len(images)):
    image = images[i]
    image.save(f"{output_dir}/image-{i}-{ckp}-{prompt}.png")

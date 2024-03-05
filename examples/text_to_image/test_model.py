from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import make_image_grid

import torch

model_path = "sd-model-finetuned-1500-clean"
pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

# unet = UNet2DConditionModel.from_pretrained(
#     model_path+"/checkpoint-500", subfolder="unet"
# )
# pipeline.unet = unet

pipeline.to("cuda")

generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(4)]
images = pipeline("yoda", generator=generator, num_images_per_prompt=4).images


make_image_grid(images, rows=2, cols=2).show()

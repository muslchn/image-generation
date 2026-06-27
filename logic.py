import gc

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from PIL import Image, ImageFilter

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
DEFAULT_NEGATIVE_PROMPT = (
    "photorealistic, realistic, photograph, 3d render, messy, blurry, "
    "low quality, bad art, ugly, sketch, grainy, unfinished, chromatic aberration"
)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _generator(seed):
    return torch.Generator(device=_device()).manual_seed(int(seed))


@torch.inference_mode()
def load_models_cached():
    device = _device()
    dtype = _dtype()
    print(f"Loading models to {device}")

    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    pipe_txt2img.enable_attention_slicing()
    pipe_txt2img.enable_vae_slicing()

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    pipe_inpaint.enable_attention_slicing()
    pipe_inpaint.enable_vae_slicing()

    return pipe_txt2img, pipe_inpaint


def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory Flushed!")


def set_scheduler(pipe, scheduler_name):
    if scheduler_name == "Euler A":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError("scheduler_name must be one of: Euler A, DPM++, DDIM")
    return pipe


@torch.inference_mode()
def generate_image(pipe, prompt, neg_prompt, seed, steps, cfg, num_images=1, scheduler_name="Euler A"):
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("Prompt must not be empty.")

    pipe = set_scheduler(pipe, scheduler_name)
    num_images = max(1, min(int(num_images), 4))
    generators = [_generator(int(seed) + idx) for idx in range(num_images)]

    result = pipe(
        prompt=[prompt] * num_images,
        negative_prompt=[neg_prompt] * num_images,
        generator=generators,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
    )
    return result.images


@torch.inference_mode()
def run_inpainting(
    pipe,
    image,
    mask,
    prompt,
    strength=1.0,
    guidance_scale=12.0,
    num_inference_steps=50,
    seed=9,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
):
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("Inpainting prompt must not be empty.")
    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")
    if image.size != mask.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)
    if mask.getbbox() is None:
        raise ValueError("Draw a non-empty mask before running inpainting.")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=_generator(seed),
    )
    return result.images[0]


def prepare_outpainting(image, expand_pixels=128):
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    new_width = width + (expand_pixels * 2)
    new_height = height + (expand_pixels * 2)
    new_width -= new_width % 8
    new_height -= new_height % 8

    bg = image.resize((new_width, new_height), resample=Image.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=50))

    canvas = bg.copy()
    paste_x = (new_width - width) // 2
    paste_y = (new_height - height) // 2
    canvas.paste(image, (paste_x, paste_y))

    mask = Image.new("L", (new_width, new_height), 255)
    inner_box = Image.new("L", (width, height), 0)
    mask.paste(inner_box, (paste_x, paste_y))

    return canvas, mask

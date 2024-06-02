import torch
from diffusers import StableDiffusionPipeline, PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL

rand_seed = torch.manual_seed(42)
NUM_INFERENCE_STEPS = 10
GUIDANCE_SCALE = 0.75
HEIGHT = 512
WIDTH = 512

def create_pipeline(model_name):
    if torch.cuda.is_available():
        print("Using GPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
        ).to("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
        ).to("mps")
    else:
        print("Using CPU")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
        )

    return pipeline

def txt2img(prompt, pipeline, progress_callback=None):
    images = pipeline(
        prompt,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=rand_seed,
        num_images_per_request=1,
        height=HEIGHT,
        width=WIDTH,
    ).images

    return images[0]

def PixArtAlphaTxt2img(prompt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", use_safetensors=True)
    pipe.to(device)

    return pipe(prompt).images[0]
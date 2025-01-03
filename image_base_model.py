import torch
from diffusers import StableDiffusion3Pipeline

# Check if a CUDA-enabled GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)



# Generate an image
image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

# Save the generated image
image.save("sd3_hello_world.png")
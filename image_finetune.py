import torch
from diffusers import DiffusionPipeline
from lycoris import create_lycoris_from_weights


def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file


model_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
adapter_repo_id = 'jimchoi/simpletuner-lora'
adapter_filename = 'pytorch_lora_weights.safetensors'
adapter_file_path = download_adapter(repo_id=adapter_repo_id)
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)  # loading directly in bf16
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = ("Graceful anime blonde girl with cat ears, white dress, blue ribbon."
          " The Girl has yellow eyes, gentle smile, natural-looking, high heels that accentuate her legs"
          "The Girl stands next to white Porsche 718 on Paris bridge, Car with black rims, yellow calipers."
          "Eiffel Tower, golden cityscape. Serene, romantic, blue sky, sunset, sunlight on leaves.")
negative_prompt = 'blurry, cropped, ugly'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8

quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to(
    'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')  # the pipeline
# is already in its target precision level
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    generator=torch.Generator(
        device='cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(
        42),
    width=1280,
    height=1024,
    guidance_scale=3.0,
).images[0]
image.save("output.png", format="PNG")

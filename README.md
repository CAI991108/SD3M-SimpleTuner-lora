# Fine-Tuning Stable Diffusion 3 Medium with SimpleTuner

## Project Overview

This project explores the potential of fine-tuning the **Stable Diffusion 3 Medium (SD3M)** model using the **SimpleTuner** toolkit to generate higher quality and more diverse 2D concept art. The research focuses on understanding the technical differences between fine-tuning SD1.5/SDXL and SD3M, and leverages these insights to develop a robust fine-tuning methodology that can be applied across various artistic domains.

## Key Features

- **Fine-Tuning SD3M**: We fine-tune the SD3M model using the SimpleTuner toolkit, which simplifies the process of customizing and fine-tuning pre-trained models.
- **Low Rank Adaptation (LoRA)**: We employ LoRA, a parameter-efficient fine-tuning technique, to adapt the model with limited computational resources.
- **Evaluation of Generative Art**: We evaluate the fine-tuned model's ability to generate high-quality and diverse concept art, comparing it with the baseline SD3M model.

## Installation

### Prerequisites

- Python 3.10 or 3.11
- Poetry (for dependency management)

### Setup
The [SimpleTuner SD3 Quickstart Guide](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/SD3.md)
provides a detailed guide on setting up the environment and fine-tuning the SD3M model using SimpleTuner.

Additionally, you might find the following instructions helpful:
1. There are built-in scripts to install the dependencies and set up the environment. (see files `poetry.lock` and `pyproject.toml`).
The following `tool.poetry.source` will accelerate the installation process for most cases. **Or simply use the file provided in this repository**.
   ```bash
    name = "mirrors"
    url = "https://to/your/mirror"
    priority = "primary"
   ```
2. Choose wisely for the network connection. The suggestion is to `export` the following environment variables:
   ```bash
   export HF_ENDPOINT=https://huggingface.com
   export WANDB_API_KEY=your_wandb_api_key_here
   ```
3. The experimental script `configure.py` provides interactive step-by-step configuration. However, one should manually modify the `multibackend.json`:
    ```bash
    "cache_dir_vae": "cache/vae/sd3/xxx",
    "instance_data_dir": "path/to/your/data",
    "cache_dir": "/cache/text/sd3/xxx",
    ```

## Example Usage of Generative Concept Art

### John Singer Sargent Style Portrait
An example fine-tuned model mincing the art style of John Singer Sargent. The model was trained on a dataset of John Singer Sargent's portraits and can generate images in a similar style.
please refer to my HuggingFace model card for detail implementation  [jimchoi/simpletuner-lora](https://huggingface.co/jimchoi/simpletuner-lora)

1. **Run the inference script**:
   ```bash
   python image_finetune.py --prompt "your_prompt_here"
   ```

2. **Adjust the prompt**:
   - For better results, adjust the prompt content to avoid overfitting or generating overly similar images.

## Research Limitations

- **Dataset Size and Diversity**: The dataset used for fine-tuning is relatively small, which may limit the model's generalizability.
- **Computational Resources**: Limited GPU resources constrained the batch size and training speed.
- **Evaluation Techniques**: The evaluation is primarily based on visual inspection, and more rigorous quantitative metrics could be applied.

## Future Work

- **Expand Dataset**: Use a larger and more diverse dataset to improve the model's generalization capabilities.
- **Multi-GPU Training**: Utilize multi-GPU environments to speed up the training process and handle larger datasets.
- **Quantitative Evaluation**: Implement more rigorous quantitative evaluation metrics for a comprehensive assessment of the model's performance.

## License
The project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- **Stability AI** for open-sourcing the SD3M model. The pre-trained SD3M model is available on Hugging Face: [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).
- **SimpleTuner** for providing an efficient fine-tuning toolkit. For more details on using SimpleTuner with SD3M, refer to the [SimpleTuner SD3 Quickstart Guide](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/SD3.md).

---

For more details, please refer to the [report](report.pdf) and the Hugging Face model card [jimchoi/simpletuner-lora](https://huggingface.co/jimchoi/simpletuner-lora).
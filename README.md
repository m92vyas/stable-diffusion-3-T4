# Run full Stable Diffusion 3 model on colab or T4 GPU with extended context length

The stable diffusion 3 [Hugging Face page](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#memory-optimisations-for-sd3) states

_"SD3 uses three text encoders, one if which is the very large T5-XXL model. This makes it challenging to run the model on GPUs with less than 24GB of VRAM, even when using fp16 precision."_

and gives some options like using quantize version of the T5 text encoder or dropping it. CPU offload does not work in the free version of colab and sequential offload to cpu takes long time to generate the image.

Use this repository to run the full sd3 model on colab or T4 gpu (details will be added in the coming days)

```python
pip install git+https://github.com/m92vyas/stable-diffusion-3-T4.git
```

```python
from sd3_t4.sd3_t4_pipeline import load_sd3_t4_pipeline, generate_sd3_t4_image
from huggingface_hub import login

login(token=<your hugging face token>)

load_sd3_t4_pipeline() #220 sec
```

```python
prompt = <your positive prompt>
negative_prompt = <your negative prompt>
generated_image = generate_sd3_t4_image(prompt, negative_prompt ,num_inference_steps=28)
```

# Run full Stable Diffusion 3 model on colab or T4 GPU with extended context length

The stable diffusion 3 [Hugging Face page](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#memory-optimisations-for-sd3) states

_"SD3 uses three text encoders, one of which is the very large T5-XXL model. This makes it challenging to run the model on GPUs with less than 24GB of VRAM, even when using fp16 precision."_

and gives some options like using quantize version of the T5 text encoder or dropping it. CPU offload does not work in the free version of colab and sequential offload to cpu takes long time to generate the image.

Good for us that colab or the T4 gpu has enough gpu memory to load all the three text encoders at once without any quantization and get text embeddings and then empty the gpu space enough to load the transformer and vae and perform next steps for the image generation.

So, the basic steps to prepare the pipeline will look like:
- load all the 3 text encoders with their tokenizer on the gpu.
- load the transformer and vae on the cpu.
- get text embeddings as all the encoders models are on gpu (with extended context length which allows to encode prompts with more than the 77 token limit of the CLIP encoders)
- to save time only move enough modules from text encoder 3 (T5) to cpu and so that the transformer and vae can be loaded onto the gpu.
- complete the image generation process.
- move text encoder 3 modules back to gpu and transformer, vae back to cpu to start a new inference step. 

The code in this repository transfers modules between devices in batches to avoid out of memory (OOM) error on the T4 machine without adding significant time to the inference step.

For implementing extended context length with prompt weighing i have taken the code from https://github.com/xhinker/sd_embed with some modifications to reduce memory usage like using torch.no_grad()

To add weights to your prompts refer to the above given repository. So now we can run the full stable diffusion 3 model with long prompts with weighing.

You will need minimum 12.5 gb of cpu ram which is provoded in colab and also on T4 machine like aws g4.dn.xlarge (16 gb cpu)

#### Installation

```python
pip install git+https://github.com/m92vyas/stable-diffusion-3-T4.git
```
#### Hugging Face login to access the sd3 model and pipeline loading

```python
from sd3_t4.sd3_t4_pipeline import load_sd3_t4_pipeline, generate_sd3_t4_image
from huggingface_hub import login

login(token=<your hugging face token>)

load_sd3_t4_pipeline() # it will take about 4 minutes to fully load the pipeline
```
#### Generate image on colab or T4

```python
prompt = <your positive prompt>
negative_prompt = <your negative prompt>
generated_image = generate_sd3_t4_image(prompt, negative_prompt ,num_inference_steps=28, guidance_scale=7) # you can add other sd3 parameters 
```

Note: The pipeline works well for lengthy prompts / long prompts but for very long prompts can result in OOM error.
In such cases try running the pipeline with little lower context length as warm up steps and try inferencing again using your long prompt.

Consider giving a star if this repo was of any help to you.

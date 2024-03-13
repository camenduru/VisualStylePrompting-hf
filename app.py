import torch
from pipelines.inverted_ve_pipeline import STYLE_DESCRIPTION_DICT, create_image_grid
import gradio as gr
import os, json

from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import AutoencoderKL
from random import randint
from utils import init_latent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

def memory_efficient(model):
    try:
        model.to(device)
    except Exception as e:
        print("Error moving model to device:", e)

    try:
        model.enable_model_cpu_offload()
    except AttributeError:
        print("enable_model_cpu_offload is not supported.")
    try:
        model.enable_vae_slicing()
    except AttributeError:
        print("enable_vae_slicing is not supported.")
    if device == 'cuda':
        try:
            model.enable_xformers_memory_efficient_attention()
        except AttributeError:
            print("enable_xformers_memory_efficient_attention is not supported.")

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch_dtype)
model = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype)

print("vae")
memory_efficient(vae)
print("SDXL")
memory_efficient(model)


# controlnet_scale, canny thres 1, 2 (2 > 1, 2:1, 3:1)

def parse_config(config):
    with open(config, 'r') as f:
        config = json.load(f)
    return config


def load_example_style():
    folder_path = 'assets/ref'
    examples = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png")):
            image_path = os.path.join(folder_path, filename)
            image_name = os.path.basename(image_path)
            style_name = image_name.split('_')[1]

            config_path = './config/{}.json'.format(style_name)
            config = parse_config(config_path)
            inf_object_name = config["inference_info"]["inf_object_list"][0]

            image_info = [image_path, style_name, inf_object_name, 1, 50]
            examples.append(image_info)

    return examples

def style_fn(image_path, style_name, content_text, output_number, diffusion_step=50):
    """

    :param style_name: 어떤 json 파일 부를거냐 ?
    :param content_text: 어떤 콘텐츠로 변화를 원하니 ?
    :param output_number: 몇개 생성할거니 ?
    :return:
    """
    config_path = './config/{}.json'.format(style_name)
    config = parse_config(config_path)

    inf_object = content_text
    inf_seeds = [randint(0, 10**10) for _ in range(int(output_number))]
    # inf_seeds = [i for i in range(int(output_number))]


    activate_layer_indices_list = config['inference_info']['activate_layer_indices_list']
    activate_step_indices_list = config['inference_info']['activate_step_indices_list']
    ref_seed = config['reference_info']['ref_seeds'][0]

    attn_map_save_steps = config['inference_info']['attn_map_save_steps']
    guidance_scale = config['guidance_scale']
    use_inf_negative_prompt = config['inference_info']['use_negative_prompt']

    style_name = config["style_name_list"][0]

    ref_object = config["reference_info"]["ref_object_list"][0]
    ref_with_style_description = config['reference_info']['with_style_description']
    inf_with_style_description = config['inference_info']['with_style_description']

    use_shared_attention = config['inference_info']['use_shared_attention']
    adain_queries = config['inference_info']['adain_queries']
    adain_keys = config['inference_info']['adain_keys']
    adain_values = config['inference_info']['adain_values']

    use_advanced_sampling = config['inference_info']['use_advanced_sampling']

    style_description_pos, style_description_neg = STYLE_DESCRIPTION_DICT[style_name][0], \
                                                   STYLE_DESCRIPTION_DICT[style_name][1]

    # Inference
    with torch.inference_mode():
        grid = None
        if ref_with_style_description:
            ref_prompt = style_description_pos.replace("{object}", ref_object)
        else:
            ref_prompt = ref_object

        if inf_with_style_description:
            inf_prompt = style_description_pos.replace("{object}", inf_object)
        else:
            inf_prompt = inf_object

        for activate_layer_indices in activate_layer_indices_list:

            for activate_step_indices in activate_step_indices_list:

                str_activate_layer, str_activate_step = model.activate_layer(
                    activate_layer_indices=activate_layer_indices,
                    attn_map_save_steps=attn_map_save_steps,
                    activate_step_indices=activate_step_indices, use_shared_attention=use_shared_attention,
                    adain_queries=adain_queries,
                    adain_keys=adain_keys,
                    adain_values=adain_values,
                )
                # ref_latent = model.get_init_latent(ref_seed, precomputed_path=None)
                ref_latent = init_latent(model, device_name=device, dtype=torch_dtype, seed=ref_seed)
                latents = [ref_latent]

                for inf_seed in inf_seeds:
                    # latents.append(model.get_init_latent(inf_seed, precomputed_path=None))
                    inf_latent = init_latent(model, device_name=device, dtype=torch_dtype, seed=inf_seed)
                    latents.append(inf_latent)

                latents = torch.cat(latents, dim=0)
                latents.to(device)

                images = model(
                    prompt=ref_prompt,
                    negative_prompt=style_description_neg,
                    guidance_scale=guidance_scale,
                    num_inference_steps=diffusion_step,
                    latents=latents,
                    num_images_per_prompt=len(inf_seeds) + 1,
                    target_prompt=inf_prompt,
                    use_inf_negative_prompt=use_inf_negative_prompt,
                    use_advanced_sampling=use_advanced_sampling
                )[0][1:]

                n_row = 1
                n_col = len(inf_seeds)  # 원본추가하려면 + 1

                # make grid
                grid = create_image_grid(images, n_row, n_col, padding=10)

        torch.cuda.empty_cache()

        return grid

description_md = """

### We introduce `Visual Style Prompting`, which reflects the style of a reference image to the images generated by a pretrained text-to-image diffusion model without finetuning or optimization (e.g., Figure N).
### 📖 [[Paper](https://arxiv.org/abs/2402.12974)] | ✨ [[Project page](https://curryjung.github.io/VisualStylePrompt)] | ✨ [[Code](https://github.com/naver-ai/Visual-Style-Prompting)]
### 🔥 [[w/ Controlnet ver](https://huggingface.co/spaces/naver-ai/VisualStylePrompting_Controlnet)]
---
### To try out our vanilla demo,
1. Choose a `style reference` from the collection of images below.
2. Enter the `text prompt`.
3. Choose the `number of outputs`.

### To achieve faster results, we recommend lowering the diffusion steps to 30.
### Enjoy ! 😄
"""

iface_style = gr.Interface(
    fn=style_fn,
    inputs=[
        gr.components.Image(label="Style Image"),
        gr.components.Textbox(label='Style name', visible=False),
        gr.components.Textbox(label="Text prompt", placeholder="Enter Text prompt"),
        gr.components.Textbox(label="Number of outputs", placeholder="Enter Number of outputs"),
        gr.components.Slider(minimum=50, maximum=50, step=10, value=50, label="Diffusion steps")
    ],
    outputs=gr.components.Image(type="pil"),
    title="🎨 Visual Style Prompting (default)",
    description=description_md,
    examples=load_example_style(),
)

iface_style.launch(debug=True)
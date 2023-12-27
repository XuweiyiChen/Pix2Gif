#%%
from __future__ import annotations

import math
import random
import sys
import json
import os
from argparse import ArgumentParser

import einops
import inflect
import gradio as gr
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
import cv2
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from consistencydecoder import ConsistencyDecoder

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

#%%
help_text = """
If you're not getting what you want, there may be a few reasons:
1. Is the image not changing enough? Your Image CFG weight may be too high. This value dictates how similar the output should be to the input. It's possible your edit requires larger changes from the original image, and your Image CFG weight isn't allowing that. Alternatively, your Text CFG weight may be too low. This value dictates how much to listen to the text instruction. The default Image CFG of 1.5 and Text CFG of 7.5 are a good starting point, but aren't necessarily optimal for each edit. Try:
    * Decreasing the Image CFG weight, or
    * Incerasing the Text CFG weight, or
2. Conversely, is the image changing too much, such that the details in the original image aren't preserved? Try:
    * Increasing the Image CFG weight, or
    * Decreasing the Text CFG weight
3. Increasing the number of steps sometimes improves results.
4. Do faces look weird? The Stable Diffusion autoencoder has a hard time with faces that are small in the image. Try:
    * Cropping the image so the face takes up a larger portion of the frame.
"""


example_instructions = [
    "Make it a picasso painting",
    "as if it were by modigliani",
    "convert to a bronze statue",
    "Turn it into an anime.",
    "have it look like a graphic novel",
    "make him gain weight",
    "what would he look like bald?",
    "Have him smile",
    "Put him in a cocktail party.",
    "move him at the beach.",
    "add dramatic lighting",
    "Convert to black and white",
    "What if it were snowing?",
    "Give him a leather jacket",
    "Turn him into a cyborg!",
    "make him wear a beanie",
]

#%%
def number_to_words(number):  
    p = inflect.engine()  
    return p.number_to_words(number)

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]]), cond["c_crossattn"][1]],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]]), torch.cat([cond["c_concat"][1], cond["c_concat"][1], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def pil_images_to_gif(pil_images, cap, fps):    
    # Save the PIL images to a GIF  
    dur = 1000//fps
    output_gif = f"viz/all-cat_per_2_e3_{cap}.gif"  
    pil_images[0].save(output_gif, save_all=True, append_images=pil_images[1:], optimize=False, duration=dur, loop=0)
    return output_gif

#%%
config_file = "configs/generate.yaml"
ckpt = "/home/t-hkandala/MSRR/model/instruct-p2p/motioninterval-tgif-all_cat-only_per/train_motioninterval-tgif-all_cat-only_per/checkpoints/trainstep_checkpoints/epoch=000007-step=000023999.ckpt"
# ckpt = "/home/t-hkandala/MSRR/model/instruct-p2p/motioninterval-tgif-per-2_20-forward/train_motioninterval-tgif-per-2_20-forward/checkpoints/trainstep_checkpoints/epoch=000001-step=000003999.ckpt"
vae_ckpt = None
config = OmegaConf.load(config_file)
model = load_model_from_config(config, ckpt, vae_ckpt)
model.eval().cuda()
decoder_consistency = ConsistencyDecoder(device="cuda:0") # Model size: 2.49 GB
model_wrap = K.external.CompVisDenoiser(model)
model_wrap_cfg = CFGDenoiser(model_wrap)
null_token = model.get_learned_conditioning([""])
# resolution = 256

def generate(
    input_image: Image.Image,
    flow_image: Image.Image,
    instruction: str,
    interval: int,
    steps: int,
    seed: int,
    text_cfg_scale: float,
    image_cfg_scale: float,
):
    resolution = 256
    scale_factor = 0.18215

    width, height = input_image.size
    # factor = resolution / max(width, height)
    # factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    # width = int((width * factor) // 64) * 64
    # height = int((height * factor) // 64) * 64
    input_image = input_image.resize((resolution, resolution))
    flow_image = flow_image.resize((resolution, resolution))
    # input_image = ImageOps.fit(input_image, (resolution, resolution), method=Image.Resampling.LANCZOS)

    if instruction == "":
        return [input_image, seed]
    else:
        word_interval = number_to_words(interval)
        instruction = instruction + ' The optical flow is ' + word_interval + '. {}'.format(interval)

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([" ".join(instruction.split(" ")[:-1])]), interval]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        flow_image = 2 * torch.tensor(np.array(flow_image)).float() / 255 - 1
        flow_image = rearrange(flow_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode(), model.encode_first_stage(flow_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": text_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        z = 1. / scale_factor * z
        x = decoder_consistency(z)
        x = x[0].cpu().numpy()
        x = (x + 1.0) * 127.5
        x = x.clip(0, 255).astype(np.uint8)
        edited_image = Image.fromarray(x.transpose(1, 2, 0))
        edited_image = edited_image.resize((width, height))
        # x = model.decode_first_stage(z)
        # x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        # x = 255.0 * rearrange(x, "1 c h w -> h w c")
        # edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        return [seed, text_cfg_scale, image_cfg_scale, edited_image]
        # return edited_image

def generate_video(
    input_image: Image.Image,
    instruction: str,
    steps: int,
    seed: int,
    text_cfg_scale: float,
    image_cfg_scale: float,
):
    img_list = []
    cap = "-".join(instruction.split(" "))
    output_dir = "/home/t-hkandala/MSRR/eval/all_cat_per/new/"
    inter_img = input_image
    output_name = f"{cap}_{steps}_{seed}_{text_cfg_scale}_{image_cfg_scale}"
    # cfg_motion = [2,4,6,8,11,13,16,19,22,24,27,29,32,34,36,39]
    # cfg_motion = [2,4,6,8,11,13,16,19]
    cfg_motion = [2,3,4,5,6,8,9,10,11,12,13,14,16,17,19]
    for i in cfg_motion:
        inter_img = generate(input_image, inter_img, instruction, i, steps, seed, text_cfg_scale, image_cfg_scale)[-1]
        img_list.append(inter_img)
    # input_image = inter_img
    # for i in cfg_motion:
    #     inter_img = generate(input_image, inter_img, instruction, i, steps, seed, text_cfg_scale, image_cfg_scale)[-1]
    #     img_list.append(inter_img)    
    output_video = pil_images_to_gif(img_list, output_name, 8)
    output_path = os.path.join(output_dir, output_video)
    return [seed, text_cfg_scale, image_cfg_scale, output_path]

#%%
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    
    with gr.Row():
        instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil", interactive=True)
        edited_image = gr.Image(label=f"Generated GIF", type="pil", interactive=False)
        input_image.style(height=1024, width=1024)
        edited_image.style(height=1024, width=1024)

    with gr.Row():
        steps = gr.Number(value=100, precision=0, label="Steps", interactive=True)
        seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True)
        text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=True)
        image_cfg_scale = gr.Number(value=2, label=f"Image CFG", interactive=True)

    with gr.Row():
        generate_video_button = gr.Button("Generate GIF")

    gr.Markdown(help_text)

    generate_video_button.click(
        fn=generate_video,
        inputs=[
            input_image,
            instruction,
            steps,
            seed,
            text_cfg_scale,
            image_cfg_scale,
        ],
        outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
    )

demo.queue(concurrency_count=1)
demo.launch(share=True)
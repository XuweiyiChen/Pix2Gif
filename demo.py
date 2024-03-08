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

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]]), cond["c_crossattn"][1]],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond)
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="pix2gif.ckpt", type=str)
    parser.add_argument("--vae_ckpt", default=None, type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().to(device="cuda:0")
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    resolution = 256

    def number_to_words(number):  
        p = inflect.engine()  
        return p.number_to_words(number)

    def pil_images_to_gif(pil_images, filename, fps):    
        dur = 1000//fps
        output_gif = f"{filename}.gif"  
        pil_images[0].save(output_gif, save_all=True, append_images=pil_images[1:], optimize=False, duration=dur, loop=0)
        return output_gif

    def generate(
        input_image: Image.Image,
        instruction: str,
        interval: int,
        steps: int,
        seed: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
    ):
        width, height = input_image.size
        input_image = input_image.resize((resolution, resolution))

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
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

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
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image = edited_image.resize((width, height))

            return edited_image

    def generate_gif(
        input_image: Image.Image,
        instruction: str,
        cfg_motion: str,
        steps: int,
        seed: int,
        text_cfg_scale: float,
        image_cfg_scale: float,
    ):  
        if instruction == "":
            raise gr.Error("Input caption is missing!")
        if instruction[-1] != ".":
            instruction = instruction + "."
        img_list = []
        cap = "-".join(instruction.split(" "))
        output_dir = "output/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        inter_img = input_image
        output_name = f"{cap}_{steps}_{seed}_{text_cfg_scale}_{image_cfg_scale}"
        cfg_motion = [int(num.strip()) for num in cfg_motion.split(',')]
        for motion in cfg_motion:
            inter_img = generate(input_image, instruction, motion, steps, seed, text_cfg_scale, image_cfg_scale)
            img_list.append(inter_img)
        output_path = os.path.join(output_dir, output_name)
        output_path = pil_images_to_gif(img_list, output_path, 4)
        return output_path

    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        
        with gr.Row():
            instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)
            cfg_motion = gr.Textbox(value="2,4,6,8,11,13,16,19", lines=1, placeholder="Enter numbers separated by commas (e.g., 2,4,6,8,10)", label="Motion List", interactive=True)

        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil", interactive=True, height=768, width=768)
            output_gif = gr.Image(label=f"Generated GIF", type="pil", interactive=False, height=768, width=768)

        with gr.Row():
            steps = gr.Number(value=100, precision=0, label="Steps", interactive=True)
            seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True)
            text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=True)
            image_cfg_scale = gr.Number(value=1.8, label=f"Image CFG", interactive=True)

        with gr.Row():
            generate_gif_button = gr.Button("Generate GIF")

        generate_gif_button.click(
            fn=generate_gif,
            inputs=[
                input_image,
                instruction,
                cfg_motion,
                steps,
                seed,
                text_cfg_scale,
                image_cfg_scale,
            ],
            outputs=output_gif,
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()
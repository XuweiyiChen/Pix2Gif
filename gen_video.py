#%%
from __future__ import annotations

import math
import random
import sys
import json
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
from tqdm import tqdm
from consistencydecoder import ConsistencyDecoder

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

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
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
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

def pil_images_to_video(pil_images, image_name, cap, fps=30):  
    # Get the size of the first image  
    width, height = pil_images[0].size  
    output_video = f"viz0/{image_name}_{cap}.mp4"
  
    # Initialize the VideoWriter object  
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))  
  
    # Write each PIL image to the video  
    for pil_image in pil_images:  
        # Convert the PIL image to an OpenCV image (BGR format)  
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  
        video.write(cv_image)  
  
    # Release the video object  
    video.release()

#%%
config_file = "configs/generate.yaml"
ckpt = "/home/t-hkandala/MSRR/model/instruct-p2p/motioninterval-tgif-all_cat-only_per/train_motioninterval-tgif-all_cat-only_per/checkpoints/trainstep_checkpoints/epoch=000003-step=000011999.ckpt"
vae_ckpt = None
config = OmegaConf.load(config_file)
model = load_model_from_config(config, ckpt, vae_ckpt)
model.eval().cuda()
decoder_consistency = ConsistencyDecoder(device="cuda:0") # Model size: 2.49 GB
model_wrap = K.external.CompVisDenoiser(model)
model_wrap_cfg = CFGDenoiser(model_wrap)
null_token = model.get_learned_conditioning([""])
resolution = 256

#%%
def load_data(image_name):
    if image_name == '':
        with open("/home/t-hkandala/MSRR/captioning-datasets/TGIF/data/train_val_dataset_motion_2_20_reduced.json", "r") as f:
            data = json.load(f)
        idx = random.randint(0, len(data["val"]))
        return data["val"][idx]
    else:
        return image_name, "", "", 0

def load_example_model(
    img_name: str
):
    image_name, _, caption, _ = load_data(img_name)
    image = Image.open(image_name).convert("RGB")
    image = image.resize((resolution, resolution))
    # image = ImageOps.fit(image, (resolution, resolution), method=Image.Resampling.LANCZOS)

    return image, caption, image_name

def generate(
    input_image: Image.Image,
    instruction: str,
    interval: int,
    steps: int,
    seed: int,
    text_cfg_scale: float,
    image_cfg_scale: float,
):
    # seed = random.randint(0, 100000) if randomize_seed else seed
    resolution = 256
    scale_factor = 0.18215

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
        z = 1. / scale_factor * z
        x = decoder_consistency(z)
        x = x[0].cpu().numpy()
        x = (x + 1.0) * 127.5
        x = x.clip(0, 255).astype(np.uint8)
        edited_image = Image.fromarray(x.transpose(1, 2, 0))
        # x = model.decode_first_stage(z)
        # x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        # x = 255.0 * rearrange(x, "1 c h w -> h w c")
        # edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        # return [seed, text_cfg_scale, image_cfg_scale, edited_image]
        return edited_image

def generate_video(
    steps: int,
    seed: int,
    text_cfg_scale: float,
    image_cfg_scale: float,
):
    img_list = []
    image, caption, image_name = load_example_model('')
    cap = "-".join(caption.split(" "))
    img_name = "-".join(image_name.split("/")[-2:])
    cfg_motion = [2,3,5,7,8,10,11,12,13,15,16,17,18,19]
    for i in cfg_motion:
        inter_img = generate(image, caption, i, steps, seed, text_cfg_scale, image_cfg_scale)
        img_list.append(inter_img)
    pil_images_to_video(img_list, img_name, cap, 4)

#%%
for i in tqdm(range(20)):
    generate_video(100, 1603, 7.5, 3.0)
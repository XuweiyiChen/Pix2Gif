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
import re
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from dreamsim import dreamsim
from consistencydecoder import ConsistencyDecoder
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import Resize
sys.path.append("./stable_diffusion")

from calculate_fvd import calculate_fvd
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

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="/home/t-hkandala/MSRR/model/instruct-p2p/motioninterval-tgif-per-2_20-forward/train_motioninterval-tgif-per-2_20-forward/checkpoints/trainstep_checkpoints/epoch=000006-step=000020999.ckpt", type=str)
    parser.add_argument("--vae_ckpt", default=None, type=str)
    parser.add_argument("--data_folder", default="/home/t-hkandala/MSRR/UCF-101", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().to(device="cuda:0")
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    resolution = 256
    model_ds, preprocess_ds = dreamsim(pretrained=True)

    def pic_metric(
        ref_img: Image.Image, 
        gen_list: list[torch.Tensor]
    ) -> float:
        ref_img = preprocess_ds(ref_img).to(device="cuda:0")
        cumm_metric = 0
        for img in gen_list:
            img = preprocess_ds(Image.fromarray(np.array(img.cpu()))).to(device="cuda:0")
            distance = model_ds(ref_img, img)
            cumm_metric += (1-distance)
        avg_metric = cumm_metric / len(gen_list)
        return avg_metric

    def number_to_words(
        number: int,
    ) -> str:  
        p = inflect.engine()  
        return p.number_to_words(number)

    def pil_images_to_gif(
        pil_images: list[Image.Image],
        filename: str,
        fps: int
    ) -> Image.Image:    
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
    ) -> torch.Tensor:
        scale_factor = 0.18215
        resolution = 256

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
            edited_image = x.type(torch.uint8)
            return edited_image

    def generate_gif(
        input_image: Image.Image,
        instruction: str,
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
        inter_img = input_image
        output_name = f"{cap}_{steps}_{seed}_{text_cfg_scale}_{image_cfg_scale}"
        cfg_motion = [2,3,4,5,6,8,9,11,12,13,14,15,16,17,18,19]
        for motion in cfg_motion:
            inter_img = generate(input_image, instruction, motion, steps, seed, text_cfg_scale, image_cfg_scale)
            img_list.append(inter_img)
        return img_list
    
    def extract_frames(
        video_path,
        start_frame=1,
        stride=3,
        num_frames=16
    ):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            return []
        
        # List to hold the extracted frames
        frames_list = []
        
        # Read frames from the video
        frame_idx = 0  # Initialize frame counter
        while True:
            ret, frame = cap.read()
            
            # Break the loop if no frame is read
            if not ret:
                break
            
            # Check if the current frame is one we want to extract
            if frame_idx >= start_frame and (frame_idx - start_frame) % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
                frames_list.append(frame)
            
            # Check if we have extracted the desired number of frames
            if len(frames_list) >= num_frames:
                break
            
            frame_idx += 1
        
        # Release the video capture object
        cap.release()
        
        return frames_list

    def resize(image, resize_length):
        image_cropped = image.resize((resize_length,resize_length))        
        return image_cropped

    def get_video(folder_path):
        # List to store all .avi file paths
        avi_files_list = []

        # Walk through the directory tree
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Check if the file is an .avi file
                if file.lower().endswith('.avi'):
                    # Add the full path to the list
                    avi_files_list.append(os.path.join(root, file))

        # Check if we have at least 2048 .avi files
        if len(avi_files_list) < 2048:
            print("There are fewer than 2048 .avi files in the directory.")
            selected_avi_files = avi_files_list  # Use all available .avi files
        else:
            # Randomly select 2048 .avi files
            selected_avi_files = random.sample(avi_files_list, 2048)
        return selected_avi_files

    def get_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        return frame_rgb

    device = torch.device("cuda")
    vid_list = get_video(args.data_folder)
    list_vid_real = []
    list_vid_gen = []
    pbar = tqdm(total=len(vid_list))
    for idx, vid in enumerate(vid_list):
        img_list_real = []
        frame_list = extract_frames(vid)
        resize_frame_list = [resize(Image.fromarray(img), 256) for img in frame_list]
        tensor_list = [torch.tensor(np.array(img)).unsqueeze(0) for img in resize_frame_list]
        img_cat_real = torch.cat(tensor_list)
        img_cat_real = img_cat_real.permute(0,3,1,2).unsqueeze(0)
        if img_cat_real.shape[1] != 16:
            continue
        list_vid_real.append(img_cat_real)
        pbar.update(1)
    pbar.close()
    ucf_vid_cat = torch.cat(list_vid_real)/255.0
    # torch.save(ucf_vid_cat, "output/ucf_cat.pt")

    pbar = tqdm(total=len(vid_list))
    pic_score_list = []
    clip_sim_score_list = []
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    for vid in vid_list:
        first_img = Image.fromarray(get_frame(vid))
        caption = re.sub(r'(?<!^)(?=[A-Z])', ' ', vid.split("/")[-2])
        img_list_gen = generate_gif(first_img, caption, 50, 1371, 7.5, 2)

        pic_score = pic_metric(first_img, img_list_gen)
        pic_score_list.append(pic_score.item())
        
        img_list_gen = [torch.tensor(img).unsqueeze(0) for img in img_list_gen]
        img_cat_gen = torch.cat(img_list_gen)
        img_cat_gen = img_cat_gen.permute(0,3,1,2)
        cap_list = [caption for i in range(16)]

        clip_sim_score = metric(img_cat_gen, cap_list)
        clip_sim_score_list.append(clip_sim_score.item())
        
        list_vid_gen.append(img_cat_gen.unsqueeze(0))
        pbar.update(1)
    pbar.close()
    pix2gif_vid_cat = torch.cat(list_vid_gen)/255.0
    # torch.save(pix2gif_vid_cat, "output/pix2gif_cat.pt")


    pic_score_avg = sum(pic_score_list)/len(pic_score_list)
    clip_sim_score_avg = sum(clip_sim_score_list)/len(clip_sim_score_list)
    return ucf_vid_cat, pix2gif_vid_cat, pic_score_avg, clip_sim_score_avg

if __name__ == "__main__":
    msr_vtt, pix2gif, pic, clip_sim = main()
    device = torch.device("cuda")

    result = {}
    result['fvd'] = calculate_fvd(msr_vtt, pix2gif, device, method='styleganv')
    result['fvd_2'] = calculate_fvd(msr_vtt, pix2gif, device, method='videogpt')
    result['pic'] = pic
    result['clip_sim'] = clip_sim
    print(json.dumps(result, indent=4))
    with open("result_eval.json","w") as f:
        json.dump(result, f)
from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from einops import rearrange
import multiprocessing  
import os  
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import inflect

import json
import matplotlib.pyplot as plt
import seaborn
import yaml
from pathlib import Path

sys.path.append("./")

from clip_similarity_video import ClipSimilarity
from edit_dataset import EditDatasetMotionEval

sys.path.append("./stable_diffusion")

from ldm.util import instantiate_from_config

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

class ImageEditor(nn.Module):
    def __init__(self, config, ckpt, vae_ckpt=None):
        super().__init__()
        
        config = OmegaConf.load(config)
        self.model = load_model_from_config(config, ckpt, vae_ckpt)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])

    def forward(
        self,
        image: torch.Tensor,
        edit: str,
        interval: int,
        scale_txt: float = 7.5,
        scale_img: float = 1.0,
        steps: int = 100,
    ) -> torch.Tensor:
        assert image.dim() == 3
        assert image.size(1) % 32 == 0
        assert image.size(2) % 32 == 0
        word_interval = number_to_words(interval)
        edit = " ".join(edit.split(" ")[:-2]) # removed [space]four.[space]4
        edit = f"{edit} {word_interval}."
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {
                "c_crossattn": [self.model.get_learned_conditioning([edit]), interval],
                "c_concat": [self.model.encode_first_stage(image[None]).mode()],
            }
            uncond = {
                "c_crossattn": [self.model.get_learned_conditioning([""])],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])],
            }
            extra_args = {
                "uncond": uncond,
                "cond": cond,
                "image_cfg_scale": scale_img,
                "text_cfg_scale": scale_txt,
            }
            sigmas = self.model_wrap.get_sigmas(steps)
            x = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            x = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(x)[0]
            return x

def compute_metrics(config,
                    model_path, 
                    vae_ckpt,
                    data_path,
                    output_path, 
                    scales_img, 
                    scales_txt,
                    scales_motion,
                    num_samples = 100, 
                    split = "val", 
                    steps = 50, 
                    res = 256, 
                    seed = 1371):
    editor = ImageEditor(config, model_path, vae_ckpt).cuda()
    clip_similarity = ClipSimilarity().cuda()

    with open(config, 'r') as f:
        gen_config = yaml.safe_load(f)

    data_path = gen_config['data']['params']['validation']['params']['path']

    outpath = Path(output_path, f"all_cat_per_video_n={num_samples}_p={split}_s={steps}_r={res}_e={seed}.jsonl")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for scale_txt in scales_txt:
        for scale_img in scales_img:
            dataset = EditDatasetMotionEval(
                    path=data_path, 
                    split=split, 
                    res=res
                    )
            assert num_samples <= len(dataset)
            print(f'Processing t={scale_txt}, i={scale_img}')
            torch.manual_seed(seed)
            perm = torch.randperm(len(dataset))
            count = 0
            i = 0

            sim_video_0_avg = 0
            sim_video_1_avg = 0
            flow_avg = []
            flow_img_avg = []
            count = 0

            pbar = tqdm(total=num_samples)
            while count < num_samples:
                
                idx = perm[i].item()
                sample = dataset[idx]
                i += 1

                gen_list = []

                for scale_motion in scales_motion:
                    gen = editor(sample["image_0"].cuda(), sample["edit"], interval=scale_motion, scale_txt=scale_txt, scale_img=scale_img, steps=steps)
                    gen_list.append(gen[None].cuda())

                sample["edit"] = " ".join(sample["edit"].split(" ")[:-1])

                sim_video_0, sim_video_1, optical_flow, optical_flow_img = clip_similarity(
                    sample["image_0"][None].cuda(), sample["image_1"][None].cuda(), gen_list, [sample["edit"]]
                )
                sim_video_0_avg += sim_video_0.item()
                sim_video_1_avg += sim_video_1.item()
                flow_avg.append(optical_flow)
                flow_img_avg.append(optical_flow_img)
                count += 1
                pbar.update(count)
            pbar.close()

            sim_video_0_avg /= count
            sim_video_1_avg /= count
            flow_avg = np.array(flow_avg)
            flow_avg = np.mean(flow_avg, axis=0)
            flow_img_avg = np.array(flow_img_avg)
            flow_img_avg = np.mean(flow_img_avg, axis=0)

            with open(outpath, "a") as f:
                f.write(f"{json.dumps(dict(sim_video_0=sim_video_0_avg, sim_video_1=sim_video_1_avg, flow=flow_avg.tolist(), flow_img=flow_img_avg.tolist(), num_samples=num_samples, split=split, scale_txt=scale_txt, scale_img=scale_img, steps=steps, res=res, seed=seed))}\n")
    return outpath

# def plot_metrics(metrics_file, output_path):
    
#     with open(metrics_file, 'r') as f:
#         data = [json.loads(line) for line in f]
        
#     plt.rcParams.update({'font.size': 11.5})
#     seaborn.set_style("darkgrid")
#     plt.figure(figsize=(20.5* 0.7, 10.8* 0.7), dpi=200)

#     x = [d["sim_gen"] for d in data]
#     y = [d["sim_image"] for d in data]

#     plt.plot(x, y, marker='o', linewidth=2, markersize=4)

#     plt.xlabel("CLIP Text-Image Direction Similarity", labelpad=10)
#     plt.ylabel("CLIP Image Similarity", labelpad=10)

#     plt.savefig(Path(output_path) / Path("plot.pdf"), bbox_inches="tight")

def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--output_path", default="analysis/", type=str)
    parser.add_argument("--ckpt", default="/home/t-hkandala/MSRR/model/instruct-p2p/motioninterval-tgif-all_cat-only_per/train_motioninterval-tgif-all_cat-only_per/checkpoints/trainstep_checkpoints/epoch=000007-step=000023999.ckpt", type=str)
    parser.add_argument("--dataset", default="data/clip-filtered-dataset/", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    args = parser.parse_args()

    # scales_img = [2.2, 2.4, 2.6, 2.8, 3.0]
    scales_img = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    scales_txt = [7.5]
    scales_motion = [2, 4, 6, 8, 11, 14, 17, 19]
    
    metrics_file = compute_metrics(
            args.config,
            args.ckpt, 
            args.vae_ckpt,
            args.dataset, 
            args.output_path, 
            scales_img, 
            scales_txt,
            scales_motion,
            steps = args.steps,
            )
    
    # plot_metrics(metrics_file, args.output_path)

if __name__ == "__main__":
    main()
  
# def worker(scale_img, gpu_id):  
#     # Set environment variable to select specific GPU  
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
      
#     # You may want to adjust batch size or other settings based on the GPU  
#     scales_txt = [7.5]  
#     scales_motion = [2, 4, 6, 8, 11, 14, 17, 19]  
#     args = ArgumentParser().parse_args(["--resolution", "256", "--steps", "100", "--config", "configs/generate.yaml", "--output_path", "analysis/", "--ckpt", "/mnt/model/instruct-p2p/motioninterval-tgif-per-all_concat/train_motioninterval-tgif-per-all_concat/checkpoints/trainstep_checkpoints/epoch=000004-step=000013999.ckpt", "--dataset", "data/clip-filtered-dataset/"])  
      
#     metrics_file = compute_metrics(  
#         args.config,  
#         args.ckpt,   
#         args.vae_ckpt,  
#         args.dataset,   
#         args.output_path,   
#         [scale_img],  # only one scale_img value  
#         scales_txt,  
#         scales_motion,  
#         steps=args.steps,  
#     )  
#     # plot_metrics(metrics_file, args.output_path)  
  
# if __name__ == "__main__":  
#     scales_img = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]  
#     num_gpus = 16  # get number of available GPUs  
  
#     # Create a process for each GPU  
#     processes = []  
#     for i, scale_img in enumerate(scales_img):  
#         p = multiprocessing.Process(target=worker, args=(scale_img, i % num_gpus))  
#         p.start()  
#         processes.append(p)  
  
#     # Wait for all processes to finish  
#     for p in processes:  
#         p.join()  

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import inflect
import os
import random
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

def number_to_words(number):  
    p = inflect.engine()  
    return p.number_to_words(number)

class EditDatasetMotionInterval(Dataset):
    """
    Temporal Editing with Interval filtered by motion
    """
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.1),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "train_val_dataset_motion_2_20_reduced.json")) as f:
            self.filt_video = json.load(f)

        self.filt_video = self.filt_video[split]

    def __len__(self) -> int:
        return len(self.filt_video)

    def __getitem__(self, i: int) -> dict[str, Any]:
        image_0, image_1, prompt, motion = self.filt_video[i]

        interval = int(motion)
        word_interval = number_to_words(interval)
        prompt = prompt + ' The optical flow is ' + word_interval + '. {}'.format(interval)

        image_0 = Image.open(image_0)
        image_1 = Image.open(image_1)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)
        return dict(edited=image_1.float(), edit=dict(c_concat=image_0.float(), c_crossattn=prompt))

class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "test",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        self.path = path
        self.res = res

        with open(Path(self.path, "test_dataset_motion.json")) as f:   # change to indicate test file path
            self.test_files = json.load(f)

        self.test_files = self.test_files[split]

    def __len__(self) -> int:
        return len(self.test_files)

    def __getitem__(self, i: int) -> dict[str, Any]:
        image_0, prompt = self.test_files[i]
        
        # Convert to word 
        interval = random.randint(1, 10)
        word_interval = number_to_words(interval)
        prompt = prompt + ' The optical flow is ' + word_interval + '. {}'.format(interval)

        image_0 = Image.open(image_0)

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, edit=prompt)

class EditDatasetMotionEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "val",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        # assert sum(splits) == 1
        self.path = path
        self.res = res

        with open(Path(self.path, "test_val_dataset_motion.json")) as f:   # change to indicate test file path
            self.test_files = json.load(f)

        self.test_files = self.test_files[split]

    def __len__(self) -> int:
        return len(self.test_files)

    def __getitem__(self, i: int) -> dict[str, Any]:
        image_0, image_1, prompt, motion = self.test_files[i]
        
        # Convert to word 
        interval = int(motion)
        word_interval = number_to_words(interval)
        prompt = prompt + ' The optical flow is ' + word_interval + '. {}'.format(interval)

        image_0 = Image.open(image_0)
        image_1 = Image.open(image_1)

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, image_1=image_1, edit=prompt)

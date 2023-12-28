from __future__ import annotations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import XCLIPProcessor, XCLIPModel, AutoProcessor, CLIPVisionModelWithProjection
import numpy as np 
import cv2  

class ClipSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = "openai/clip-vit-base-patch32"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.model.eval().requires_grad_(False)

        x_model_name = "microsoft/xclip-base-patch32"
        self.x_processor = XCLIPProcessor.from_pretrained(x_model_name)
        self.x_model = XCLIPModel.from_pretrained(x_model_name)
        self.x_model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def repeat_and_concat(self, tensors):  
        # Repeat each tensor 4 times along dimension 0  
        repeated_tensors = [t.repeat(4, 1, 1, 1) for t in tensors]  
    
        # Concatenate the repeated tensors along dimension 0  
        result = torch.cat(repeated_tensors, dim=0)  
    
        return result 

    def denormalize_image(self, image):
        # Assume rearranged_tensor is your normalized tensor  
        rearranged_tensor = image.squeeze(0)  # This is just an example. Replace it with your actual tensor.  
        
        # Rearrange the dimensions back from (channels, height, width) to (height, width, channels)  
        tensor = rearrange(rearranged_tensor, "c h w -> h w c")  
        
        # Convert the normalized tensor back to the common range (0, 255)  
        image = ((tensor + 1) * 255 / 2).clamp(0, 255).unsqueeze(0)
        return image.int()
  
    def calculate_optical_flow_new(self, image_list):  
        # Initialize list to hold optical flow  
        optical_flow_magnitudes = []  
    
        # Iterate over all images in the list  
        for i in range(len(image_list) - 1):  
            # Load current and next image  
            curr_image = image_list[i][0].cpu().numpy().astype(np.uint8)  
            next_image = image_list[i + 1][0].cpu().numpy().astype(np.uint8)  
    
            # Convert images to grayscale  
            curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY)  
            next_image_gray = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)  
    
            # Calculate optical flow between current and next image  
            flow = cv2.calcOpticalFlowFarneback(curr_image_gray, next_image_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  
    
            magnitude, angle = cv2.cartToPolar(flow[:,:, 0], flow[:,:, 1])  
        
            # Compute the average magnitude across the entire image  
            avg_magnitude = np.mean(magnitude)  
    
            # Append average magnitude to list  
            optical_flow_magnitudes.append(avg_magnitude)  
  
        return optical_flow_magnitudes

    def calculate_optical_flow_old(self, image_list):  
        # Initialize list to hold optical flow  
        optical_flow_magnitudes = []  
    
        # Iterate over all images in the list  
        for i in range(len(image_list) - 1):  
            # Load current and next image  
            curr_image = image_list[i][0].cpu().numpy().astype(np.uint8)  
            next_image = image_list[i + 1][0].cpu().numpy().astype(np.uint8)  
    
            # Convert images to grayscale  
            curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY)  
            next_image_gray = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)  

            # Detect good features to track in the first frame  
            prev_pts = cv2.goodFeaturesToTrack(curr_image_gray, maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)  
        
            if prev_pts is None:  
                return 0  
        
            # Calculate optical flow using Lucas-Kanade method  
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(curr_image_gray, next_image_gray, prev_pts, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  
        
            # Compute the flow vectors  
            flow_vectors = curr_pts - prev_pts  
        
            # Calculate the magnitude of motion  
            magnitude = np.sqrt(np.sum(flow_vectors ** 2, axis=2))

            # filter out points having very slight optical flow vector (mostly noise or insignificant)
            mag_great_than_1 = [f for f in magnitude if f > 1]
        
            if len(mag_great_than_1) > 0:
                mean_mag = (sum(mag_great_than_1)/len(mag_great_than_1))[0]
            else:
                mean_mag = 0
            optical_flow_magnitudes.append(mean_mag)  
  
        return optical_flow_magnitudes 

    def calculate_optical_flow_image(self, image_list, image_0):  
        # Initialize list to hold optical flow  
        optical_flow_magnitudes = []  
        image = self.denormalize_image(image_0)
        curr_image = image[0].cpu().numpy().astype(np.uint8)
        curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_RGB2GRAY)  
    
        # Detect good features to track in the first frame  
        prev_pts = cv2.goodFeaturesToTrack(curr_image_gray, maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)  
    
        if prev_pts is None:  
            return 0  

        # Iterate over all images in the list  
        for i in range(len(image_list)):  
            # Load next image  
            next_image = image_list[i][0].cpu().numpy().astype(np.uint8)  
    
            # Convert images to grayscale  
            next_image_gray = cv2.cvtColor(next_image, cv2.COLOR_RGB2GRAY)  

            # Calculate optical flow using Lucas-Kanade method  
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(curr_image_gray, next_image_gray, prev_pts, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  
        
            # Compute the flow vectors  
            flow_vectors = curr_pts - prev_pts  
        
            # Calculate the magnitude of motion  
            magnitude = np.sqrt(np.sum(flow_vectors ** 2, axis=2))

            # filter out points having very slight optical flow vector (mostly noise or insignificant)
            mag_great_than_1 = [f for f in magnitude if f > 1]
        
            if len(mag_great_than_1) > 0:
                mean_mag = (sum(mag_great_than_1)/len(mag_great_than_1))[0]
            else:
                mean_mag = 0
            optical_flow_magnitudes.append(mean_mag)  
  
        return optical_flow_magnitudes 

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = self.denormalize_image(image)
        inputs = self.processor(images=image, return_tensors="pt").pixel_values
        inputs = inputs.to(self.device)
        self.model.to(self.device)
        outputs = self.model(inputs)
        return outputs.image_embeds

    def encode_video(self, images: list[torch.Tensor], text: list[str]):
        video = torch.cat(images, dim=0)
        inputs = self.x_processor(text=text, videos=list(video), return_tensors="pt", padding=True)
        # forward pass
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        self.x_model.to(self.device)
        outputs = self.x_model(**inputs)
        return outputs.video_embeds

    def forward(
        self, image_0: torch.Tensor, image_1: torch.Tensor, images: list[torch.Tensor], text: list[str]
    ):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        img_list = [self.denormalize_image(img) for img in images]
        video_features = self.encode_video(img_list, text)
        sim_video_0 = F.cosine_similarity(image_features_0, video_features)
        sim_video_1 = F.cosine_similarity(image_features_1, video_features)
        optical_flow = self.calculate_optical_flow_old(img_list)
        optical_flow_img = self.calculate_optical_flow_image(img_list, image_0)
        return sim_video_0, sim_video_1, optical_flow, optical_flow_img

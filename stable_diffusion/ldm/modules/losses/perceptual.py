import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models  
from torch.autograd import Function, Variable

def vgg19_4channel_input():  
    # Load the pre-trained VGG-19 model  
    vgg19 = models.vgg19(pretrained=True)
      
    # Modify the first layer to accept 4 channels  
    original_first_layer = vgg19.features[0]  
    modified_first_layer = nn.Conv2d(4, original_first_layer.out_channels, kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride, padding=original_first_layer.padding)  
      
    # Copy the pre-trained weights from the original first layer (excluding the weights of the 4th channel)  
    modified_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data  
      
    # Initialize the weights of the 4th channel using the mean of the pre-trained weights  
    modified_first_layer.weight.data[:, 3, :, :] = torch.mean(original_first_layer.weight.data, dim=1)  
      
    # Replace the first layer with the modified layer  
    vgg19.features[0] = modified_first_layer  
      
    return vgg19  
  

def resize_feature_map(feature_map, new_shape=(224, 224)):  
    """  
    Resizes a feature map of shape (bs, 4, 32, 32) to (bs, 4, 224, 224).  
      
    Args:  
        feature_map (Tensor): Input feature map of shape (bs, 4, 32, 32).  
        new_shape (tuple): The target shape for the feature map (height, width).  
          
    Returns:  
        Tensor: Output feature map of shape (bs, 4, 224, 224).  
    """  
    # Resize the feature map  
    resized_feature_map = torch.zeros(feature_map.shape[0], 4, *new_shape, device=feature_map.device, dtype=feature_map.dtype)  
    for i in range(feature_map.shape[0]):  
        resized_feature_map[i] = Resize(new_shape, interpolation=2)(feature_map[i])  
  
    return resized_feature_map
  
def vgg19_feature_extractor():  
    # Load the modified VGG-19 model with 4-channel input  
    vgg19_4channel = vgg19_4channel_input()
    vgg19 = vgg19_4channel.features  
    for param in vgg19.parameters():  
        param.requires_grad = False  
    return vgg19.eval()  
  
def perceptual_loss(input_img, warped_img, layer_indices=[0, 5, 10, 19, 28]): 
    feature_extractor = vgg19_feature_extractor().cuda()
    # feature_extractor = feature_extractor.half()
    input_features = []  
    warped_features = []  
      
    for i, layer in enumerate(feature_extractor):  
        input_img = layer(input_img)  
        warped_img = layer(warped_img)  
          
        if i in layer_indices:  
            input_features.append(input_img)  
            warped_features.append(warped_img)  
              
            if i == layer_indices[-1]:  # Break early if we have reached the last desired layer  
                break  
                  
    loss = sum([torch.mean((input_feat - warped_feat)**2) for input_feat, warped_feat in zip(input_features, warped_features)])  
    return loss  
  
# # Load the pre-trained VGG-19 feature extractor  
# feature_extractor = vgg19_feature_extractor()  
  
# # Assuming input_img and warped_img tensors have shape (bs, 4, 32, 32)  
# input_img = torch.randn(16, 4, 32, 32)  # Replace with your input image  
# warped_img = torch.randn(16, 4, 32, 32)  # Replace with your warped image  
  
# # Calculate the perceptual loss  
# loss = perceptual_loss(input_img, warped_img)  
# print("Perceptual Loss:", loss.item())  

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np

class soft_dice(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Args:
            smooth (float): A smoothing factor to avoid division by zero. Default: 1e-5.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, digits, y_true):
         """
        Args:
            digits (torch.Tensor): Raw logits of shape (N, C, H, W), it is before softmax
            y_true (torch.Tensor): Ground truth one-hot encoded tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The computed Soft Dice Loss (scalar).
        """
         y_pred = torch.softmax(digits, dim=1)
         intersection = torch.sum((y_pred*y_true), dim=(2, 3)) + self.smooth
         union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
         dice = 2*intersection/union
         return 1 - dice.mean()
    

class hard_dice(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Args:
            smooth (float): A smoothing factor to avoid division by zero. Default: 1e-5.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, digits, y_true):
         """
        Args:
            digits (torch.Tensor): Raw logits of shape (N, C, H, W), it is not processed through argmax of softmax
            y_true (torch.Tensor): Ground truth one-hot encoded tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The computed Soft Dice Loss (scalar).
        """
         y_pred = digits.argmax(dim=1)
         # convert y_pred to one hot 
         y_pred_one_hot = F.one_hot(y_pred, num_classes=y_true.shape[1]).permute(0, 3, 1, 2)
         intersection = torch.sum((y_pred_one_hot*y_true), dim=(2, 3)) + self.smooth
         union = y_pred_one_hot.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
         dice = 2*intersection/union
         return 1 - dice.mean()




def get_unique(mask_dir):
    """
    Extracts all unique RGB values from mask images in a directory and saves them to colormap.txt.
    If colormap.txt already exists, then the function will return immediately 
    Args:
        mask_dir (str): Path to the directory containing mask images.
    
    Output:
        Writes unique RGB values to 'colormap.txt' in the format 'R, G, B' per line.
    """
    if (os.path.exists("./colormap.txt")):
        return

    # Set to store unique RGB values (using tuples for hashability)
    unique_colors = set()

    # Iterate over all files in the mask directory
    for filename in sorted(os.listdir(mask_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # Open the image
            mask_path = os.path.join(mask_dir, filename)
            mask = Image.open(mask_path)

            # Convert to RGB if not already (in case of RGBA or other modes)
            if mask.mode != 'RGB':
                mask = mask.convert('RGB')

            # Convert to numpy array
            mask_array = np.array(mask)

            # Reshape to (num_pixels, 3) to get list of RGB values
            pixels = mask_array.reshape(-1, 3)

            # Add unique RGB tuples to the set
            for pixel in pixels:
                unique_colors.add(tuple(pixel))

    # Convert set to sorted list for consistent output
    unique_colors_list = sorted(list(unique_colors))

    # Write to colormap.txt
    with open('colormap.txt', 'w') as f:
        for r, g, b in unique_colors_list:
            f.write(f"{r}, {g}, {b}\n")

    print(f"Found {len(unique_colors_list)} unique RGB values. Saved to 'colormap.txt'.")



def tomasks(img: torch.Tensor, colormap: list) -> torch.Tensor:
    """
    Converts a tensor (C, H, W) to a mask (H, W) using a colormap.
    
    Args:
        img (torch.Tensor): Input tensor of shape (C, H, W), where C=1 (grayscale) or C=3 (RGB).
        colormap (list): List of RGB triplets, e.g., [(243, 231, 132), (211, 222, 111)].
    
    Returns:
        torch.Tensor: Mask of shape (H, W) with integer values corresponding to colormap indices.
    """
    # Ensure colormap is a tensor for efficient comparison
    colormap_tensor = torch.tensor(colormap, dtype=torch.uint8)  # Shape: (N, 3), N = num colors

    # Check input tensor shape
    if img.dim() != 3 or img.shape[0] not in [1, 3]:
        raise ValueError(f"Expected tensor of shape (C, H, W) with C=1 or 3, got {img.shape}")

    # Handle grayscale (C=1) or RGB (C=3)
    if img.shape[0] == 1:
        # Assume grayscale values are normalized [0, 1] or [0, 255]
        # Convert to RGB-like format by repeating the channel
        if img.max() <= 1.0:  # If normalized to [0, 1], scale to [0, 255]
            img_rgb = (img * 255).byte().expand(3, img.shape[1], img.shape[2])
        else:  # Assume [0, 255] range
            img_rgb = img.byte().expand(3, img.shape[1], img.shape[2])
    else:  # C=3, RGB case
        if img.max() <= 1.0:  # If normalized to [0, 1], scale to [0, 255]
            img_rgb = (img * 255).byte()
        else:  # Assume [0, 255] range
            img_rgb = img.byte()

    # Reshape img_rgb to (H*W, 3) for comparison
    H, W = img_rgb.shape[1], img_rgb.shape[2]
    img_flat = img_rgb.permute(1, 2, 0).reshape(-1, 3)  # Shape: (H*W, 3)

    # Initialize mask with -1 (unmatched pixels)
    mask_flat = torch.full((H * W,), -1, dtype=torch.long)

    # Compare each pixel with colormap entries
    for idx, color in enumerate(colormap_tensor):
        # Find pixels matching this color
        matches = (img_flat == color).all(dim=1)
        mask_flat[matches] = idx

    # Reshape to (H, W)
    mask = mask_flat.reshape(H, W)

    # Replace unmatched pixels (-1) with 0 (or another default value)
    mask[mask == -1] = 0

    return mask



def readColormap(path: str) -> list:
    """
    Reads RGB values from a colormap file and returns them as a list of tuples.
    
    Args:
        path (str): Path to the colormap file (e.g., 'colormap.txt').
    
    Returns:
        list: List of RGB tuples, e.g., [(0, 0, 0), (51, 221, 255), (250, 50, 83)].
    """
    colormap = []
    
    # Open and read the file
    with open(path, 'r') as f:
        for line in f:
            # Strip whitespace and split by comma
            r, g, b = map(int, line.strip().split(','))
            # Add the RGB triplet as a tuple
            colormap.append((r, g, b))
    
    return colormap

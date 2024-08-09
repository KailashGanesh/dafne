# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Dafne-Imaging Team

import numpy as np
from scipy.ndimage import binary_erosion, label
from skimage.morphology import area_opening
import matplotlib.pyplot as plt
import time

from ..MedSAM.segment_anything import sam_model_registry, SamPredictor
import torch
import torch.nn.functional as F
from skimage import io, transform
import os
from typing import Callable, Optional
import requests

from ..config import GlobalConfig

CHECKPOINT_MODELS = {
        "Med Sam": { 
            'type': 'vit_b', 
            'file_name': 'medsam_vit_b.pth', 
            'url': 'https://zenodo.org/records/10689643/files/medsam_vit_b.pth',
            'size': 375049145 
        },
        "Sam Large": {
            'type': 'vit_h',
            'file_name': 'sam_vit_h_4b8939.pth',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'size':  2564550879
        },
        "Sam Medium": {
            'type': 'vit_l',
            'file_name': 'sam_vit_l_01ec64.pth',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'size':  1249524607
        },
        "Sam Small": {
            'type': 'vit_b',
            'file_name': 'sam_vit_b_01ec64.pt',
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'size':  375042383
        },
}

image_embedding = None
predictor = None
old_img = None

def process_image(image, device):
    """
    Preprocess and resize the given image.

    Parameters:
    - image (numpy.ndarray): Input image as a NumPy array.
    - device (torch.device): The device to move the tensor to.

    Returns:
    - img_1024_tensor (torch.Tensor): The processed image tensor.
    - H (int): Original height of the image.
    - W (int): Original width of the image.
    """
    # Determine the original height and width
    H, W = image.shape[:2]

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Resize the image to 1024x1024
    img_1024 = transform.resize(
        processed_image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)

    # Normalize the image to [0, 1]
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )

    # Convert the image shape to (3, 1024, 1024)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    return img_1024_tensor, H, W

def preprocess_image(image):
    """
    Preprocess the given image to ensure it is in the correct range and format for display.

    Parameters:
    image (numpy.ndarray): Input image as a NumPy array.

    Returns:
    numpy.ndarray: Preprocessed image suitable for RGB display.
    """
    if image.dtype == np.uint8:
        processed_image = image
    elif image.dtype == np.uint16:
        max_val = image.max()
        processed_image = (image / max_val) * 255.0
        processed_image = processed_image.astype(np.uint8)
    elif image.dtype in [np.int16, np.int32]:
        min_val = image.min()
        max_val = image.max()
        processed_image = ((image - min_val) / (max_val - min_val)) * 255.0
        processed_image = processed_image.astype(np.uint8)
    elif image.dtype in [np.float32, np.float64]:
        if image.max() > 1.0:
            processed_image = image / image.max()
        else:
            processed_image = image
        processed_image = (processed_image * 255.0).astype(np.uint8)
    elif image.dtype == np.bool_:
        processed_image = image.astype(np.uint8) * 255
    else:
        raise ValueError(f"Unsupported image data type: {image.dtype}")

    processed_image = np.clip(processed_image, 0, 255)

    if processed_image.ndim == 2:
        rgb_image = np.stack((processed_image,) * 3, axis=-1)
    elif processed_image.ndim == 3:
        if processed_image.shape[-1] == 3:
            rgb_image = processed_image
        else:
            print(f"Multi-channel image detected with shape {processed_image.shape}. "
                  f"Only the first 3 channels will be used for display.")
            rgb_image = processed_image[..., :3]
    else:
        raise ValueError(f"Unsupported number of dimensions: {processed_image.ndim}")

    return rgb_image

# source: https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)

    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (H, W)
    
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg



def load_sam(model_choice, progress_callback: Optional[Callable[[int, int], None]] = None):
    model_details = CHECKPOINT_MODELS[model_choice]
    checkpoint_path = os.path.join(GlobalConfig['MODEL_PATH'], model_details['file_name'])

    try:
        size = os.path.getsize(checkpoint_path)
    except FileNotFoundError:
        size = 0

    if size != model_details['size']:
        print('Downloading SAM checkpoint...')
        # model needs to be downloaded
        r = requests.get( model_details['url'] , stream=True)
        if r.ok:
            success = True
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            print("Size to download:", total_size_in_bytes)
            block_size = 1024 * 1024  # 1 MB
            current_size = 0
            with open(checkpoint_path, 'wb') as file:
                for data in r.iter_content(block_size):
                    current_size += len(data)
                    if progress_callback is not None:
                        progress_callback(current_size, total_size_in_bytes)
                    file.write(data)

            print("Downloaded size", current_size)
            if current_size != total_size_in_bytes:
                print("Download failed!")
                raise requests.ConnectionError("Error downloading model checkpoint")
    sam = sam_model_registry[model_details['type']](checkpoint=checkpoint_path)

    device = determine_device()
    sam.to(device)
    sam.eval()

    return sam


def dice_score(mask1, mask2):
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)

    # Calculate Dice score
    dice = 2. * intersection.sum() / (mask1_sum + mask2_sum)

    return dice


def enlarge_bounding_box(binary_mask, enlargement_factor=0.2):
    # Find the coordinates of the non-zero elements
    rows, cols = np.where(binary_mask)

    # Determine the bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Calculate the enlargement amount
    height = max_row - min_row
    width = max_col - min_col
    expand_height = int(height * enlargement_factor) // 2
    expand_width = int(width * enlargement_factor) // 2

    # Enlarge the bounding box
    min_row = max(0, min_row - expand_height)
    max_row = min(binary_mask.shape[0], max_row + expand_height)
    min_col = max(0, min_col - expand_width)
    max_col = min(binary_mask.shape[1], max_col + expand_width)

    return np.array([min_col, min_row, max_col, max_row])


def generate_points_from_mask(mask):
    npixel = np.sum(mask)
    accumulated_pixels = np.zeros_like(mask)
    begin_time = time.perf_counter()
    while npixel > 0:
        t = time.perf_counter()
        mask = binary_erosion(mask)
        mask_opened = area_opening(mask, 9)
        isolated_pixels = np.logical_and(mask, np.logical_not(mask_opened))
        if np.any(isolated_pixels):
            accumulated_pixels = np.logical_or(accumulated_pixels, isolated_pixels)
        mask = mask_opened
        npixel = np.sum(mask)
        elapsed_time = time.perf_counter() - t
        # plt.imshow(mask)
        # plt.pause(0.1)

    # Label connected components
    labeled_array, num_features = label(accumulated_pixels)

    # Create an output array initialized to zero
    output_map = np.zeros_like(accumulated_pixels)

    point_list = []

    # Iterate through each connected component
    for component in range(1, num_features + 1):
        # Find the coordinates of the voxels in the current component
        voxels = np.argwhere(labeled_array == component)

        # Select the first voxel (or any other voxel) from the component
        if voxels.size > 0:
            point_list.append([voxels[0][1], voxels[0][0]])

    total_time = time.perf_counter() - begin_time
    # print('Total time', total_time)

    return np.array(point_list)


def determine_device():
    if GlobalConfig['USE_GPU_FOR'] != 'Autosegmentation' and torch.cuda.is_available():
        print('SAM loaded on GPU')
        return 'cuda'
    print('SAM loaded on CPU')
    return 'cpu'


def enhance_mask(img, mask, progress_callback: Optional[Callable[[int, int], None]] = None):
    global old_img, image_embedding, predictor

    # if there is no mask, return the original mask
    if not mask.any():
        return mask

    device = determine_device()
    model_choice = GlobalConfig['SAM_MODEL']

    def show_progress(current, maximum):
        if progress_callback is not None:
            progress_callback(current, maximum)

    show_progress(0, 100)

    print('Loading SAM...')
    sam = load_sam(model_choice, progress_callback)

    show_progress(30, 100)

    if img is not old_img:
        print('Loading image...')
        old_img = img
        img_norm = img * 255 / img.max()
        image_embedding = None
        predictor = None

    bbox = enlarge_bounding_box(mask, GlobalConfig['SAM_BBOX_EXPAND_FACTOR'])
    H, W = img.shape[:2]
    if image_embedding is None:
        img_1024_tensor, H, W = process_image(img, device)
        with torch.no_grad():
            image_embedding = sam.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
        if image_embedding is None:
            raise ValueError("Image embedding is not set. Check the condition that assigns image_embedding.")

    show_progress(80, 100)
    box_1024 = bbox / np.array([W, H, W, H]) * 1024
    box_1024 = box_1024[None, :]  # Ensure shape is (1, 4)
    box_1024 = box_1024[:, None, :]  # Ensure shape is (1, 1, 4)
    masks = medsam_inference(sam, image_embedding, box_1024, H, W)
    torch.cuda.empty_cache()
    return masks
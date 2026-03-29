import cv2
import numpy as np

def alpha_composite(src_frame, bg_frame, mask, feather_k=5):
    """
    Composites foreground onto a background frame.
    
    src_frame: H x W x 3 RGB image (uint8)
    bg_frame: H_bg x W_bg x 3 RGB image (uint8). Process handles resizing.
    mask: H x W soft mask in [0, 1] (float32)
    feather_k: Kernel size for optional spatial feathering of the mask
    Returns: Composited H x W x 3 RGB image (uint8)
    """
    H, W = src_frame.shape[:2]
    
    # Match background dimensions to source sequence
    if bg_frame.shape[:2] != (H, W):
        bg_frame = cv2.resize(bg_frame, (W, H))
        
    # Spatial mask smoothing (feathering the boundary)
    if feather_k > 0:
        kernel_size = feather_k if feather_k % 2 == 1 else feather_k + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
    # Broadcasting dimensions: (H, W) -> (H, W, 1)
    mask_3d = mask[..., np.newaxis]
    
    src_float = src_frame.astype(np.float32)
    bg_float = bg_frame.astype(np.float32)
    
    # Linear blend
    composited = src_float * mask_3d + bg_float * (1.0 - mask_3d)
    
    return np.clip(composited, 0, 255).astype(np.uint8)

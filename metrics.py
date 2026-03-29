import numpy as np
import cv2

def compute_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute Intersection over Union (Jaccard Index) for a single frame.
    pred_mask: HxW float in [0, 1]
    gt_mask: HxW float in [0, 1]
    """
    p = (pred_mask > threshold).astype(bool)
    g = (gt_mask > threshold).astype(bool)
    
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_flicker_score(masks):
    """
    Temporal stability metric based on Mean Absolute Difference between adjacent frames.
    masks: list of HxW float arrays in [0, 1]
    """
    if len(masks) < 2:
        return 0.0
        
    flicker_sum = 0.0
    pixels = masks[0].size
    
    for i in range(1, len(masks)):
        diff = np.abs(masks[i] - masks[i-1])
        flicker_sum += diff.sum() / pixels
        
    return flicker_sum / (len(masks) - 1)

def compute_psnr(pred, target):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) for images.
    pred, target: HxWx3 uint8 numpy arrays
    """
    # ensure float before subtraction to avoid uint8 underflow weirdness
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

import numpy as np
import cv2

def smooth_masks(mask_list, window_size=5, use_median=True):
    """
    Apply temporal smoothing over a list of masks to reduce flicker.
    
    mask_list: list of 2D numpy arrays (H x W)
    window_size: odd integer for kernel size (temporal window)
    use_median: boolean flag to apply median filtering instead of mean
    Returns: smoothed list of masks
    """
    if not mask_list:
        return []
    
    # Structure: T x H x W
    masks_np = np.stack(mask_list, axis=0)
    T, H, W = masks_np.shape
    
    smoothed = np.empty_like(masks_np)
    half_w = window_size // 2
    
    for t in range(T):
        # Determine sliding window boundaries
        start_idx = max(0, t - half_w)
        end_idx = min(T, t + half_w + 1)
        
        windowed_masks = masks_np[start_idx:end_idx]
        
        # Temporal smoothing
        if use_median:
            smoothed_t = np.median(windowed_masks, axis=0)
        else:
            smoothed_t = np.mean(windowed_masks, axis=0)
            
        # Optional spatial smoothing
        # smoothed_t = cv2.GaussianBlur(smoothed_t, (5, 5), 0)
        smoothed[t] = smoothed_t
        
    return list(smoothed)

import os
import numpy as np
import cv2
import torch

from io_utils import read_video, write_video
from segmenter import PersonSegmenter
from temporal_smoothing import smooth_masks
from compositing import alpha_composite
from diffusion_refiner import DiffusionRefiner


def run_baseline(src_video_path, bg_item_path, out_video_path, is_bg_video=False):
    """
    End-to-end baseline pipeline linking all modules.
    """
    print(f"Reading source video: {src_video_path}")
    src_frames, fps = read_video(src_video_path)
    if not src_frames:
        print("Error: Could not read any frames from source.")
        return
        
    # Load background (image or video)
    if is_bg_video:
        bg_frames, _ = read_video(bg_item_path)
    else:
        bg_img = cv2.imread(bg_item_path)
        if bg_img is None:
            # Create dummy bg if fails
            print("Warning: Could not read background. Using green screen.")
            bg_img = np.zeros_like(src_frames[0])
            bg_img[:] = (0, 255, 0)
        else:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_frames = [bg_img] * len(src_frames)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Initialize Segmenter...")
    segmenter = PersonSegmenter(device=device)
    
    print("Segmenting frames...")
    raw_masks = []
    for i, frame in enumerate(src_frames):
        mask = segmenter.segment_frame(frame)
        raw_masks.append(mask)
        
    print("Smoothing masks temporally...")
    smoothed_masks = smooth_masks(raw_masks, window_size=5, use_median=True)
    
    print("Initialize Diffusion Refiner...")
    refiner = DiffusionRefiner(device=device)
    
    print("Compositing and Refining...")
    final_frames = []
    for i, (src, mask) in enumerate(zip(src_frames, smoothed_masks)):
        bg_frame = bg_frames[i % len(bg_frames)]
        
        # 1. Initial alpha composite
        comp = alpha_composite(src, bg_frame, mask, feather_k=5)
        
        # 2. Refine composites via SD
        refined = refiner.refine_composited_frame(comp, strength=0.15)
        
        final_frames.append(refined)
        print(f"Processed frame {i+1}/{len(src_frames)}", end='\r')
        
    print(f"\nWriting output to: {out_video_path}")
    write_video(final_frames, out_video_path, fps)
    print("Done!")

if __name__ == "__main__":
    # Mock / Testing entrypoint for sanity checking code structure
    print("Running baseline sanity checks with dummy data...")
    H, W = 480, 640
    # Dummy mock data
    mock_src_frames = [np.random.randint(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(5)]
    mock_bg = np.ones((H, W, 3), dtype=np.uint8) * 128
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing segmenter...")
    segmenter = PersonSegmenter(device=device)
    mask = segmenter.segment_frame(mock_src_frames[0])
    assert mask.shape == (H, W), f"Expected shape {(H, W)}, got {mask.shape}"
    
    print("Testing temporal smoothing...")
    mock_masks = [np.random.rand(H, W).astype(np.float32) for _ in range(5)]
    smoothed = smooth_masks(mock_masks)
    assert len(smoothed) == 5, f"Expected 5 smoothed masks, got {len(smoothed)}"
    assert smoothed[0].shape == (H, W)
    
    print("Testing compositing...")
    comp = alpha_composite(mock_src_frames[0], mock_bg, mask)
    assert comp.shape == (H, W, 3)
    
    print("Testing refiner...")
    refiner = DiffusionRefiner(device=device)
    refined = refiner.refine_composited_frame(comp)
    assert refined.shape == (H, W, 3)
    
    print("Sanity checks passed! All core module datatypes and shapes are correct.")

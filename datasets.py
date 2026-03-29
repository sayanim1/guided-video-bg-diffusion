import os
import cv2
import glob
import numpy as np

class DavisDatasetHelper:
    """
    Helper to load sequence data conforming roughly to the DAVIS dataset directory structure.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
    def load_sequence(self, sequence_name, resolution='480p'):
        img_dir = os.path.join(self.root_dir, 'JPEGImages', resolution, sequence_name)
        mask_dir = os.path.join(self.root_dir, 'Annotations', resolution, sequence_name)
        
        if not os.path.exists(img_dir):
            print(f"Warning: Image directory {img_dir} not found.")
            return [], []
            
        img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        frames = []
        masks = []
        
        for p in img_paths:
            img = cv2.imread(p)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
            else:
                continue
            
            filename = os.path.splitext(os.path.basename(p))[0]
            mask_path = os.path.join(mask_dir, filename + '.png')
            
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                m = m.astype(np.float32) / 255.0
                masks.append(m)
            else:
                masks.append(np.zeros(img.shape[:2], dtype=np.float32))
                
        return frames, masks


class VideoMatteDatasetHelper:
    """
    Helper to load VideoMatte240K dataset sequences.
    VideoMatte provides foreground (fgr) and alpha matte (pha) sequences.
    """
    def __init__(self, root_dir):
        """
        Expects a directory structure like:
        root_dir/
            fgr/
                sequence_name/
                    0000.jpg ...
            pha/
                sequence_name/
                    0000.png ...
        """
        self.root_dir = root_dir
        
    def load_sequence(self, sequence_name, bg_color=(0, 255, 0)):
        """
        Loads Foreground and Alpha frames. 
        Synthetically composites the foreground onto a solid color background 
        to create a dummy 'Source Video' (Vs) that our pipeline can benchmark against.
        
        Returns:
            synthetic_frames: list of HxWx3 uint8 RGB arrays representing the source video
            masks: list of HxW float32 [0.0, 1.0] ground-truth alpha maps
        """
        fgr_dir = os.path.join(self.root_dir, 'fgr', sequence_name)
        pha_dir = os.path.join(self.root_dir, 'pha', sequence_name)
        
        if not os.path.exists(fgr_dir):
            print(f"Warning: Foreground directory {fgr_dir} not found.")
            return [], []
            
        # Support various image extensions (.jpg, .png, etc.)
        fgr_paths = sorted(glob.glob(os.path.join(fgr_dir, '*.*')))
        pha_paths = sorted(glob.glob(os.path.join(pha_dir, '*.*')))
        
        if len(fgr_paths) != len(pha_paths):
            print(f"Warning: Foreground ({len(fgr_paths)}) and Alpha ({len(pha_paths)}) counts mismatch!")
        
        frames = []
        masks = []
        
        for f_path, p_path in zip(fgr_paths, pha_paths):
            # Read Foreground
            fgr = cv2.imread(f_path)
            if fgr is None:
                continue
            fgr = cv2.cvtColor(fgr, cv2.COLOR_BGR2RGB)
            
            # Read Alpha Matte
            pha = cv2.imread(p_path, cv2.IMREAD_GRAYSCALE)
            if pha is None:
                continue
            pha = pha.astype(np.float32) / 255.0
            
            # Create a synthetic Source Video (Vs) by compositing the VideoMatte foreground 
            # over a solid green background. Our main pipeline will then try to remove the green 
            # background and test the IoU/PSNR against the new target background!
            bg = np.zeros_like(fgr)
            bg[:] = bg_color
            
            p_3d = pha[..., np.newaxis]
            synthetic_src = (fgr.astype(np.float32) * p_3d + bg.astype(np.float32) * (1.0 - p_3d))
            synthetic_src = np.clip(synthetic_src, 0, 255).astype(np.uint8)
            
            frames.append(synthetic_src)
            masks.append(pha)
            
        return frames, masks

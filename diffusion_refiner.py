import torch
from PIL import Image
import numpy as np

try:
    from diffusers import AutoPipelineForImage2Image
except ImportError:
    AutoPipelineForImage2Image = None


class DiffusionRefiner:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device='cpu'):
        """
        Wraps a diffusion model for per-frame bounding/lighting refinement.
        We default to a relatively lightweight standard model for demonstration.
        """
        self.device = device
        if AutoPipelineForImage2Image is not None:
            dtype = torch.float16 if 'cuda' in device else torch.float32
            
            # Use diffusers pipeline with img2img capability
            self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                requires_safety_checker=False
            )
            self.pipeline.to(device)
            # Optimization hooks
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            self.pipeline = None
            print("Warning: `diffusers` library not found. Diffusion refinement is mocked.")

    def refine_composited_frame(self, comp_frame, prompt="", strength=0.2, guidance_scale=7.5):
        """
        Takes a raw linear composite and uses low-strength img2img to unify 
        lighting and edge artifacts.
        
        comp_frame: H x W x 3 RGB numpy array
        strength: Lower values (e.g. 0.1-0.2) keep the result structurally close to original
        Returns: Refined H x W x 3 RGB numpy array
        """
        if self.pipeline is None:
            return comp_frame
            
        init_image = Image.fromarray(comp_frame)
        
        # A sensible default prompt focusing on realism
        final_prompt = prompt if prompt else \
            "high quality photography of a person blended harmoniously into the modern background, highly detailed, photorealistic lighting"
            
        # Predict 
        result = self.pipeline(
            prompt=final_prompt,
            image=init_image,
            strength=strength, 
            guidance_scale=guidance_scale
        ).images[0]
        
        res_frame = np.array(result)
        
        # Ensure dimensions match requested output, SD changes dimensions if not multiple of 8
        if res_frame.shape[:2] != comp_frame.shape[:2]:
            import cv2
            res_frame = cv2.resize(res_frame, (comp_frame.shape[1], comp_frame.shape[0]))
            
        return res_frame

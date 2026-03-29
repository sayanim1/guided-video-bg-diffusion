import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import numpy as np

class PersonSegmenter:
    def __init__(self, device='cpu'):
        """
        Initializes the model on the specified device.
        """
        self.device = device
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.transform = weights.transforms()
        self.model = deeplabv3_resnet101(weights=weights).to(device)
        self.model.eval()

    @torch.no_grad()
    def segment_frame(self, frame_np):
        """
        Segments the primary person in the given frame.
        frame_np: H x W x 3 RGB image, uint8
        Returns: H x W soft mask in [0, 1], float32
        """
        # Convert to tensor and add batch dimension (N, C, H, W)
        img_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
        
        # Apply standard torchvision transforms
        input_tensor = self.transform(img_tensor).unsqueeze(0).to(self.device)
        
        out = self.model(input_tensor)['out'][0]
        
        # DeepLabV3 output is num_classes x H x W
        # The 'person' class is index 15 in the COCO dataset
        probs = torch.softmax(out, dim=0)
        person_prob = probs[15].cpu().numpy()
        
        # DeepLab interpolation might change dimensions if input had specific aspect ratios,
        # but typically the transform maintains them or resizes. We resize back to original H_org, W_org just in case.
        H_org, W_org = frame_np.shape[:2]
        if person_prob.shape != (H_org, W_org):
            person_prob = cv2.resize(person_prob, (W_org, H_org))
            
        return person_prob

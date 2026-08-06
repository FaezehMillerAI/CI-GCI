import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class VQARadCausalDataset(Dataset):
    def __init__(self, json_path, img_dir, img_size=(224, 224)):
        self.img_dir = img_dir
        self.img_size = img_size
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.image_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.data)
        
    def _generate_organ_mask(self, organ):
        """
        Generates a physics-informed anatomical template mask when pixel segmentations 
        are not present (e.g. VQA-RAD).
        """
        mask = np.zeros(self.img_size, dtype=np.float32)
        H, W = self.img_size
        organ = str(organ).upper()
        
        if "HEAD" in organ or "BRAIN" in organ:
            # Ellipse mask centered in the brain region
            y, x = np.ogrid[:H, :W]
            center_y, center_x = H // 2, W // 2
            rx, ry = W // 3, H // 3
            mask[((x - center_x)/rx)**2 + ((y - center_y)/ry)**2 <= 1] = 1.0
        elif "CHEST" in organ or "LUNG" in organ:
            # Left and right lung bounding boxes
            # Left Lung: x: [0.15, 0.45], y: [0.2, 0.8]
            # Right Lung: x: [0.55, 0.85], y: [0.2, 0.8]
            mask[int(H*0.2):int(H*0.8), int(W*0.15):int(W*0.45)] = 1.0
            mask[int(H*0.2):int(H*0.8), int(W*0.55):int(W*0.85)] = 1.0
        elif "ABDOMEN" in organ:
            # Central abdominal region
            mask[int(H*0.3):int(H*0.8), int(W*0.25):int(W*0.75)] = 1.0
        else:
            # Default center mask
            mask[int(H*0.25):int(H*0.75), int(W*0.25):int(W*0.75)] = 1.0
            
        return torch.from_numpy(mask).unsqueeze(0) # (1, H, W)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item["image_name"]
        
        # Path
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Fallback black placeholder
            image = Image.new('RGB', self.img_size, color='black')
            
        img_tensor = self.image_transform(image)
        
        # Generate anatomical mask based on the organ field
        organ = item.get("image_organ", "chest")
        target_mask = self._generate_organ_mask(organ)
        
        return {
            "image": img_tensor,
            "mask": target_mask,
            "question": item["question"],
            "answer": item["answer"],
            "location": organ,
            "answer_type": item.get("answer_type", "OPEN"),
            "id": item.get("qid", str(idx))
        }

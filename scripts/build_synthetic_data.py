import os
import json
import random
import argparse
import numpy as np
from PIL import Image

def generate_synthetic_data(num_samples: int, output_dir: str):
    """Generate synthetic images and metadata for VQA RAD/SLAKE replication."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    modalities = ["Chest X-Ray", "Brain MRI", "CT Abdomen"]
    questions_pool = {
        "Chest X-Ray": [
            {
                "question": "Is there evidence of pleural effusion?",
                "answer": "yes",
                "question_type": "existence",
                "level_1": ["Is there fluid in the pleural cavity?", "Where is the costophrenic angle located?"],
                "level_2": ["Is the pleural line thickened?", "Is the fluid accumulation unilateral?"],
                "level_3": ["Does this pattern indicate acute pleural effusion?"]
            },
            {
                "question": "Are the lungs clear?",
                "answer": "no",
                "question_type": "attribute",
                "level_1": ["Are there opacities in either lung field?", "Where are the lung fields located?"],
                "level_2": ["Is the opacity located in the right lower lobe?", "Is the infiltration diffuse?"],
                "level_3": ["Do these opacities suggest multi-lobar pneumonia?"]
            }
        ],
        "Brain MRI": [
            {
                "question": "Is there a lesion in the right hemisphere?",
                "answer": "yes",
                "question_type": "localization",
                "level_1": ["Is there an abnormal mass or hyperintensity?", "Where is the right hemisphere?"],
                "level_2": ["Is the lesion hyperintense on T2?", "Is the border of the lesion well-defined?"],
                "level_3": ["Is the lesion consistent with a low-grade glioma?"]
            }
        ],
        "CT Abdomen": [
            {
                "question": "Is the appendix normal?",
                "answer": "no",
                "question_type": "attribute",
                "level_1": ["Is the appendix visible in the right lower quadrant?", "Where is the caecum located?"],
                "level_2": ["Is the appendix diameter greater than 6mm?", "Is there surrounding fat stranding?"],
                "level_3": ["Does this suggest acute appendicitis?"]
            }
        ]
    }
    
    dataset = []
    
    for i in range(num_samples):
        # Create a random dummy image
        img_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        # Add some shapes to make it look like a medical scan
        center_x, center_y = random.randint(50, 170), random.randint(50, 170)
        radius = random.randint(10, 30)
        # Simple drawn circle for localization
        for y in range(224):
            for x in range(224):
                if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                    img_arr[y, x, :] = random.choice([200, 50]) # bright or dark spot
        
        img_name = f"sample_{i}.png"
        img_path = os.path.join(output_dir, "images", img_name)
        img = Image.fromarray(img_arr)
        img.save(img_path)
        
        # Pick modality and VQA pair
        mod = random.choice(modalities)
        vqa_template = random.choice(questions_pool[mod])
        
        # Determine train/val/test split
        r = random.random()
        if r < 0.7:
            split = "train"
        elif r < 0.85:
            split = "val"
        else:
            split = "test"
            
        sample = {
            "id": f"SYN_{i:04d}",
            "image_path": os.path.abspath(img_path),
            "question": vqa_template["question"],
            "answer": vqa_template["answer"],
            "question_type": vqa_template["question_type"],
            "dataset": "synthetic",
            "modality": mod,
            "split": split,
            "curriculum": {
                "level_1": vqa_template["level_1"],
                "level_2": vqa_template["level_2"],
                "level_3": vqa_template["level_3"]
            },
            # Answers to curriculum questions (can be yes/no or descriptive)
            "curriculum_answers": {
                "level_1": [random.choice(["yes", "no"]) for _ in vqa_template["level_1"]],
                "level_2": [random.choice(["yes", "no"]) for _ in vqa_template["level_2"]],
                "level_3": [random.choice(["yes", "no"]) for _ in vqa_template["level_3"]]
            },
            # Grounding bounding box (mock)
            "grounding_box": [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            "hallucinated": random.choice([0, 1]) # 0 = faithful, 1 = hallucinated
        }
        dataset.append(sample)
        
    metadata_path = os.path.join(output_dir, "dataset.json")
    with open(metadata_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"[Data] Generated {num_samples} synthetic samples in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="data/processed/synthetic")
    args = parser.parse_args()
    generate_synthetic_data(args.num_samples, args.output_dir)
